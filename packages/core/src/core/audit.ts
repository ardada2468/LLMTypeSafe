import type { LLMCallOptions } from '../types/language-model';
import type { Callback, CallbackEndBase } from './callbacks';
import { isThenable } from './callbacks';
import { sumUsage, type LMUsage } from './usage';

/** One line of the audit trail. Every field is JSON-serialisable. */
export interface AuditRecord {
    /** ISO-8601 time the invocation finished. */
    timestamp: string;
    event: 'module' | 'lm' | 'tool';
    /** Unique per invocation; `parentCallId` links it to what called it. */
    callId: string;
    parentCallId?: string;
    durationMs: number;
    status: 'ok' | 'error';
    /** The error's class name, when `status` is `error`. */
    errorType?: string;
    /** The provider's HTTP status, when the error carried one. */
    errorStatus?: number;
    /** `Predict#3`, for module and LM records. */
    moduleId?: string;
    /** `openai/gpt-5.2`-style keys of every model the invocation used. */
    models?: string[];
    /** Which model method ran, for LM records. */
    method?: string;
    toolName?: string;
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
    cachedTokens?: number;
    cost?: number;
    /**
     * The call's `options.metadata` — where a caller attaches who the request
     * was for (`userId`, `tenantId`, `requestId`). LM records only.
     */
    metadata?: Record<string, unknown>;
    /** Inputs and outputs, only with `includeContent`. */
    inputs?: unknown;
    outputs?: unknown;
    /** The error message, only with `includeContent`: it can quote model output. */
    errorMessage?: string;
}

export interface AuditLoggerOptions {
    /**
     * Where records go: a function, or anything with a `write(string)` method
     * (a file stream, `process.stdout`), which receives one JSON line per
     * record. A sink that throws or rejects never fails the program.
     */
    sink: ((record: AuditRecord) => void | Promise<void>) | { write(line: string): unknown };
    /**
     * Record inputs, outputs and error messages. Off by default: they
     * routinely hold personal data, and an audit trail usually outlives it.
     */
    includeContent?: boolean;
    /** Which events to record. Defaults to all three. */
    events?: Array<AuditRecord['event']>;
    /** Fields merged into every record: service name, environment, region. */
    base?: Record<string, unknown>;
}

function usageFields(usage: LMUsage | undefined): Partial<AuditRecord> {
    if (!usage) return {};
    const models = Object.keys(usage);
    if (models.length === 0) return {};
    const total = sumUsage(Object.values(usage));
    return {
        models,
        promptTokens: total.promptTokens,
        completionTokens: total.completionTokens,
        totalTokens: total.totalTokens,
        ...(total.cachedTokens !== undefined ? { cachedTokens: total.cachedTokens } : {}),
        ...(total.totalCost !== undefined ? { cost: total.totalCost } : {}),
    };
}

function outcome(event: CallbackEndBase, includeContent: boolean): Partial<AuditRecord> {
    if (event.error === undefined) return { status: 'ok' };
    const error = event.error;
    const status = (error as { status?: unknown } | null)?.status;
    return {
        status: 'error',
        errorType: error instanceof Error ? error.constructor.name || error.name : typeof error,
        ...(typeof status === 'number' ? { errorStatus: status } : {}),
        ...(includeContent
            ? { errorMessage: error instanceof Error ? error.message : String(error) }
            : {}),
    };
}

/** Values that survive `JSON.stringify` without throwing on cycles or BigInt. */
function safe(value: unknown): unknown {
    try {
        return JSON.parse(JSON.stringify(value));
    } catch {
        return String(value);
    }
}

/**
 * A callback that writes one structured record per module call, model call
 * and tool call: who ran what, when, for how long, on which model, at what
 * token cost, and whether it failed — the trail an auditor or a cost report
 * needs, without the prompts unless you ask for them.
 *
 * ```ts
 * import { createWriteStream } from 'node:fs';
 *
 * configure({ callbacks: [auditLogger({ sink: createWriteStream('audit.jsonl', { flags: 'a' }) })] });
 * await program.forward(inputs, { metadata: { userId: 'u_42', requestId } });
 * ```
 *
 * `options.metadata` on a call is copied onto its LM records, which is how a
 * record is attributed to a user, tenant or request.
 */
export function auditLogger(options: AuditLoggerOptions): Callback {
    const { sink, includeContent = false, base } = options;
    const wanted = new Set(options.events ?? ['module', 'lm', 'tool']);
    // `onLMEnd` does not carry the call's options, so they are held from start.
    const metadataByCall = new Map<string, Record<string, unknown>>();
    const inputsByCall = new Map<string, unknown>();

    const write = (record: AuditRecord) => {
        const full = base ? { ...base, ...record } : record;
        try {
            const returned: unknown =
                typeof sink === 'function'
                    ? sink(full as AuditRecord)
                    : sink.write(`${JSON.stringify(full)}\n`);
            if (isThenable(returned)) returned.then(undefined, () => {});
        } catch {
            // The audit trail must never fail the work it describes.
        }
    };

    const common = (event: CallbackEndBase, includeInputs = true) => {
        const inputs = inputsByCall.get(event.callId);
        inputsByCall.delete(event.callId);
        return {
            timestamp: new Date(event.endTime).toISOString(),
            callId: event.callId,
            ...(event.parentCallId ? { parentCallId: event.parentCallId } : {}),
            durationMs: event.durationMs,
            ...outcome(event, includeContent),
            ...(includeContent && includeInputs && inputs !== undefined ? { inputs } : {}),
        } as Pick<AuditRecord, 'timestamp' | 'callId' | 'durationMs' | 'status'>;
    };

    const callback: Callback = {};

    if (wanted.has('module')) {
        if (includeContent) {
            callback.onModuleStart = (event) => {
                inputsByCall.set(event.callId, safe(event.inputs));
            };
        }
        callback.onModuleEnd = (event) => {
            write({
                ...common(event),
                event: 'module',
                moduleId: event.moduleId,
                ...usageFields(event.usage),
                ...(includeContent && event.outputs !== undefined
                    ? { outputs: safe(event.outputs) }
                    : {}),
            });
        };
    }

    if (wanted.has('lm')) {
        callback.onLMStart = (event) => {
            const metadata = (event.options as LLMCallOptions | undefined)?.metadata;
            if (metadata && Object.keys(metadata).length > 0) {
                metadataByCall.set(event.callId, safe(metadata) as Record<string, unknown>);
            }
            if (includeContent) {
                inputsByCall.set(event.callId, safe(event.messages ?? event.prompt));
            }
        };
        callback.onLMEnd = (event) => {
            const metadata = metadataByCall.get(event.callId);
            metadataByCall.delete(event.callId);
            const usage = usageFields(event.usage);
            write({
                ...common(event),
                event: 'lm',
                method: event.method,
                ...(event.moduleId ? { moduleId: event.moduleId } : {}),
                ...usage,
                // A call that failed before the provider answered has no usage
                // to key on; the model name is still worth recording.
                ...(usage.models ? {} : { models: [event.model] }),
                ...(metadata ? { metadata } : {}),
                ...(includeContent && event.outputs !== undefined
                    ? { outputs: safe(event.outputs) }
                    : {}),
            });
        };
    }

    if (wanted.has('tool')) {
        if (includeContent) {
            callback.onToolStart = (event) => {
                inputsByCall.set(event.callId, safe(event.args));
            };
        }
        callback.onToolEnd = (event) => {
            write({
                ...common(event),
                event: 'tool',
                toolName: event.toolName,
                ...(includeContent && event.outputs !== undefined
                    ? { outputs: safe(event.outputs) }
                    : {}),
            });
        };
    }

    return callback;
}
