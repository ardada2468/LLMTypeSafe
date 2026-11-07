import {
    configure,
    getCallbacks,
    VERSION,
    type Callback,
    type CallbackEventBase,
    type LLMCallOptions,
    type LMEndEvent,
    type LMStartEvent,
    type LMUsage,
    type ModuleEndEvent,
    type ModuleStartEvent,
    type ToolEndEvent,
    type ToolStartEvent,
} from '@ts-dspy/core';
import {
    context,
    SpanKind,
    SpanStatusCode,
    trace,
    type Attributes,
    type Context,
    type Span,
    type Tracer,
} from '@opentelemetry/api';

/** The instrumentation scope name spans are reported under. */
export const INSTRUMENTATION_NAME = '@ts-dspy/otel';

export interface OpenTelemetryCallbackOptions {
    /**
     * Tracer to create spans with. Defaults to
     * `trace.getTracer('@ts-dspy/otel')`, which resolves through whatever
     * `TracerProvider` is registered globally — the usual setup with
     * `@opentelemetry/sdk-node`.
     */
    tracer?: Tracer;
    /**
     * Record prompts, completions, module inputs/outputs and tool arguments/
     * results on spans. Off by default: they routinely contain personal data
     * and secrets, and a trace backend is rarely the place either should end
     * up without someone deciding so.
     */
    captureContent?: boolean;
}

/**
 * Reports a ts-dspy program's execution as OpenTelemetry spans.
 *
 * Every module invocation becomes a `ts-dspy.module <Class>` span, every
 * language-model call a `chat <model>` span, and every tool call an
 * `execute_tool <name>` span, nested exactly as the calls were. LM and tool
 * spans follow the OpenTelemetry GenAI semantic conventions
 * (`gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`,
 * `gen_ai.usage.input_tokens`, …), so GenAI-aware backends render them
 * without configuration.
 *
 * **Why parenting goes through `parentCallId`.** OpenTelemetry's own context
 * propagation only parents a span correctly when the code creating it runs
 * inside the parent's context, which would mean wrapping every module call
 * in `context.with(...)`. ts-dspy already tracks the logical call stack with
 * `AsyncLocalStorage` and hands each event the id of the call it ran inside,
 * so spans are parented from that instead: correct under any amount of
 * concurrency, with or without a context manager registered. A top-level
 * call — one with no ts-dspy parent — is parented to whatever span is active
 * in OpenTelemetry's context, so an HTTP request span around a program still
 * becomes its root.
 *
 * Register it with `configure({ callbacks: [new OpenTelemetryCallback()] })`,
 * or with {@link instrument}.
 */
export class OpenTelemetryCallback implements Callback {
    private readonly tracer: Tracer;
    private readonly captureContent: boolean;
    /** Open spans by `callId`, so children can find their parent and ends their start. */
    private readonly spans = new Map<string, Span>();
    /** LM calls whose provider was unknown at start, to fill in from usage at the end. */
    private readonly providerless = new Set<string>();

    constructor(options: OpenTelemetryCallbackOptions = {}) {
        this.tracer = options.tracer ?? trace.getTracer(INSTRUMENTATION_NAME, VERSION);
        this.captureContent = options.captureContent ?? false;
    }

    onModuleStart(event: ModuleStartEvent): void {
        const className = moduleClassName(event);
        const attributes: Attributes = {
            'ts-dspy.module.id': event.moduleId,
            'ts-dspy.module.class': className,
        };
        if (this.captureContent) attributes['ts-dspy.module.inputs'] = toJson(event.inputs);
        this.open(event, `ts-dspy.module ${className}`, SpanKind.INTERNAL, attributes);
    }

    onModuleEnd(event: ModuleEndEvent): void {
        this.close(event, (span) => {
            span.setAttributes(usageAttributes(event.usage, 'ts-dspy.usage'));
            if (this.captureContent && event.outputs !== undefined) {
                span.setAttribute('ts-dspy.module.outputs', toJson(event.outputs));
            }
        });
    }

    onLMStart(event: LMStartEvent): void {
        const model = event.options?.model ?? event.model;
        const provider = providerOf(event.lm);
        const attributes: Attributes = {
            'gen_ai.operation.name': 'chat',
            'gen_ai.request.model': model,
            'ts-dspy.lm.method': event.method,
            ...requestAttributes(event.options),
        };
        if (provider) {
            attributes['gen_ai.provider.name'] = provider;
            // The pre-1.37 name for the same thing, still the one many
            // backends key on.
            attributes['gen_ai.system'] = provider;
        } else {
            this.providerless.add(event.callId);
        }
        if (event.moduleId) attributes['ts-dspy.module.id'] = event.moduleId;
        if (event.method === 'generateStructured') attributes['gen_ai.output.type'] = 'json';
        if (this.captureContent) {
            attributes['gen_ai.input.messages'] = toJson(
                event.messages ?? [{ role: 'user', content: event.prompt ?? '' }],
                redactImages
            );
        }
        this.open(event, `chat ${model}`, SpanKind.CLIENT, attributes);
    }

    onLMEnd(event: LMEndEvent): void {
        const providerless = this.providerless.delete(event.callId);
        this.close(event, (span) => {
            span.setAttributes(usageAttributes(event.usage, 'gen_ai.usage'));
            const key = soleUsageKey(event.usage);
            if (key) {
                span.setAttribute('gen_ai.response.model', key.model);
                if (providerless) {
                    span.setAttribute('gen_ai.provider.name', key.provider);
                    span.setAttribute('gen_ai.system', key.provider);
                }
            }

            const finishReason = finishReasonOf(event.outputs);
            if (finishReason)
                span.setAttribute('gen_ai.response.finish_reasons', [finishReason]);

            if (this.captureContent && event.outputs !== undefined) {
                span.setAttribute(
                    'gen_ai.output.messages',
                    toJson(outputMessages(event.outputs))
                );
            }
        });
    }

    onToolStart(event: ToolStartEvent): void {
        const attributes: Attributes = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': event.toolName,
            'gen_ai.tool.type': 'function',
        };
        if (this.captureContent) attributes['gen_ai.tool.call.arguments'] = toJson(event.args);
        this.open(event, `execute_tool ${event.toolName}`, SpanKind.INTERNAL, attributes);
    }

    onToolEnd(event: ToolEndEvent): void {
        this.close(event, (span) => {
            if (this.captureContent && event.outputs !== undefined) {
                span.setAttribute('gen_ai.tool.call.result', toJson(event.outputs));
            }
        });
    }

    private open(
        event: CallbackEventBase,
        name: string,
        kind: SpanKind,
        attributes: Attributes
    ): void {
        const span = this.tracer.startSpan(
            name,
            { kind, attributes, startTime: event.startTime },
            this.parentContext(event.parentCallId)
        );
        this.spans.set(event.callId, span);
    }

    private close(
        event: CallbackEventBase & { endTime: number; error?: unknown },
        annotate: (span: Span) => void
    ): void {
        const span = this.spans.get(event.callId);
        if (!span) return;
        this.spans.delete(event.callId);

        try {
            if (event.error !== undefined) {
                recordError(span, event.error);
            } else {
                annotate(span);
            }
        } finally {
            span.end(event.endTime);
        }
    }

    private parentContext(parentCallId: string | undefined): Context {
        const parent = parentCallId ? this.spans.get(parentCallId) : undefined;
        return parent ? trace.setSpan(context.active(), parent) : context.active();
    }
}

export type InstrumentOptions = OpenTelemetryCallbackOptions;

/**
 * Register an {@link OpenTelemetryCallback} globally, alongside any callbacks
 * already configured, and return it.
 *
 * ```ts
 * import { instrument } from '@ts-dspy/otel';
 * instrument(); // after the OpenTelemetry SDK has started
 * ```
 *
 * Calling it again replaces the previous instance rather than adding a second
 * one, which would report every span twice. Undo it with {@link uninstrument}.
 */
export function instrument(options: InstrumentOptions = {}): OpenTelemetryCallback {
    const callback = new OpenTelemetryCallback(options);
    configure({
        callbacks: [
            ...getCallbacks().filter(
                (existing) => !(existing instanceof OpenTelemetryCallback)
            ),
            callback,
        ],
    });
    return callback;
}

/** Remove what {@link instrument} registered, leaving other callbacks in place. */
export function uninstrument(): void {
    configure({
        callbacks: getCallbacks().filter(
            (existing) => !(existing instanceof OpenTelemetryCallback)
        ),
    });
}

function moduleClassName(event: ModuleStartEvent): string {
    const name = (event.instance as { constructor?: { name?: string } }).constructor?.name;
    // `moduleId` is `<Class>#<n>`; the class alone keeps span names low-cardinality.
    return name || event.moduleId.replace(/#\d+$/, '');
}

/**
 * The provider name. `BaseLM` keeps it in a protected `provider` field; a
 * hand-written `ILanguageModel` may have none, and the attribute is then
 * filled from the usage key when the call ends, if at all.
 */
function providerOf(lm: unknown): string | undefined {
    const provider = (lm as { provider?: unknown }).provider;
    return typeof provider === 'string' && provider.length > 0 ? provider : undefined;
}

function requestAttributes(options: LLMCallOptions | undefined): Attributes {
    const attributes: Attributes = {};
    if (!options) return attributes;
    if (options.temperature !== undefined) {
        attributes['gen_ai.request.temperature'] = options.temperature;
    }
    if (options.maxTokens !== undefined)
        attributes['gen_ai.request.max_tokens'] = options.maxTokens;
    if (options.topP !== undefined) attributes['gen_ai.request.top_p'] = options.topP;
    if (options.frequencyPenalty !== undefined) {
        attributes['gen_ai.request.frequency_penalty'] = options.frequencyPenalty;
    }
    if (options.presencePenalty !== undefined) {
        attributes['gen_ai.request.presence_penalty'] = options.presencePenalty;
    }
    if (options.stopSequences?.length) {
        attributes['gen_ai.request.stop_sequences'] = options.stopSequences;
    }
    return attributes;
}

/**
 * Token counts summed across the models in a usage map. An LM call touches
 * one model (two under a fallback), and a module span reports every call made
 * beneath it.
 */
function usageAttributes(usage: LMUsage, prefix: string): Attributes {
    const entries = Object.values(usage ?? {});
    if (entries.length === 0) return {};
    let input = 0;
    let output = 0;
    let cached = 0;
    let reasoning = 0;
    for (const stats of entries) {
        input += stats.promptTokens;
        output += stats.completionTokens;
        cached += stats.cachedTokens ?? 0;
        reasoning += stats.reasoningTokens ?? 0;
    }
    const attributes: Attributes = {
        [`${prefix}.input_tokens`]: input,
        [`${prefix}.output_tokens`]: output,
    };
    if (cached > 0) attributes[`${prefix}.cache_read.input_tokens`] = cached;
    if (reasoning > 0) attributes[`${prefix}.reasoning.output_tokens`] = reasoning;
    return attributes;
}

/**
 * The provider and model that actually answered, from the `provider/model`
 * usage key — present only when exactly one model was used.
 */
function soleUsageKey(usage: LMUsage): { provider: string; model: string } | undefined {
    const keys = Object.keys(usage ?? {});
    if (keys.length !== 1) return undefined;
    const slash = keys[0].indexOf('/');
    if (slash <= 0) return undefined;
    return { provider: keys[0].slice(0, slash), model: keys[0].slice(slash + 1) };
}

/**
 * Only `chatWithTools` reports why the model stopped; the text methods
 * return a bare string, and a reason invented for them would be a guess.
 */
function finishReasonOf(outputs: unknown): string | undefined {
    if (typeof outputs !== 'object' || outputs === null) return undefined;
    const reason = (outputs as { finishReason?: unknown }).finishReason;
    return typeof reason === 'string' ? reason : undefined;
}

function outputMessages(outputs: unknown): unknown[] {
    if (typeof outputs === 'string') return [{ role: 'assistant', content: outputs }];
    if (typeof outputs === 'object' && outputs !== null && 'content' in outputs) {
        const result = outputs as { content: unknown; toolCalls?: unknown };
        return [
            {
                role: 'assistant',
                content: result.content,
                ...(result.toolCalls ? { toolCalls: result.toolCalls } : {}),
            },
        ];
    }
    // A structured result: the parsed object itself is the completion.
    return [{ role: 'assistant', content: outputs }];
}

function recordError(span: Span, error: unknown): void {
    if (error instanceof Error) {
        span.recordException(error);
        span.setAttribute('error.type', error.constructor.name || error.name);
    } else {
        span.recordException(String(error));
        span.setAttribute('error.type', '_OTHER');
    }
    span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : String(error),
    });
}

/**
 * Image payloads are routinely megabytes of base64, which no trace backend
 * wants as an attribute; keep that an image was sent and its type.
 */
function redactImages(key: string, value: unknown): unknown {
    if (key === 'source' && typeof value === 'object' && value !== null) {
        const source = value as { kind?: string; url?: string; mediaType?: string };
        if (source.kind === 'base64') {
            return { kind: 'base64', mediaType: source.mediaType, data: '[omitted]' };
        }
        if (source.kind === 'url' && source.url?.startsWith('data:')) {
            return { kind: 'url', mediaType: source.mediaType, url: '[data URI omitted]' };
        }
    }
    return value;
}

function toJson(value: unknown, replacer?: (key: string, value: unknown) => unknown): string {
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value, replacer) ?? String(value);
    } catch {
        // Circular structures and BigInts: fall back rather than lose the span.
        return String(value);
    }
}
