import { type ILanguageModel, type UsageStats } from '../types/language-model';
import { type LMCall, type TraceEntry } from '../types/module';
import { getTraceHandler, getTraceHistorySize, isTracingEnabled } from './config';

/**
 * Recorded traces, oldest first. Bounded by `configure({ traceHistorySize })`;
 * the oldest entries are dropped once it is full, so a long-running process
 * cannot grow this without limit.
 */
const history: TraceEntry[] = [];

let moduleSequence = 0;

/** Allocate a stable identifier for a module instance, e.g. `Predict#3`. */
export function nextModuleId(name: string): string {
    moduleSequence += 1;
    return `${name}#${moduleSequence}`;
}

/**
 * Difference two cumulative usage snapshots. `getUsage()` counts from the
 * lifetime of the language model, so a per-call figure is the delta between a
 * snapshot taken before the call and one taken after. Values are clamped at
 * zero in case `resetUsage()` ran mid-flight.
 */
function diffUsage(before: UsageStats, after: UsageStats): UsageStats {
    const delta = (later: number | undefined, earlier: number | undefined): number =>
        Math.max(0, (later ?? 0) - (earlier ?? 0));

    const usage: UsageStats = {
        promptTokens: delta(after.promptTokens, before.promptTokens),
        completionTokens: delta(after.completionTokens, before.completionTokens),
        totalTokens: delta(after.totalTokens, before.totalTokens),
        requestCount: delta(after.requestCount, before.requestCount),
        errorCount: delta(after.errorCount, before.errorCount),
    };

    if (after.totalCost !== undefined) {
        usage.totalCost = delta(after.totalCost, before.totalCost);
    }

    return usage;
}

/** Total the per-call figures, so an entry agrees with its own `calls`. */
function sumUsage(calls: LMCall[]): UsageStats {
    const total: UsageStats = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        requestCount: 0,
        errorCount: 0,
    };

    for (const call of calls) {
        total.promptTokens += call.usage.promptTokens;
        total.completionTokens += call.usage.completionTokens;
        total.totalTokens += call.usage.totalTokens;
        total.requestCount = (total.requestCount ?? 0) + (call.usage.requestCount ?? 0);
        total.errorCount = (total.errorCount ?? 0) + (call.usage.errorCount ?? 0);
        if (call.usage.totalCost !== undefined) {
            total.totalCost = (total.totalCost ?? 0) + call.usage.totalCost;
        }
    }

    return total;
}

/**
 * One module invocation in progress. Created only when tracing is enabled, so
 * nothing here runs — and nothing is allocated — on the default path.
 *
 * Callers bracket each language-model call with {@link startCall} and
 * {@link endCall}. A call left open — because the provider threw — is still
 * recorded, with an empty reply: the prompt behind a failure is the whole point.
 */
export class TraceSpan {
    private readonly moduleId: string;
    private readonly lm: ILanguageModel;
    private readonly input: Record<string, any>;
    private readonly timestamp = new Date();
    private readonly startedAt = Date.now();
    private readonly calls: LMCall[] = [];

    private pending?: { prompt: string; startedAt: number; usageBefore: UsageStats };

    constructor(moduleId: string, lm: ILanguageModel, input: Record<string, any>) {
        this.moduleId = moduleId;
        this.lm = lm;
        // Shallow copy: enough to survive the caller reassigning a field, not a
        // defence against mutation deeper inside a nested input object.
        this.input = { ...input };
    }

    /** Note the prompt about to be sent. Pairs with {@link endCall}. */
    startCall(prompt: string): void {
        this.closePending('');
        this.pending = {
            prompt,
            startedAt: Date.now(),
            // Copy: a provider is free to hand back its own mutable counters.
            usageBefore: { ...this.lm.getUsage() },
        };
    }

    /** Close the open call with the provider's reply, verbatim. */
    endCall(rawOutput: string): void {
        this.closePending(rawOutput);
    }

    /** Close the span for a successful invocation and record it. */
    finish(output: Record<string, any>): TraceEntry {
        return record(this.build(output));
    }

    /** Close the span for a failed invocation and record it. */
    fail(error: unknown): TraceEntry {
        const entry = this.build({});
        entry.error = error;
        return record(entry);
    }

    private closePending(rawOutput: string): void {
        const pending = this.pending;
        if (!pending) {
            return;
        }
        this.pending = undefined;

        this.calls.push({
            prompt: pending.prompt,
            rawOutput,
            duration: Date.now() - pending.startedAt,
            usage: diffUsage(pending.usageBefore, { ...this.lm.getUsage() }),
        });
    }

    private build(output: Record<string, any>): TraceEntry {
        // An open call means the provider threw. Keep its prompt.
        this.closePending('');
        const last = this.calls.at(-1);

        return {
            moduleId: this.moduleId,
            timestamp: this.timestamp,
            duration: Date.now() - this.startedAt,
            input: this.input,
            output: { ...output },
            rawLMInput: last?.prompt ?? '',
            rawLMOutput: last?.rawOutput ?? '',
            usage: sumUsage(this.calls),
            calls: [...this.calls],
        };
    }
}

/**
 * Open a span for a module invocation, or return `undefined` when tracing is
 * off. Callers guard every trace operation with `span?.`, which keeps the
 * untraced path to a single boolean check.
 */
export function beginTrace(
    moduleId: string,
    lm: ILanguageModel,
    input: Record<string, any>
): TraceSpan | undefined {
    if (!isTracingEnabled()) {
        return undefined;
    }
    return new TraceSpan(moduleId, lm, input);
}

/**
 * Append an entry to the history and hand it to the configured `onTrace`
 * handler. A handler that throws is ignored: instrumentation must not be able
 * to fail the run it is instrumenting.
 */
function record(entry: TraceEntry): TraceEntry {
    history.push(entry);

    const limit = getTraceHistorySize();
    if (history.length > limit) {
        history.splice(0, history.length - limit);
    }

    const handler = getTraceHandler();
    if (handler) {
        try {
            handler(entry);
        } catch {
            // Deliberately swallowed — see the doc comment above.
        }
    }

    return entry;
}

/**
 * The last `n` recorded traces, oldest first &mdash; the prompts actually sent,
 * the raw replies, and what was parsed out of them. Omit `n` for everything
 * still retained.
 *
 * Returns an empty array unless `configure({ tracing: true })` was set before
 * the calls ran.
 */
export function inspectHistory(n?: number): TraceEntry[] {
    if (n === undefined) {
        return [...history];
    }
    if (n <= 0) {
        return [];
    }
    return history.slice(-n);
}

/** Discard every recorded trace. Useful between tests. */
export function clearHistory(): void {
    history.length = 0;
}
