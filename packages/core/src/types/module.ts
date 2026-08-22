import type { UsageStats } from './language-model';

/** One language-model round trip made while a module ran. */
export interface LMCall {
    /** The prompt text sent to the provider. */
    prompt: string;
    /**
     * The provider's reply, verbatim. Structured-output calls return an object
     * rather than text; it is stored here as JSON.
     */
    rawOutput: string;
    /** Wall-clock milliseconds spent on this call. */
    duration: number;
    /**
     * Tokens attributable to this call, not the model's lifetime totals.
     * Derived by differencing the model's counters either side of the call, so
     * it is exact for sequential calls and approximate when several run
     * concurrently against one shared model.
     */
    usage: UsageStats;
}

/**
 * A single module invocation, recorded when `configure({ tracing: true })` is
 * set. Attached to the returned `Prediction` as `trace` and appended to
 * the history read by `inspectHistory()`.
 */
export interface TraceEntry {
    /** Identifies the module instance, e.g. `ChainOfThought#2`. */
    moduleId: string;
    /** When the invocation started. */
    timestamp: Date;
    /** Wall-clock milliseconds for the whole invocation. */
    duration: number;
    /** The inputs the module was called with. */
    input: Record<string, any>;
    /** The parsed, validated outputs. Empty when the invocation threw. */
    output: Record<string, any>;
    /** Prompt of the last call — the one that produced `output`. */
    rawLMInput: string;
    /** Raw reply to the last call, before parsing. */
    rawLMOutput: string;
    /** Tokens used across the whole invocation, summed over `calls`. */
    usage: UsageStats;
    /**
     * Every language-model call made, in order. Multi-step modules make several,
     * and a call the provider failed is present with an empty `rawOutput`.
     */
    calls: LMCall[];
    /** Set when the invocation threw rather than returning a prediction. */
    error?: unknown;
}
