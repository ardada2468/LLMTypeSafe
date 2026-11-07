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
     * Tokens attributable to exactly this call, not the model's lifetime
     * totals — collected in an async-local scope around the call, so it stays
     * exact when many calls run concurrently against one shared model.
     */
    usage: UsageStats;
}

/**
 * A single module invocation, recorded when `configure({ tracing: true })` is
 * set. Attached to the returned `Prediction` as `trace` and appended to
 * the history read by `inspectHistory()`.
 */
export interface TraceEntry {
    /**
     * Unique per invocation; the same value as the `callId` callbacks see for
     * this module call.
     */
    id: string;
    /**
     * The `id` of the enclosing module invocation, when this one ran inside
     * another — so traces of a composite program nest. The parent has an entry
     * of its own only if it records traces itself (the built-in modules do; a
     * hand-written composite's `forward` usually does not).
     */
    parentId?: string;
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
