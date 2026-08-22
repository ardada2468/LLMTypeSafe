import { type Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import { type Module } from '../core/module';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';

/**
 * Score for one example. `true`/`false` are read as `1`/`0`, so a boolean
 * metric needs no ceremony.
 */
export type MetricScore = number | boolean;

/**
 * Grades one prediction against the example it came from.
 *
 * The shape is deliberately tiny and is treated as stable: optimizers consume
 * exactly this signature, so a metric written for `evaluate` works unchanged
 * when handed to a compiler.
 */
export type Metric = (
    example: Example,
    prediction: Prediction
) => MetricScore | Promise<MetricScore>;

/**
 * Anything `evaluate` can run against one example's inputs — a module, or a
 * bare function when the program under test is not a single module.
 */
export type EvaluationProgram =
    | Module
    | ((
          inputs: Record<string, any>,
          options?: LLMCallOptions
      ) => Prediction | Promise<Prediction>);

/** Outcome for a single example. A failure here never aborts the run. */
export interface EvaluationResult {
    /** Position in the dataset. Results come back in dataset order. */
    index: number;
    /** The dataset example, with its input keys resolved. */
    example: Example;
    /** What was fed to the program. Empty when the inputs could not be read. */
    inputs: Record<string, any>;
    /** The example's expected outputs. Empty when they could not be read. */
    expected: Record<string, any>;
    /**
     * Absent only when the program itself threw. A metric that throws still
     * leaves the prediction here, so you can see what it choked on.
     */
    prediction?: Prediction;
    /** Metric score, or `0` when this example failed. */
    score: number;
    /** Set when the program or the metric threw for this example. */
    error?: Error;
    /** Wall-clock time for this example, in milliseconds. */
    durationMs: number;
}

/**
 * Token and latency totals for the run, obtained by diffing the language
 * model's own counters around it.
 *
 * There is deliberately no cost field. ts-dspy carries no price table — the one
 * it used to carry went stale and reported figures wrong by more than an order
 * of magnitude. Multiply these token counts by prices you control instead.
 */
export interface EvaluationUsage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    /** Language-model calls made during the run. */
    requestCount: number;
    /** Provider-level failures recorded by the model during the run. */
    errorCount: number;
    /** Mean latency per language-model request, in milliseconds. */
    averageLatency: number;
    /** Wall-clock duration of the whole run, in milliseconds. */
    durationMs: number;
}

/** Aggregate result of an evaluation run. */
export interface EvaluationReport {
    /** Mean score across every example, failures included. */
    score: number;
    /** Sum of every example's score. */
    totalScore: number;
    /** Number of examples evaluated. */
    count: number;
    /** Number of examples whose program or metric threw. */
    errorCount: number;
    /** Per-example outcomes, in dataset order. */
    results: EvaluationResult[];
    /** Tokens and latency across the run. */
    usage: EvaluationUsage;
}

export interface EvaluateOptions {
    /** Maximum examples in flight at once. Defaults to `4`. */
    concurrency?: number;
    /**
     * Input field names, applied with `Example.withInputs` to every example.
     * Omit when the dataset already carries them.
     */
    inputKeys?: string[];
    /** Per-call provider options, forwarded to the program. */
    callOptions?: LLMCallOptions;
    /**
     * Model whose counters are diffed for {@link EvaluationUsage}. Defaults to
     * the module's own model, then to the configured default. Usage comes back
     * zeroed when neither exists.
     *
     * A function program has no model to find, so it falls straight through to
     * the configured default — set this explicitly when the function calls a
     * model other than that one, or the token counts describe an idle model.
     */
    lm?: ILanguageModel;
    /**
     * Called as each example finishes, in completion order. A reporting hook
     * only: anything it throws is swallowed and never affects a score.
     */
    onResult?: (result: EvaluationResult) => void;
}
