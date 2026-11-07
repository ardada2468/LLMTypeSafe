import { type Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import { type Module } from '../core/module';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import type { LMUsage } from '../core/usage';

/**
 * What a metric returns for one example: a numeric score, or a straight
 * pass/fail. `evaluate` reads `true`/`false` as `1`/`0`; an optimizer counts a
 * numeric score as a pass when it meets its `threshold`.
 */
export type MetricResult = number | boolean;

/**
 * @deprecated Use {@link MetricResult}; this is the same type under its old name.
 */
export type MetricScore = MetricResult;

/**
 * Grades one prediction against the example it came from.
 *
 * The shape is deliberately tiny and is treated as stable: `evaluate` and the
 * optimizers consume exactly this signature, so a metric written for one works
 * unchanged when handed to the other.
 */
export type Metric = (
    example: Example,
    prediction: Prediction
) => MetricResult | Promise<MetricResult>;

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
 * Token and latency totals for the run: exactly the language-model calls the
 * evaluation made, collected in an async-local scope around it, so concurrent
 * work on a shared model is never counted.
 *
 * Models implementing `ILanguageModel` by hand report usage only if they call
 * `recordLMUsage`; `BaseLM` subclasses do so automatically.
 */
export interface EvaluationUsage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    /** Language-model calls made during the run. */
    requestCount: number;
    /** Provider-level failures recorded during the run. */
    errorCount: number;
    /** Mean latency per language-model request, in milliseconds. */
    averageLatency: number;
    /** Wall-clock duration of the whole run, in milliseconds. */
    durationMs: number;
    /** The same usage, per `provider/model`. */
    byModel: LMUsage;
    /** Spend across priced models, when `configure({ pricing })` covers any. */
    totalCost?: number;
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
     * @deprecated Ignored. Usage used to be measured by differencing this
     * model's counters; it is now collected exactly from every call the run
     * makes, whichever model serves it.
     */
    lm?: ILanguageModel;
    /**
     * Called as each example finishes, in completion order. A reporting hook
     * only: anything it throws is swallowed and never affects a score.
     */
    onResult?: (result: EvaluationResult) => void;
}
