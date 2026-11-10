import { type Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import { type Module } from '../core/module';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import type { LMUsage } from '../core/usage';

/**
 * A score with an explanation: what a metric returns when it can say *why* an
 * output fell short. `evaluate` and most optimizers read only `score`;
 * reflective optimizers (`GEPA`) hand `feedback` to a model that rewrites the
 * program's instructions, which teaches it far more than a bare number does.
 */
export interface ScoreWithFeedback {
    score: number;
    feedback?: string;
}

/**
 * What a metric returns for one example: a numeric score, a straight
 * pass/fail, or a score with feedback. `evaluate` reads `true`/`false` as
 * `1`/`0` and `{ score }` as `score`; an optimizer counts a score as a pass
 * when it meets its `threshold`.
 */
export type MetricResult = number | boolean | ScoreWithFeedback;

/**
 * @deprecated Use {@link MetricResult}; this is the same type under its old name.
 */
export type MetricScore = MetricResult;

/**
 * One predictor call recorded while a program ran: the predictor's path in the
 * program (see `Module.namedPredictors`), and that call's inputs and outputs.
 */
export interface MetricTraceStep {
    path: string;
    inputs: Record<string, any>;
    outputs: Record<string, any>;
}

/**
 * Grades one prediction against the example it came from.
 *
 * Most metrics need only the first two arguments, and `evaluate` passes only
 * those. Optimizers may pass more, as DSPy does:
 *
 * - `trace` — every predictor call the run made. Present while an optimizer
 *   bootstraps demos, so a metric can be stricter there (DSPy's
 *   `trace is not None` idiom) or check intermediate steps.
 * - `predName` / `predTrace` — set by `GEPA` when it asks for feedback on one
 *   predictor: that predictor's path, and just its calls. A metric can then
 *   return feedback aimed at that step.
 *
 * The extra parameters are optional, so a two-argument metric is still a
 * `Metric`, and one metric works with `evaluate` and every optimizer.
 */
export type Metric = (
    example: Example,
    prediction: Prediction,
    trace?: MetricTraceStep[],
    predName?: string,
    predTrace?: MetricTraceStep[]
) => MetricResult | Promise<MetricResult>;

/**
 * The number in a metric result: booleans become `1`/`0`, `{ score }` becomes
 * `score`, and anything non-finite becomes `0` — a metric that could not
 * produce a number must not poison an average.
 */
export function metricScore(result: MetricResult): number {
    if (typeof result === 'boolean') return result ? 1 : 0;
    const value = typeof result === 'number' ? result : Number(result?.score);
    return Number.isFinite(value) ? value : 0;
}

/** The feedback in a metric result, when it carries any. */
export function metricFeedback(result: MetricResult): string | undefined {
    if (typeof result === 'object' && result !== null && typeof result.feedback === 'string') {
        return result.feedback;
    }
    return undefined;
}

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
    /** The metric's feedback, when it returned `{ score, feedback }`. */
    feedback?: string;
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
    /**
     * Give up once more than this many examples have failed, rejecting with a
     * `TsDspyError` whose `cause` is the first failure. Unset, a run always
     * completes — which is right for a flaky row, and wrong for a revoked API
     * key, where every remaining call is wasted.
     */
    maxErrors?: number;
    /** Score recorded for an example whose program or metric threw. Default `0`. */
    failureScore?: number;
}
