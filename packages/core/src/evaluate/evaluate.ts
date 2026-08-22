import { type Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import { getDefaultLM } from '../core/config';
import type { ILanguageModel, LLMCallOptions, UsageStats } from '../types/language-model';
import type {
    EvaluateOptions,
    EvaluationProgram,
    EvaluationReport,
    EvaluationResult,
    EvaluationUsage,
    Metric,
    MetricScore,
} from './types';

const DEFAULT_CONCURRENCY = 4;

/**
 * Run `worker` over `items` with at most `limit` in flight, preserving input
 * order in the returned array.
 *
 * Small and local on purpose: an evaluation needs no more than this, and
 * `worker` is expected never to reject.
 */
async function mapWithConcurrency<T, R>(
    items: readonly T[],
    limit: number,
    worker: (item: T, index: number) => Promise<R>
): Promise<R[]> {
    const results = new Array<R>(items.length);
    // Only a missing or unusable limit falls back to the default: a caller who
    // computed `0` from a rate-limit budget must not get four in flight.
    const requested = Number.isFinite(limit) ? Math.floor(limit) : DEFAULT_CONCURRENCY;
    const lanes = Math.min(Math.max(1, requested), items.length);

    let cursor = 0;
    const runners = Array.from({ length: lanes }, async () => {
        while (cursor < items.length) {
            const index = cursor;
            cursor += 1;
            results[index] = await worker(items[index], index);
        }
    });

    await Promise.all(runners);
    return results;
}

function toError(cause: unknown): Error {
    return cause instanceof Error ? cause : new Error(String(cause));
}

/** Booleans become 1/0; anything non-finite becomes 0. */
function toScore(value: MetricScore): number {
    if (typeof value === 'boolean') return value ? 1 : 0;
    return Number.isFinite(value) ? value : 0;
}

function hasUsageCounters(value: unknown): value is ILanguageModel {
    return typeof (value as ILanguageModel | undefined)?.getUsage === 'function';
}

/**
 * The model whose counters bracket the run: the caller's choice, else the
 * program's own, else the configured default, else none.
 */
function resolveLM(
    program: EvaluationProgram,
    override?: ILanguageModel
): ILanguageModel | undefined {
    if (override) return override;

    // `Module.lm` is protected in the type system but a plain runtime property.
    const own = (program as unknown as { lm?: unknown }).lm;
    if (hasUsageCounters(own)) return own;

    try {
        return getDefaultLM();
    } catch {
        return undefined;
    }
}

function counter(value: number | undefined): number {
    return Number.isFinite(value) ? (value as number) : 0;
}

/**
 * Difference two usage snapshots. `averageLatency` is a mean, so it is
 * re-weighted by request count before subtracting. Negative deltas — a
 * `resetUsage()` mid-run — are floored at zero rather than reported.
 */
function diffUsage(
    before: UsageStats | undefined,
    after: UsageStats | undefined,
    durationMs: number
): EvaluationUsage {
    if (!before || !after) {
        return {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            requestCount: 0,
            errorCount: 0,
            averageLatency: 0,
            durationMs,
        };
    }

    const promptTokens = Math.max(
        0,
        counter(after.promptTokens) - counter(before.promptTokens)
    );
    const completionTokens = Math.max(
        0,
        counter(after.completionTokens) - counter(before.completionTokens)
    );
    const requestCount = Math.max(
        0,
        counter(after.requestCount) - counter(before.requestCount)
    );
    const latencySum = Math.max(
        0,
        counter(after.averageLatency) * counter(after.requestCount) -
            counter(before.averageLatency) * counter(before.requestCount)
    );

    return {
        promptTokens,
        completionTokens,
        totalTokens: promptTokens + completionTokens,
        requestCount,
        errorCount: Math.max(0, counter(after.errorCount) - counter(before.errorCount)),
        averageLatency: requestCount > 0 ? latencySum / requestCount : 0,
        durationMs,
    };
}

async function runProgram(
    program: EvaluationProgram,
    inputs: Record<string, any>,
    callOptions?: LLMCallOptions
): Promise<Prediction> {
    return typeof program === 'function'
        ? program(inputs, callOptions)
        : program.forward(inputs, callOptions);
}

/** Expected outputs, or `{}` when the example has no input keys to split on. */
function safeOutputs(example: Example): Record<string, any> {
    try {
        return example.getOutputs();
    } catch {
        return {};
    }
}

/**
 * Run `program` over `dataset`, grade each prediction with `metric`, and report
 * the aggregate.
 *
 * Failures are data, not control flow: an example whose program or metric
 * throws is recorded as a zero-score result with the error attached, and the
 * run continues. An evaluation that dies on row 40 of 500 tells you nothing.
 *
 * ```ts
 * const report = await evaluate(program, dataset, exactMatch, { concurrency: 8 });
 * report.score; // mean across every example
 * ```
 */
export async function evaluate(
    program: EvaluationProgram,
    dataset: readonly Example[],
    metric: Metric,
    options: EvaluateOptions = {}
): Promise<EvaluationReport> {
    const { concurrency = DEFAULT_CONCURRENCY, inputKeys, callOptions, onResult } = options;

    const prepared = dataset.map((example) =>
        inputKeys ? example.withInputs(...inputKeys) : example
    );

    const lm = resolveLM(program, options.lm);
    const before = lm?.getUsage();
    const startedAt = Date.now();

    const results = await mapWithConcurrency(prepared, concurrency, async (example, index) => {
        const exampleStartedAt = Date.now();
        let inputs: Record<string, any> = {};
        // Held outside the try so a metric that throws still reports what the
        // model produced — otherwise debugging your own metric is guesswork.
        let prediction: Prediction | undefined;
        let result: EvaluationResult;

        try {
            inputs = example.getInputs();
            prediction = await runProgram(program, inputs, callOptions);
            const score = toScore(await metric(example, prediction));

            result = {
                index,
                example,
                inputs,
                expected: safeOutputs(example),
                prediction,
                score,
                durationMs: Date.now() - exampleStartedAt,
            };
        } catch (error) {
            result = {
                index,
                example,
                inputs,
                expected: safeOutputs(example),
                prediction,
                score: 0,
                error: toError(error),
                durationMs: Date.now() - exampleStartedAt,
            };
        }

        try {
            onResult?.(result);
        } catch {
            // A reporting hook must not be able to change a score.
        }

        return result;
    });

    const usage = diffUsage(before, lm?.getUsage(), Date.now() - startedAt);
    const totalScore = results.reduce((sum, result) => sum + result.score, 0);

    return {
        score: results.length > 0 ? totalScore / results.length : 0,
        totalScore,
        count: results.length,
        errorCount: results.filter((result) => result.error !== undefined).length,
        results,
        usage,
    };
}
