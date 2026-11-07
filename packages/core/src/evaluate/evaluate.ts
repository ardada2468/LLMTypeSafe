import { type Example } from '../core/example';
import { mapWithConcurrency } from '../utils/pool';
import { type Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import { collectUsage, sumUsage, type UsageCollector } from '../core/usage';
import type { LLMCallOptions } from '../types/language-model';
import type {
    EvaluateOptions,
    EvaluationProgram,
    EvaluationReport,
    EvaluationResult,
    EvaluationUsage,
    Metric,
} from './types';
import { metricFeedback, metricScore } from './types';

const DEFAULT_CONCURRENCY = 4;

/**
 * Run `worker` over `items` with at most `limit` in flight, preserving input
 * order.
 *
 * Delegates to the shared pool in `utils/pool`; the clamp lives here because
 * evaluation treats a computed `0` as "one at a time" — a caller deriving the
 * limit from a rate-limit budget must not get the default four — whereas the
 * shared pool rejects a non-positive limit outright.
 */
async function runPooled<T, R>(
    items: readonly T[],
    limit: number,
    worker: (item: T, index: number) => Promise<R>
): Promise<R[]> {
    if (items.length === 0) return [];

    const requested = Number.isFinite(limit) ? Math.floor(limit) : DEFAULT_CONCURRENCY;
    const lanes = Math.min(Math.max(1, requested), items.length);

    const settled = await mapWithConcurrency([...items], (item, index) => worker(item, index), {
        concurrency: lanes,
    });

    return settled.map((result) => {
        // The evaluation worker captures its own failures, so a rejection here
        // is a bug in this module rather than a bad example.
        if (result.status === 'rejected') throw result.reason;
        return result.value as R;
    });
}

function toError(cause: unknown): Error {
    return cause instanceof Error ? cause : new Error(String(cause));
}

/** Fold a run's collected usage into the report's shape. */
function toEvaluationUsage(usage: UsageCollector, durationMs: number): EvaluationUsage {
    const byModel = usage.snapshot();
    const total = sumUsage(Object.values(byModel));
    const report: EvaluationUsage = {
        promptTokens: total.promptTokens,
        completionTokens: total.completionTokens,
        totalTokens: total.totalTokens,
        requestCount: total.requestCount ?? 0,
        errorCount: total.errorCount ?? 0,
        averageLatency: total.averageLatency ?? 0,
        durationMs,
        byModel,
    };
    if (total.totalCost !== undefined) report.totalCost = total.totalCost;
    return report;
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
 * Fail before any model call when no example says which fields are inputs.
 *
 * Without this every row would fail individually with the same complaint, and
 * the report would read as a 0% score rather than a setup mistake. A dataset
 * where only some examples lack the split still runs: those rows fail on their
 * own, as any other bad row does.
 */
function assertInputKeysDesignated(dataset: readonly Example[]): void {
    if (dataset.length === 0 || dataset.some((example) => example.hasInputKeys())) return;

    throw new TsDspyError(
        'No example in the dataset designates its input fields, so evaluate cannot ' +
            'tell inputs from expected outputs. Call example.withInputs(...keys) when ' +
            'building the dataset, or pass { inputKeys: [...] } to evaluate().'
    );
}

/**
 * Run `program` over `dataset`, grade each prediction with `metric`, and report
 * the aggregate.
 *
 * Failures are data, not control flow: an example whose program or metric
 * throws is recorded as a zero-score result with the error attached, and the
 * run continues. An evaluation that dies on row 40 of 500 tells you nothing.
 *
 * @throws {TsDspyError} up front when no example designates its input fields
 * (via `withInputs`) and no `inputKeys` option is given.
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
    assertInputKeysDesignated(prepared);

    const startedAt = Date.now();

    // Usage is collected in an async-local scope around the run, so it counts
    // exactly the calls this evaluation made — whichever models they went to —
    // and nothing a concurrent job did on a shared model.
    const { result: results, usage } = await collectUsage(() =>
        runPooled(prepared, concurrency, async (example, index) => {
            const exampleStartedAt = Date.now();
            let inputs: Record<string, any> = {};
            // Held outside the try so a metric that throws still reports what the
            // model produced — otherwise debugging your own metric is guesswork.
            let prediction: Prediction | undefined;
            let result: EvaluationResult;

            try {
                inputs = example.getInputs();
                prediction = await runProgram(program, inputs, callOptions);
                const verdict = await metric(example, prediction);
                const score = metricScore(verdict);
                const feedback = metricFeedback(verdict);

                result = {
                    index,
                    example,
                    inputs,
                    expected: safeOutputs(example),
                    prediction,
                    score,
                    ...(feedback !== undefined ? { feedback } : {}),
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
        })
    );

    const report = toEvaluationUsage(usage, Date.now() - startedAt);
    const totalScore = results.reduce((sum, result) => sum + result.score, 0);

    return {
        score: results.length > 0 ? totalScore / results.length : 0,
        totalScore,
        count: results.length,
        errorCount: results.filter((result) => result.error !== undefined).length,
        results,
        usage: report,
    };
}
