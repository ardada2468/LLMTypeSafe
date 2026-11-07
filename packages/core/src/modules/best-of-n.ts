import { Module } from '../core/module';
import type { Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import { runRollout } from '../core/rollout';
import type { LLMCallOptions } from '../types/language-model';

/**
 * Scores one attempt. Receives the program's inputs and the attempt's
 * prediction; higher is better. May be async — an LLM judge, say.
 */
export type RewardFn = (
    inputs: Record<string, any>,
    prediction: Prediction
) => number | Promise<number>;

/** Options shared by {@link BestOfN} and `Refine`. */
export interface BestOfNOptions {
    /** Maximum number of attempts. A positive integer. */
    n: number;
    /** Scores each attempt; see {@link RewardFn}. */
    rewardFn: RewardFn;
    /** Stop at the first attempt whose reward reaches this. */
    threshold: number;
    /**
     * How many attempts may throw before the error is rethrown. Defaults to
     * `n`, so any number of failures is tolerated as long as one attempt
     * succeeds; `0` rethrows the first failure.
     */
    failCount?: number;
    /**
     * Sampling temperature for every call an attempt makes. Defaults to `1.0`,
     * as in DSPy: attempts differ by their rollout id, and a temperature high
     * enough to actually sample something different.
     */
    temperature?: number;
}

/** What one attempt produced, for {@link selectBest}. */
interface AttemptOutcome {
    prediction: Prediction;
    reward: number;
}

/**
 * Run `module` up to `n` times and keep the best prediction — DSPy's
 * `BestOfN`.
 *
 * Each attempt runs a {@link Module.deepCopy} of `module` with its models
 * wrapped so every call carries the attempt's temperature and a distinct
 * `metadata.rolloutId`. The rollout id is part of the response-cache key, so
 * with caching on each attempt is a genuinely new sample rather than the
 * first attempt's reply served again; a re-run of the whole program still
 * hits the cache attempt for attempt.
 *
 * Returns the first prediction whose reward reaches `threshold`, or, when
 * none does, the one with the highest reward. An attempt that throws — or
 * whose reward function throws — counts against `failCount`; once more than
 * `failCount` have failed the latest error is rethrown, and if every attempt
 * failed the last error is rethrown too.
 *
 * ```ts
 * const qa = new BestOfN(new ChainOfThought('question -> answer'), {
 *     n: 3,
 *     rewardFn: (_inputs, pred) => (pred.answer.split(' ').length === 1 ? 1 : 0),
 *     threshold: 1,
 * });
 * ```
 */
export class BestOfN extends Module {
    /** The program being sampled. Its predictors are this module's. */
    readonly module: Module;
    protected readonly n: number;
    protected readonly rewardFn: RewardFn;
    protected readonly threshold: number;
    protected readonly failCount: number;
    protected readonly temperature: number;

    constructor(module: Module, options: BestOfNOptions) {
        super();
        if (!(module instanceof Module)) {
            throw new TsDspyError(`${new.target.name} expects a Module to sample.`);
        }
        if (!Number.isInteger(options.n) || options.n < 1) {
            throw new RangeError(`n must be a positive integer, received ${String(options.n)}`);
        }
        if (typeof options.rewardFn !== 'function') {
            throw new TsDspyError(`${new.target.name} requires a rewardFn.`);
        }
        this.module = module;
        this.n = options.n;
        this.rewardFn = options.rewardFn;
        this.threshold = options.threshold;
        this.failCount = options.failCount ?? options.n;
        this.temperature = options.temperature ?? 1.0;
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction & Record<string, any>> {
        const outcomes: AttemptOutcome[] = [];
        let failures = 0;
        let lastError: unknown;

        for (let rolloutId = 0; rolloutId < this.n; rolloutId++) {
            options?.signal?.throwIfAborted();
            try {
                const prediction = await runRollout(this.module, inputs, options, {
                    rolloutId,
                    temperature: this.temperature,
                });
                const reward = await this.rewardFn(inputs, prediction);
                outcomes.push({ prediction, reward });
                if (reward >= this.threshold) break;
            } catch (error) {
                // A cancelled run is not a failed attempt: stop at once rather
                // than burn the remaining attempts, each failing on the abort.
                rethrowIfAborted(options?.signal, error);
                failures += 1;
                lastError = error;
                if (failures > this.failCount) throw error;
            }
        }

        return selectBest(outcomes, lastError);
    }
}

/** Rethrow `error` when `signal` has aborted. Shared with `Refine`. */
export function rethrowIfAborted(signal: AbortSignal | undefined, error: unknown): void {
    if (signal?.aborted) throw error;
}

/**
 * The highest-rewarded prediction — the earliest among ties, NaN lowest — or
 * the last error when no attempt produced one.
 */
export function selectBest(
    outcomes: readonly AttemptOutcome[],
    lastError: unknown
): Prediction & Record<string, any> {
    // A NaN reward ranks below every number, as in DSPy (which starts from
    // `-inf` and keeps only a strictly greater reward): a reward function that
    // could not score an attempt must not win over one that did.
    const rank = (reward: number) => (Number.isNaN(reward) ? -Infinity : reward);
    let best: AttemptOutcome | undefined;
    for (const outcome of outcomes) {
        if (!best || rank(outcome.reward) > rank(best.reward)) best = outcome;
    }
    if (!best) {
        throw lastError ?? new TsDspyError('No attempt produced a prediction.');
    }
    return best.prediction as Prediction & Record<string, any>;
}
