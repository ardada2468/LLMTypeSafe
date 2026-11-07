import { type Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import {
    mapWithConcurrency,
    DEFAULT_CONCURRENCY,
    assertValidConcurrency,
    type SettledResult,
} from '../utils/pool';
import type { LLMCallOptions } from '../types/language-model';

/** One unit of work: a module, its inputs, and optionally per-call options. */
export type ExecPair =
    | readonly [module: Module, inputs: Record<string, any>]
    | readonly [module: Module, inputs: Record<string, any>, options: LLMCallOptions];

export interface ParallelOptions<TReturnFailed extends boolean = boolean> {
    /** Maximum number of pairs in flight at once. Defaults to 8. */
    concurrency?: number;
    /**
     * Abandon the run — start nothing new, wait for what is in flight, then
     * reject — once this many pairs have failed. Unlimited by default, so a
     * failing pair costs only its own slot, as with `Module.batch`.
     */
    maxErrors?: number;
    /**
     * Resolve to `{ results, failed }` instead of a bare results array, so
     * each failure's inputs and error can be inspected.
     */
    returnFailed?: TReturnFailed;
    /** Stop starting new pairs once aborted, then reject with the abort reason. */
    signal?: AbortSignal;
}

/** A pair that threw, in {@link ParallelOutcome.failed}. */
export interface FailedExec {
    /** Position of the pair in the array given to {@link Parallel.run}. */
    index: number;
    module: Module;
    inputs: Record<string, any>;
    error: unknown;
}

/** What {@link Parallel.run} resolves to with `returnFailed: true`. */
export interface ParallelOutcome {
    /** One slot per pair, in input order; `undefined` where the pair failed. */
    results: Array<(Prediction & Record<string, any>) | undefined>;
    /** Every failure, in input order. */
    failed: FailedExec[];
}

/** Rejected with when a run reaches `maxErrors`. `cause` is the failure that tipped it. */
export class ParallelError extends TsDspyError {
    constructor(
        message: string,
        /** Every failure observed before the run stopped, in input order. */
        readonly failed: FailedExec[],
        options?: { cause?: unknown }
    ) {
        super(message, options);
    }
}

/**
 * Run many `[module, inputs]` pairs concurrently — DSPy's `Parallel`.
 *
 * Different modules may share one run, which is what sets this apart from
 * `Module.batch`. Results come back **in input order**, whatever order the
 * calls finished in, with `undefined` in the slot of a pair that failed; set
 * `returnFailed` to also get those failures with their inputs and errors.
 *
 * ```ts
 * const [summary, sentiment] = await new Parallel({ concurrency: 4 }).run([
 *     [summarize, { text }],
 *     [classify, { text }],
 * ]);
 * ```
 *
 * Each pair runs in the caller's async context, so `settings.context`
 * overrides, callbacks and usage tracking around `run()` apply to every pair.
 */
export class Parallel<TReturnFailed extends boolean = false> {
    readonly concurrency: number;
    readonly maxErrors: number;
    readonly returnFailed: TReturnFailed;
    private readonly signal?: AbortSignal;

    constructor(options: ParallelOptions<TReturnFailed> = {}) {
        this.concurrency = options.concurrency ?? DEFAULT_CONCURRENCY;
        assertValidConcurrency(this.concurrency);
        this.maxErrors = options.maxErrors ?? Infinity;
        if (!(this.maxErrors >= 1)) {
            throw new RangeError(
                `maxErrors must be at least 1, received ${String(options.maxErrors)}`
            );
        }
        this.returnFailed = (options.returnFailed ?? false) as TReturnFailed;
        this.signal = options.signal;
    }

    /**
     * Run every pair. Resolves to the results in input order — or, with
     * `returnFailed`, to `{ results, failed }`.
     *
     * @throws {ParallelError} once `maxErrors` pairs have failed.
     * @throws the signal's abort reason when `signal` aborts.
     */
    async run(
        execPairs: readonly ExecPair[]
    ): Promise<
        TReturnFailed extends true
            ? ParallelOutcome
            : Array<(Prediction & Record<string, any>) | undefined>
    > {
        const failed: FailedExec[] = [];
        // An internal controller lets reaching `maxErrors` stop the pool from
        // starting new work, the same way a caller's abort does.
        const controller = new AbortController();
        const onAbort = () => controller.abort(this.signal!.reason);
        if (this.signal?.aborted) throw this.signal.reason;
        this.signal?.addEventListener('abort', onAbort, { once: true });

        let tooMany: ParallelError | undefined;
        let settled: SettledResult<Prediction & Record<string, any>>[];
        try {
            settled = await mapWithConcurrency(
                execPairs,
                async (pair, index) => {
                    const [module, inputs, options] = pair;
                    try {
                        return (await module.forward(inputs, options)) as Prediction &
                            Record<string, any>;
                    } catch (error) {
                        failed.push({ index, module, inputs, error });
                        if (failed.length >= this.maxErrors && !tooMany) {
                            tooMany = new ParallelError(
                                `Parallel stopped after ${failed.length} failed execution(s) ` +
                                    `(maxErrors: ${this.maxErrors}).`,
                                failed,
                                { cause: error }
                            );
                            controller.abort(tooMany);
                        }
                        throw error;
                    }
                },
                { concurrency: this.concurrency, signal: controller.signal }
            );
        } catch (error) {
            failed.sort((a, b) => a.index - b.index);
            throw tooMany ?? error;
        } finally {
            this.signal?.removeEventListener('abort', onAbort);
        }

        failed.sort((a, b) => a.index - b.index);
        const results = settled.map((slot) =>
            slot.status === 'fulfilled' ? slot.value : undefined
        );
        return (this.returnFailed ? { results, failed } : results) as TReturnFailed extends true
            ? ParallelOutcome
            : Array<(Prediction & Record<string, any>) | undefined>;
    }
}
