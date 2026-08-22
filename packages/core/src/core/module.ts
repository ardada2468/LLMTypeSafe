import { type Signature } from './signature';
import { type Prediction } from './prediction';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { getDefaultLM } from './config';
import { mapWithConcurrency, DEFAULT_CONCURRENCY, type SettledResult } from '../utils/pool';

/** Per-input outcome of {@link Module.batch}, in the shape of `Promise.allSettled`. */
export type BatchResult<T = Prediction> = SettledResult<T>;

/** Options for {@link Module.batch}: pool controls plus the usual per-call options. */
export interface BatchOptions extends LLMCallOptions {
    /** Maximum number of inputs in flight at once. Defaults to 8. */
    concurrency?: number;
    /** Called as each input settles, whether it succeeded or failed. */
    onProgress?: (done: number, total: number) => void;
    /** Reject the whole batch on the first failure instead of capturing it. */
    stopOnError?: boolean;
    /** Stop starting new inputs once aborted, then reject with the abort reason. */
    signal?: AbortSignal;
}

export abstract class Module {
    protected lm: ILanguageModel;
    protected signature?: typeof Signature | string;

    constructor(signature?: typeof Signature | string, lm?: ILanguageModel) {
        this.signature = signature;
        this.lm = lm || getDefaultLM();
    }

    abstract forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction>;

    /** Alias for {@link forward}, so modules read like function calls. */
    async call(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /** Alias for {@link forward}, mirroring DSPy's Python naming. */
    async __call__(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /**
     * Run {@link forward} over many inputs with a bounded number of calls in flight.
     *
     * Results come back **in input order**, whatever order the calls finished in.
     * By default a failing input is captured rather than thrown, so one bad row
     * does not destroy a ten-thousand-row job:
     *
     * ```ts
     * const results = await predict.batch(rows, { concurrency: 16 });
     * const answers = results.filter((r) => r.status === 'fulfilled').map((r) => r.value);
     * ```
     *
     * Every other option — `temperature`, `timeout`, `retries` and the rest — is
     * passed through to each underlying call unchanged.
     *
     * An aborted batch rejects rather than returning the inputs that already
     * finished, so wrap it in a `try` if you cancel it.
     *
     * @throws RangeError if `concurrency` is not a positive integer.
     * @throws the lowest-indexed failure when `stopOnError` is set, or the abort
     *   reason when `signal` aborts.
     */
    async batch(
        inputs: Record<string, any>[],
        options: BatchOptions = {}
    ): Promise<BatchResult<Awaited<ReturnType<this['forward']>>>[]> {
        const {
            concurrency = DEFAULT_CONCURRENCY,
            onProgress,
            stopOnError = false,
            signal,
            ...callOptions
        } = options;

        return mapWithConcurrency(inputs, (input) => this.forward(input, callOptions), {
            concurrency,
            onSettled: onProgress,
            stopOnError,
            signal,
        }) as Promise<BatchResult<Awaited<ReturnType<this['forward']>>>[]>;
    }
}
