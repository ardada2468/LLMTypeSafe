import type {
    ChatMessage,
    ChatResult,
    ILanguageModel,
    LLMCallOptions,
    ModelCapabilities,
    StreamChunk,
    UsageStats,
} from '../types/language-model';
import { BaseLM } from './base-lm';
import { type LMMethod } from './callbacks';
import { LMError, RateLimitError, TimeoutError, TsDspyError } from './errors';

/** One language-model call as middleware sees it. */
export interface LMRequest {
    method: LMMethod;
    /** For `generate` and `generateStructured`. */
    prompt?: string;
    /** For `generateStructured`. */
    schema?: unknown;
    /** For `chat`, `chatWithTools` and `chatStream`. */
    messages?: ChatMessage[];
    options?: LLMCallOptions;
}

/**
 * Continue down the chain. Resolves to whatever the method resolves to: a
 * string, a structured value, a `ChatResult`, or — for `chatStream` — an
 * async generator of chunks whose first chunk has already arrived, so a
 * failure to connect surfaces here, where `retry` and `fallback` can act on it.
 */
export type LMNext = (request: LMRequest) => Promise<unknown>;

/**
 * Intercepts calls to a wrapped model. Call `next` (any number of times, with
 * the request as-is or modified) to proceed, or return without calling it to
 * short-circuit.
 */
export type LMMiddleware = (request: LMRequest, next: LMNext) => Promise<unknown>;

/**
 * Send `request` straight to `lm`, with no middleware. Exported for
 * middleware that needs to route a request to a different model, as
 * {@link fallback} does.
 */
export async function dispatchLM(lm: ILanguageModel, request: LMRequest): Promise<unknown> {
    const { options } = request;
    switch (request.method) {
        case 'generate':
            return lm.generate(request.prompt ?? '', options);
        case 'generateStructured':
            return lm.generateStructured(request.prompt ?? '', request.schema, options);
        case 'chat':
            return lm.chat(request.messages ?? [], options);
        case 'chatWithTools':
            if (typeof lm.chatWithTools === 'function') {
                return lm.chatWithTools(request.messages ?? [], options);
            }
            return {
                content: await lm.chat(request.messages ?? [], options),
                finishReason: 'stop',
            } satisfies ChatResult;
        case 'chatStream':
            if (typeof lm.chatStream !== 'function') {
                throw new TsDspyError(`${lm.getModelName()} does not support chatStream.`);
            }
            return openStream(lm.chatStream(request.messages ?? [], options));
    }
}

/**
 * Pull a stream's first chunk before handing it on, so connection failures
 * reject here rather than on the consumer's first read, then replay it.
 */
async function openStream(
    stream: AsyncGenerator<StreamChunk, void, unknown>
): Promise<AsyncGenerator<StreamChunk, void, unknown>> {
    const first = await stream.next();
    return (async function* replay() {
        try {
            if (first.done) return;
            yield first.value;
            yield* stream;
        } finally {
            await stream.return?.(undefined);
        }
    })();
}

/**
 * A model with middleware around every call. Everything that is not a call —
 * capabilities, usage, the model name — is delegated to the wrapped model, so
 * the wrapper can stand in for it anywhere.
 */
class MiddlewareLM implements ILanguageModel {
    chatWithTools?: (messages: ChatMessage[], options?: LLMCallOptions) => Promise<ChatResult>;
    chatStream?: (
        messages: ChatMessage[],
        options?: LLMCallOptions
    ) => AsyncGenerator<StreamChunk, void, unknown>;
    generateStream?: (
        prompt: string,
        options?: LLMCallOptions
    ) => AsyncGenerator<StreamChunk, void, unknown>;
    setModel?: (model: string) => void;
    listModels?: () => Promise<string[]>;
    isHealthy?: () => Promise<boolean>;
    getCostEstimate?: (prompt: string, options?: LLMCallOptions) => Promise<number>;

    private readonly chain: LMNext;

    constructor(
        readonly inner: ILanguageModel,
        middleware: readonly LMMiddleware[]
    ) {
        this.chain = middleware.reduceRight<LMNext>(
            (next, layer) => (request) => layer(request, next),
            (request) => dispatchLM(inner, request)
        );

        // Optional methods are mirrored only when the wrapped model has them:
        // callers feature-detect them (`typeof lm.chatStream === 'function'`),
        // so defining them unconditionally would advertise abilities the
        // model lacks.
        if (typeof inner.chatWithTools === 'function') {
            // ReAct recognises "no native tool calling" by BaseLM's default
            // implementation being in place. Keep that identity: the default
            // only calls `this.chat`, which on this wrapper runs the chain.
            this.chatWithTools =
                inner.chatWithTools === BaseLM.prototype.chatWithTools
                    ? BaseLM.prototype.chatWithTools
                    : (messages, options) =>
                          this.chain({
                              method: 'chatWithTools',
                              messages,
                              options,
                          }) as Promise<ChatResult>;
        }
        if (typeof inner.chatStream === 'function') {
            this.chatStream = (messages, options) => this.stream(messages, options);
            this.generateStream = (prompt, options) =>
                this.stream([{ role: 'user', content: prompt }], options);
        }
        if (typeof inner.setModel === 'function') {
            this.setModel = (model) => inner.setModel!(model);
        }
        if (typeof inner.listModels === 'function') {
            this.listModels = () => inner.listModels!();
        }
        if (typeof inner.isHealthy === 'function') {
            this.isHealthy = () => inner.isHealthy!();
        }
        if (typeof inner.getCostEstimate === 'function') {
            this.getCostEstimate = (prompt, options) => inner.getCostEstimate!(prompt, options);
        }
    }

    generate(prompt: string, options?: LLMCallOptions): Promise<string> {
        return this.chain({ method: 'generate', prompt, options }) as Promise<string>;
    }

    generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        return this.chain({
            method: 'generateStructured',
            prompt,
            schema,
            options,
        }) as Promise<T>;
    }

    chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        return this.chain({ method: 'chat', messages, options }) as Promise<string>;
    }

    private async *stream(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        const stream = (await this.chain({
            method: 'chatStream',
            messages,
            options,
        })) as AsyncGenerator<StreamChunk, void, unknown>;
        yield* stream;
    }

    getUsage(): UsageStats {
        return this.inner.getUsage();
    }

    resetUsage(): void {
        this.inner.resetUsage();
    }

    getCapabilities(): ModelCapabilities {
        return this.inner.getCapabilities();
    }

    getModelName(): string {
        return this.inner.getModelName();
    }
}

/**
 * Wrap a model with middleware, outermost first:
 *
 * ```ts
 * const lm = wrapLM(
 *     new OpenAILM({ model: 'gpt-5.2' }),
 *     rateLimit({ requestsPerMinute: 500 }),
 *     retry({ maxRetries: 4 }),
 *     fallback([new AnthropicLM({ model: 'claude-sonnet-4-5' })]),
 * );
 * configure({ lm });
 * ```
 *
 * The wrapper delegates capabilities, usage and the model name to `lm`. Usage
 * of every model actually called — including a fallback — still lands in
 * per-call usage, because each provider records its own.
 */
export function wrapLM(lm: ILanguageModel, ...middleware: LMMiddleware[]): ILanguageModel {
    return new MiddlewareLM(lm, middleware);
}

// --------------------------------------------------------------------- retry

export interface BackoffOptions {
    /** Retries after the first attempt. Default 3. */
    maxRetries?: number;
    /** Delay before the first retry; doubles per attempt. Default 500ms. */
    initialDelayMs?: number;
    /** Ceiling on the computed backoff. Default 20s. */
    maxDelayMs?: number;
    /**
     * Randomise each delay to between half and all of its computed value, so
     * many clients throttled at once do not retry in lockstep. Default true.
     */
    jitter?: boolean;
    /** Stops retrying (and cuts a pending backoff short) once aborted. */
    signal?: AbortSignal;
    /** Whether a failure is worth another attempt. `attempt` counts from 0. */
    shouldRetry?: (error: unknown, attempt: number) => boolean;
}

/**
 * Run `attempt` until it succeeds, retrying failures `shouldRetry` accepts
 * with exponential backoff.
 *
 * An error carrying a numeric `retryAfterMs` (as {@link RateLimitError} can)
 * is waited out for at least that long, even past `maxDelayMs` — retrying
 * before the provider said to only earns another rejection. An aborted
 * `signal` rejects with its reason, before an attempt or during a backoff.
 */
export async function retryWithBackoff<T>(
    attempt: (attempt: number) => Promise<T>,
    options: BackoffOptions = {}
): Promise<T> {
    const {
        maxRetries = 3,
        initialDelayMs = 500,
        maxDelayMs = 20_000,
        jitter = true,
        signal,
        shouldRetry = isRetryableLMError,
    } = options;

    for (let n = 0; ; n++) {
        signal?.throwIfAborted();
        try {
            return await attempt(n);
        } catch (error) {
            if (n >= maxRetries || signal?.aborted || !shouldRetry(error, n)) throw error;
            await sleep(backoffDelay(n, error, { initialDelayMs, maxDelayMs, jitter }), signal);
        }
    }
}

function backoffDelay(
    attempt: number,
    error: unknown,
    options: { initialDelayMs: number; maxDelayMs: number; jitter: boolean }
): number {
    const exponential = Math.min(options.maxDelayMs, options.initialDelayMs * 2 ** attempt);
    const delay = options.jitter
        ? exponential / 2 + (Math.random() * exponential) / 2
        : exponential;

    const retryAfter = (error as { retryAfterMs?: unknown } | null)?.retryAfterMs;
    if (typeof retryAfter === 'number' && Number.isFinite(retryAfter) && retryAfter > 0) {
        return Math.max(delay, retryAfter);
    }
    return delay;
}

/** Sleep, unless the signal aborts first. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
    // An `abort` listener never fires for a signal that aborted earlier.
    if (signal?.aborted) return Promise.reject(signal.reason);
    if (ms <= 0) return Promise.resolve();

    return new Promise((resolve, reject) => {
        const onAbort = () => {
            clearTimeout(timer);
            reject(signal?.reason);
        };
        const timer = setTimeout(() => {
            signal?.removeEventListener('abort', onAbort);
            resolve();
        }, ms);
        signal?.addEventListener('abort', onAbort, { once: true });
    });
}

/**
 * The default retry policy: rate limits, timeouts, and server-side (5xx)
 * failures. Authentication, context-length, content-filter and other client
 * errors are final — retrying them only costs time.
 */
export function isRetryableLMError(error: unknown): boolean {
    if (error instanceof RateLimitError || error instanceof TimeoutError) return true;
    return error instanceof LMError && error.status !== undefined && error.status >= 500;
}

export interface RetryOptions extends Omit<BackoffOptions, 'signal' | 'shouldRetry'> {
    /** Override which failures are retried. Defaults to {@link isRetryableLMError}. */
    retryOn?: (error: unknown, attempt: number) => boolean;
}

/**
 * Retry failed calls with exponential backoff and jitter, honouring a
 * provider's `retryAfterMs` and the call's abort signal.
 *
 * The official OpenAI and Anthropic SDKs already retry; this is for models
 * that do not, for retrying across a {@link fallback}, or for a policy of
 * your own.
 */
export function retry(options: RetryOptions = {}): LMMiddleware {
    const { retryOn, ...backoff } = options;
    return (request, next) =>
        retryWithBackoff(() => next(request), {
            ...backoff,
            signal: request.options?.signal,
            shouldRetry: retryOn ?? isRetryableLMError,
        });
}

// ------------------------------------------------------------------ fallback

export interface FallbackOptions {
    /** Which failures move on to the next model. Defaults to any failure. */
    fallbackOn?: (error: unknown) => boolean;
}

/**
 * When a call fails, try the same request against each of `lms` in order,
 * returning the first success. Rethrows the last failure if every model fails.
 *
 * A per-call `options.model` override is passed along unchanged, so avoid one
 * when the fallbacks are different models. An aborted call is never retried
 * elsewhere.
 */
export function fallback(lms: ILanguageModel[], options: FallbackOptions = {}): LMMiddleware {
    const { fallbackOn = () => true } = options;
    return async (request, next) => {
        let lastError: unknown;
        try {
            return await next(request);
        } catch (error) {
            lastError = error;
        }

        for (const lm of lms) {
            if (request.options?.signal?.aborted || !fallbackOn(lastError)) throw lastError;
            try {
                return await dispatchLM(lm, request);
            } catch (error) {
                lastError = error;
            }
        }
        throw lastError;
    };
}

// ------------------------------------------------------------------ defaults

/**
 * Default call options, applied beneath whatever each call passes — so
 * `defaults({ temperature: 0 })` pins temperature without stopping a caller
 * from overriding it.
 */
export function defaults(options: LLMCallOptions): LMMiddleware {
    return (request, next) => next({ ...request, options: { ...options, ...request.options } });
}

// ---------------------------------------------------------------- rate limit

export interface RateLimitOptions {
    /** Maximum calls started per minute, spaced evenly. */
    requestsPerMinute: number;
}

/**
 * Space calls evenly so no more than `requestsPerMinute` start in any minute.
 * Calls over the budget wait their turn (cut short if their signal aborts)
 * rather than failing. The budget is per middleware instance: share one
 * wrapped model to share the budget.
 */
export function rateLimit(options: RateLimitOptions): LMMiddleware {
    const { requestsPerMinute } = options;
    if (!Number.isFinite(requestsPerMinute) || requestsPerMinute <= 0) {
        throw new RangeError('rateLimit requires a positive requestsPerMinute.');
    }

    const interval = 60_000 / requestsPerMinute;
    let nextSlot = 0;

    return async (request, next) => {
        const now = Date.now();
        const slot = Math.max(now, nextSlot);
        nextSlot = slot + interval;
        if (slot > now) await sleep(slot - now, request.options?.signal);
        return next(request);
    };
}
