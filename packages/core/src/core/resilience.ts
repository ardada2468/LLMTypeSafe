import type { StreamChunk, UsageStats } from '../types/language-model';
import { TsDspyError } from './errors';
import { isRetryableLMError, type LMMiddleware } from './middleware';

// ----------------------------------------------------------- circuit breaker

export type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * Thrown by {@link circuitBreaker} when a call is rejected without being
 * sent, because the model behind it has been failing.
 */
export class CircuitOpenError extends TsDspyError {
    /** Milliseconds until the breaker lets a probe call through. */
    readonly retryAfterMs: number;

    constructor(retryAfterMs: number) {
        super(
            `Circuit breaker is open; the model has been failing. ` +
                `Next attempt allowed in ${Math.ceil(retryAfterMs)}ms.`
        );
        this.retryAfterMs = retryAfterMs;
    }
}

export interface CircuitBreakerOptions {
    /** Consecutive failures that open the circuit. Default 5. */
    failureThreshold?: number;
    /** How long the circuit stays open before one probe call is let through. Default 30s. */
    resetTimeoutMs?: number;
    /**
     * Which errors count against the model. Defaults to
     * {@link isRetryableLMError} — rate limits, timeouts and 5xx — so a caller's
     * own mistakes (a bad key, an oversized prompt) never open the circuit.
     */
    isFailure?: (error: unknown) => boolean;
    /** Called on every state transition, for metrics and alerting. */
    onStateChange?: (state: CircuitState, previous: CircuitState) => void;
}

/**
 * Stop calling a model that keeps failing. After `failureThreshold`
 * consecutive failures the circuit opens and calls reject immediately with
 * {@link CircuitOpenError}; after `resetTimeoutMs` one probe call is let
 * through, and its outcome closes the circuit or re-opens it.
 *
 * Failing fast keeps a struggling provider from holding every request for its
 * full timeout, and gives {@link fallback} something to act on at once:
 *
 * ```ts
 * wrapLM(primary, fallback([secondary]), circuitBreaker(), retry());
 * ```
 *
 * State is per middleware instance: share one wrapped model to share it.
 */
export function circuitBreaker(options: CircuitBreakerOptions = {}): LMMiddleware {
    const {
        failureThreshold = 5,
        resetTimeoutMs = 30_000,
        isFailure = isRetryableLMError,
        onStateChange,
    } = options;
    if (!Number.isInteger(failureThreshold) || failureThreshold < 1) {
        throw new RangeError('circuitBreaker requires a failureThreshold of at least 1.');
    }
    if (!Number.isFinite(resetTimeoutMs) || resetTimeoutMs < 0) {
        throw new RangeError('circuitBreaker requires a non-negative resetTimeoutMs.');
    }

    let state: CircuitState = 'closed';
    let failures = 0;
    let openedAt = 0;
    let probing = false;

    const transition = (next: CircuitState) => {
        if (next === state) return;
        const previous = state;
        state = next;
        // An observer that throws must not change the outcome of a model call.
        try {
            onStateChange?.(next, previous);
        } catch {
            /* ignored */
        }
    };

    const open = () => {
        openedAt = Date.now();
        failures = 0;
        transition('open');
    };

    return async (request, next) => {
        let isProbe = false;
        if (state !== 'closed') {
            const remaining = openedAt + resetTimeoutMs - Date.now();
            if (remaining > 0 || probing) {
                throw new CircuitOpenError(Math.max(remaining, 0));
            }
            probing = true;
            isProbe = true;
            transition('half-open');
        }

        try {
            const result = await next(request);
            failures = 0;
            if (isProbe) transition('closed');
            return result;
        } catch (error) {
            if (request.options?.signal?.aborted) {
                // The caller gave up; that says nothing about the model. A
                // cancelled probe leaves the circuit open for the next call
                // to probe again.
                if (isProbe) transition('open');
            } else if (isFailure(error)) {
                if (isProbe || ++failures >= failureThreshold) open();
            } else {
                // The model answered, just not happily (4xx, validation).
                failures = 0;
                if (isProbe) transition('closed');
            }
            throw error;
        } finally {
            if (isProbe) probing = false;
        }
    };
}

// --------------------------------------------------------- concurrency limit

export interface ConcurrencyLimitOptions {
    /** Maximum calls in flight at once. */
    maxConcurrent: number;
}

/**
 * Allow at most `maxConcurrent` calls in flight; the rest wait their turn in
 * arrival order (cut short if their signal aborts). A streaming call holds its
 * slot until the stream is finished or abandoned.
 *
 * Use it alongside {@link rateLimit}: that one spaces out when calls start,
 * this one bounds how many are open — which is what provider concurrency
 * quotas and connection pools actually limit.
 */
export function concurrencyLimit(options: ConcurrencyLimitOptions): LMMiddleware {
    const { maxConcurrent } = options;
    if (!Number.isInteger(maxConcurrent) || maxConcurrent < 1) {
        throw new RangeError('concurrencyLimit requires a maxConcurrent of at least 1.');
    }

    let active = 0;
    const waiting: Array<() => void> = [];

    const acquire = (signal?: AbortSignal): Promise<void> => {
        if (signal?.aborted) return Promise.reject(signal.reason);
        if (active < maxConcurrent) {
            active++;
            return Promise.resolve();
        }
        return new Promise((resolve, reject) => {
            const grant = () => {
                signal?.removeEventListener('abort', onAbort);
                resolve();
            };
            const onAbort = () => {
                const index = waiting.indexOf(grant);
                if (index !== -1) waiting.splice(index, 1);
                reject(signal?.reason);
            };
            waiting.push(grant);
            signal?.addEventListener('abort', onAbort, { once: true });
        });
    };

    const release = () => {
        // Hand the slot straight to the next waiter, so `active` never dips
        // and a new arrival cannot jump the queue.
        const grant = waiting.shift();
        if (grant) grant();
        else active--;
    };

    return async (request, next) => {
        await acquire(request.options?.signal);
        let result: unknown;
        try {
            result = await next(request);
        } catch (error) {
            release();
            throw error;
        }
        if (request.method === 'chatStream') {
            return releaseWhenDone(
                result as AsyncGenerator<StreamChunk, void, unknown>,
                release
            );
        }
        release();
        return result;
    };
}

/**
 * Wrap a stream so `release` runs exactly once, when the stream ends, throws,
 * or is abandoned — including abandoned before its first read, which a
 * generator function's `finally` would miss.
 */
function releaseWhenDone(
    stream: AsyncGenerator<StreamChunk, void, unknown>,
    release: () => void
): AsyncGenerator<StreamChunk, void, unknown> {
    let released = false;
    const done = () => {
        if (released) return;
        released = true;
        release();
    };

    const wrapped: AsyncGenerator<StreamChunk, void, unknown> = {
        async next(value?: unknown) {
            try {
                const step = await stream.next(value);
                if (step.done) done();
                return step;
            } catch (error) {
                done();
                throw error;
            }
        },
        async return(value) {
            try {
                return await stream.return(value);
            } finally {
                done();
            }
        },
        async throw(error) {
            try {
                return await stream.throw(error);
            } finally {
                done();
            }
        },
        [Symbol.asyncIterator]() {
            return wrapped;
        },
    };
    return wrapped;
}

// -------------------------------------------------------------------- budget

/** Which limit a {@link BudgetExceededError} tripped. */
export type BudgetLimit = 'requests' | 'tokens' | 'cost';

/** Thrown by {@link budget} when a call is rejected for exceeding a limit. */
export class BudgetExceededError extends TsDspyError {
    readonly limit: BudgetLimit;
    /** The configured ceiling. */
    readonly max: number;
    /** What had been spent when the call was rejected. */
    readonly spent: number;

    constructor(limit: BudgetLimit, max: number, spent: number) {
        super(`LM budget exceeded: ${limit} at ${spent}, limit ${max}.`);
        this.limit = limit;
        this.max = max;
        this.spent = spent;
    }
}

export interface BudgetOptions {
    /** Maximum calls sent through this middleware. */
    maxRequests?: number;
    /** Maximum total tokens (prompt + completion). Needs `usage`. */
    maxTokens?: number;
    /**
     * Maximum estimated spend, in the currency of your `configure({ pricing })`
     * table. Needs `usage`, and a price for the model — an unpriced model
     * reports no cost and so never trips this limit.
     */
    maxCost?: number;
    /**
     * Where to read spend from: usually the model being wrapped. Required for
     * `maxTokens` and `maxCost`.
     */
    usage?: { getUsage(): UsageStats };
}

/**
 * Refuse further calls once a spending limit is reached, with
 * {@link BudgetExceededError} — a hard stop for a runaway agent loop or an
 * optimizer run that would otherwise bill until someone notices.
 *
 * ```ts
 * const base = new OpenAILM({ model: 'gpt-5.2' });
 * const lm = wrapLM(base, budget({ usage: base, maxCost: 5, maxRequests: 2_000 }));
 * ```
 *
 * Limits are checked before each call, against spend measured from when the
 * middleware was created; a call already in flight when the limit is crossed
 * completes, so the final total can overshoot by the calls in flight. Cache
 * hits cost nothing and count only against `maxRequests`.
 */
export function budget(options: BudgetOptions): LMMiddleware {
    const { maxRequests, maxTokens, maxCost, usage } = options;
    for (const [name, value] of Object.entries({ maxRequests, maxTokens, maxCost })) {
        if (value !== undefined && !(Number.isFinite(value) && value >= 0)) {
            throw new RangeError(`budget requires ${name} to be a non-negative number.`);
        }
    }
    if (maxRequests === undefined && maxTokens === undefined && maxCost === undefined) {
        throw new RangeError(
            'budget requires at least one of maxRequests, maxTokens, maxCost.'
        );
    }
    if ((maxTokens !== undefined || maxCost !== undefined) && !usage) {
        throw new RangeError(
            'budget needs `usage` (the model being wrapped) for maxTokens or maxCost.'
        );
    }

    // Spend is accumulated from deltas between observations, so a
    // `resetUsage()` on the model does not hand the budget back.
    let seenTokens = 0;
    let seenCost = 0;
    let tokens = 0;
    let cost = 0;
    const observe = () => {
        if (!usage) return;
        const now = usage.getUsage();
        const nowCost = now.totalCost ?? 0;
        tokens +=
            now.totalTokens >= seenTokens ? now.totalTokens - seenTokens : now.totalTokens;
        cost += nowCost >= seenCost ? nowCost - seenCost : nowCost;
        seenTokens = now.totalTokens;
        seenCost = nowCost;
    };
    // Whatever was spent before the budget existed is not charged to it.
    observe();
    tokens = 0;
    cost = 0;
    let requests = 0;

    return async (request, next) => {
        if (maxRequests !== undefined && requests >= maxRequests) {
            throw new BudgetExceededError('requests', maxRequests, requests);
        }
        observe();
        if (maxTokens !== undefined && tokens >= maxTokens) {
            throw new BudgetExceededError('tokens', maxTokens, tokens);
        }
        if (maxCost !== undefined && cost >= maxCost) {
            throw new BudgetExceededError('cost', maxCost, cost);
        }
        requests++;
        return next(request);
    };
}
