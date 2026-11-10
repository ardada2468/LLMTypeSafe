/** Regression tests for failure modes found in a production-readiness review. */
import { buildCacheKey, type Cache } from './cache';
import { context } from './config';
import { LMError, RateLimitError } from './errors';
import {
    defaults,
    isRetryableLMError,
    rateLimit,
    retryWithBackoff,
    wrapLM,
} from './middleware';
import { Module } from './module';
import { InMemoryVectorStore } from './retrieve';
import { toSignatureClass } from './signature';
import { Predict } from '../modules/predict';
import { MockLM } from '../test-utils';

const hi = [{ role: 'user' as const, content: 'hi' }];

describe('observers that reject', () => {
    it('does not leak an unhandled rejection from an async callback or onTrace', async () => {
        const unhandled: unknown[] = [];
        const onUnhandled = (reason: unknown) => unhandled.push(reason);
        process.on('unhandledRejection', onUnhandled);
        try {
            const lm = new MockLM({ respond: () => '[[ ## answer ## ]]\nok' });
            await context(
                {
                    lm,
                    tracing: true,
                    onTrace: (async () => {
                        throw new Error('trace exporter down');
                    }) as () => void,
                    callbacks: [
                        {
                            async onModuleEnd() {
                                throw new Error('exporter down');
                            },
                            async onLMEnd() {
                                throw new Error('exporter down');
                            },
                        },
                    ],
                },
                () => new Predict('question -> answer').forward({ question: 'q' })
            );
            await new Promise((resolve) => setImmediate(resolve));
            await new Promise((resolve) => setImmediate(resolve));
            expect(unhandled).toEqual([]);
        } finally {
            process.off('unhandledRejection', onUnhandled);
        }
    });
});

describe('a failing cache backend', () => {
    it('treats a get failure as a miss and survives a set failure', async () => {
        const broken: Cache = {
            get: async () => {
                throw new Error('redis ECONNREFUSED');
            },
            set: async () => {
                throw new Error('redis OOM');
            },
        };
        const lm = new MockLM({ responses: ['paid for'] });

        await expect(context({ cache: broken }, () => lm.chat(hi))).resolves.toBe('paid for');
        expect(lm.calls).toHaveLength(1);
    });
});

describe('cache key', () => {
    it('separates calls that differ only in tools or toolChoice', () => {
        const base = { provider: 'p', model: 'm', operation: 'chat' as const, messages: hi };
        const tools = [{ name: 't', description: 'd', parameters: { type: 'object' } }] as any;

        const plain = buildCacheKey(base);
        const withTools = buildCacheKey({ ...base, options: { tools } });
        const required = buildCacheKey({ ...base, options: { tools, toolChoice: 'required' } });

        expect(new Set([plain, withTools, required]).size).toBe(3);
        expect(buildCacheKey({ ...base, options: { tools: [] } })).toBe(plain);
    });
});

describe('retryWithBackoff', () => {
    afterEach(() => vi.useRealTimers());

    it('falls back to the default limit when maxRetries is NaN', async () => {
        let attempts = 0;
        const run = retryWithBackoff(
            async () => {
                attempts++;
                throw new RateLimitError('mock', '429');
            },
            { maxRetries: Number('unset'), initialDelayMs: 0 }
        );
        await expect(run).rejects.toBeInstanceOf(RateLimitError);
        expect(attempts).toBe(4);
    });

    it('waits out a retry-after longer than setTimeout can hold', async () => {
        vi.useFakeTimers();
        let attempts = 0;
        const run = retryWithBackoff(
            async () => {
                if (attempts++ === 0) {
                    throw new RateLimitError('mock', '429', { retryAfterMs: 2 ** 31 + 1000 });
                }
                return 'ok';
            },
            { maxRetries: 1 }
        );
        await vi.advanceTimersByTimeAsync(60_000);
        expect(attempts).toBe(1);
        await vi.advanceTimersByTimeAsync(2 ** 31);
        await expect(run).resolves.toBe('ok');
    });

    it('treats dropped connections as retryable, but not client errors', () => {
        const reset = Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' });
        expect(isRetryableLMError(reset)).toBe(true);
        expect(isRetryableLMError(new TypeError('fetch failed'))).toBe(true);
        expect(
            isRetryableLMError(new LMError('openai', 'Connection error.', { cause: reset }))
        ).toBe(true);
        expect(isRetryableLMError(new LMError('openai', 'bad request', { status: 400 }))).toBe(
            false
        );
        expect(isRetryableLMError(new LMError('openai', 'unknown'))).toBe(false);
        expect(isRetryableLMError(new Error('boom'))).toBe(false);
    });
});

describe('rateLimit', () => {
    afterEach(() => vi.useRealTimers());

    it('returns the slots of calls aborted while queued', async () => {
        vi.useFakeTimers();
        const lm = new MockLM({ respond: () => 'ok' });
        const wrapped = wrapLM(lm, rateLimit({ requestsPerMinute: 60 }));
        const controller = new AbortController();

        await wrapped.chat(hi);
        const queued = Array.from({ length: 50 }, () =>
            wrapped.chat(hi, { signal: controller.signal }).catch(() => 'aborted')
        );
        controller.abort();
        expect(new Set(await Promise.all(queued))).toEqual(new Set(['aborted']));

        const next = wrapped.chat(hi);
        await vi.advanceTimersByTimeAsync(1000);
        await expect(next).resolves.toBe('ok');
        expect(lm.calls).toHaveLength(2);
    });
});

describe('defaults', () => {
    it('is not overridden by an explicit undefined', async () => {
        const lm = new MockLM({ respond: () => 'ok' });
        const wrapped = wrapLM(lm, defaults({ temperature: 0, maxTokens: 10 }));
        await wrapped.chat(hi, { temperature: undefined, maxTokens: 5 });
        expect(lm.calls[0]!.options).toMatchObject({ temperature: 0, maxTokens: 5 });
    });
});

describe('Module.batch', () => {
    it('passes its abort signal to each forward call', async () => {
        const seen: unknown[] = [];
        class Echo extends Module {
            async forward(inputs: Record<string, any>, options?: { signal?: AbortSignal }) {
                seen.push(options?.signal);
                return inputs as any;
            }
        }
        const controller = new AbortController();
        await new Echo().batch([{ q: '1' }], { signal: controller.signal });
        expect(seen).toEqual([controller.signal]);
    });
});

describe('InMemoryVectorStore', () => {
    it('rejects a batch containing a vector of the wrong dimension', async () => {
        const vectors: Record<string, number[]> = {
            a: [1, 0, 0],
            b: [0, 1],
            c: [0, Number.NaN, 1],
        };
        const store = new InMemoryVectorStore({
            embed: async (texts: string[]) => texts.map((t) => vectors[t]!),
        });
        await store.addAll(['a']);
        await expect(store.addAll(['b'])).rejects.toThrow(RangeError);
        await expect(store.addAll(['c'])).rejects.toThrow(RangeError);
        await expect(store.retrieve('a', 1)).resolves.toHaveLength(1);
    });
});

describe('string signatures', () => {
    it('does not inherit descriptions from Object.prototype', () => {
        const fields: Record<string, any> = toSignatureClass(
            'question -> constructor, toString'
        ).getOutputFields();
        expect(fields.constructor.description).toBe('Output field: constructor');
        expect(fields.toString.type).toBe('string');
    });
});
