import {
    defaults,
    fallback,
    isRetryableLMError,
    rateLimit,
    retry,
    retryWithBackoff,
    wrapLM,
    type LMMiddleware,
} from './middleware';
import { AuthError, LMError, RateLimitError, TimeoutError } from './errors';
import { BaseLM } from './base-lm';
import { Predict } from '../modules/predict';
import { RespAct } from '../modules/respact';
import { MockLM } from '../test-utils';

/** Advance fake time until `promise` settles, returning its outcome. */
async function settle<T>(promise: Promise<T>): Promise<{ value?: T; error?: unknown }> {
    let outcome: { value?: T; error?: unknown } | undefined;
    promise.then(
        (value) => (outcome = { value }),
        (error) => (outcome = { error })
    );
    for (let i = 0; i < 100 && !outcome; i++) {
        await vi.runOnlyPendingTimersAsync();
    }
    return outcome!;
}

/** A MockLM whose first `failures` chat calls throw. */
function flaky(failures: unknown[], responses: string[]): MockLM {
    const lm = new MockLM({ responses });
    const chat = lm.chat.bind(lm);
    vi.spyOn(lm, 'chat').mockImplementation(async (messages, options) => {
        const failure = failures.shift();
        if (failure) throw failure;
        return chat(messages, options);
    });
    return lm;
}

describe('retry', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.spyOn(Math, 'random').mockReturnValue(1);
    });
    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('retries retryable failures with exponential backoff', async () => {
        const lm = flaky(
            [new RateLimitError('mock', '429'), new LMError('mock', '503', { status: 503 })],
            ['ok']
        );
        const wrapped = wrapLM(lm, retry({ initialDelayMs: 100 }));
        const timeouts: number[] = [];
        const realSetTimeout = globalThis.setTimeout;
        vi.spyOn(globalThis, 'setTimeout').mockImplementation(((fn: () => void, ms: number) => {
            timeouts.push(ms);
            return realSetTimeout(fn, ms);
        }) as typeof setTimeout);

        const { value } = await settle(wrapped.generate('hi'));

        expect(value).toBe('ok');
        expect(lm.chat).toHaveBeenCalledTimes(3);
        // Math.random() pinned to 1, so jitter yields the full delay.
        expect(timeouts).toEqual([100, 200]);
    });

    it('applies jitter between half and all of the delay', async () => {
        vi.spyOn(Math, 'random').mockReturnValue(0);
        const delays: number[] = [];
        const realSetTimeout = globalThis.setTimeout;
        vi.spyOn(globalThis, 'setTimeout').mockImplementation(((fn: () => void, ms: number) => {
            delays.push(ms);
            return realSetTimeout(fn, ms);
        }) as typeof setTimeout);

        const attempt = vi
            .fn()
            .mockRejectedValueOnce(new TimeoutError('mock', 'slow'))
            .mockResolvedValue('ok');
        await settle(retryWithBackoff(attempt, { initialDelayMs: 1000 }));

        expect(delays).toEqual([500]);
    });

    it('caps the backoff at maxDelayMs but honours a longer retryAfterMs', async () => {
        const delays: number[] = [];
        const realSetTimeout = globalThis.setTimeout;
        vi.spyOn(globalThis, 'setTimeout').mockImplementation(((fn: () => void, ms: number) => {
            delays.push(ms);
            return realSetTimeout(fn, ms);
        }) as typeof setTimeout);

        const attempt = vi
            .fn()
            .mockRejectedValueOnce(new LMError('mock', '500', { status: 500 }))
            .mockRejectedValueOnce(new RateLimitError('mock', '429', { retryAfterMs: 7_000 }))
            .mockResolvedValue('ok');
        const { value } = await settle(
            retryWithBackoff(attempt, { initialDelayMs: 4_000, maxDelayMs: 5_000 })
        );

        expect(value).toBe('ok');
        expect(delays).toEqual([4_000, 7_000]);
    });

    it('gives up after maxRetries and rethrows the last failure', async () => {
        const lm = flaky(
            [1, 2, 3].map((n) => new RateLimitError('mock', `429 #${n}`)),
            ['never']
        );
        const { error } = await settle(wrapLM(lm, retry({ maxRetries: 2 })).generate('hi'));

        expect((error as Error).message).toContain('429 #3');
        expect(lm.chat).toHaveBeenCalledTimes(3);
    });

    it('does not retry final errors', async () => {
        const lm = flaky([new AuthError('mock', 'bad key', { status: 401 })], ['never']);
        const { error } = await settle(wrapLM(lm, retry()).generate('hi'));

        expect(error).toBeInstanceOf(AuthError);
        expect(lm.chat).toHaveBeenCalledTimes(1);
        expect(isRetryableLMError(new LMError('mock', 'x', { status: 400 }))).toBe(false);
        expect(isRetryableLMError(new Error('plain'))).toBe(false);
    });

    it('accepts a custom retryOn policy', async () => {
        const lm = flaky([new Error('custom transient')], ['ok']);
        const { value } = await settle(
            wrapLM(
                lm,
                retry({ retryOn: (error) => (error as Error).message.includes('transient') })
            ).generate('hi')
        );

        expect(value).toBe('ok');
    });

    it('stops waiting as soon as the call is aborted', async () => {
        const controller = new AbortController();
        const lm = flaky([new RateLimitError('mock', '429')], ['ok']);
        const pending = wrapLM(lm, retry({ initialDelayMs: 60_000 })).generate('hi', {
            signal: controller.signal,
        });
        const outcome = settleLater(pending);

        await vi.advanceTimersByTimeAsync(10);
        controller.abort(new Error('cancelled'));
        await vi.advanceTimersByTimeAsync(0);

        expect((await outcome).error).toEqual(new Error('cancelled'));
        expect(lm.chat).toHaveBeenCalledTimes(1);
    });

    it('rejects before any attempt when already aborted', async () => {
        const attempt = vi.fn();
        const controller = new AbortController();
        controller.abort(new Error('early'));

        await expect(retryWithBackoff(attempt, { signal: controller.signal })).rejects.toThrow(
            'early'
        );
        expect(attempt).not.toHaveBeenCalled();
    });
});

function settleLater<T>(promise: Promise<T>): Promise<{ value?: T; error?: unknown }> {
    return promise.then(
        (value) => ({ value }),
        (error) => ({ error })
    );
}

describe('fallback', () => {
    it('tries each fallback model in order until one succeeds', async () => {
        const primary = flaky([new LMError('mock', 'down', { status: 503 })], []);
        const second = flaky([new LMError('mock', 'also down')], []);
        const third = new MockLM({ responses: ['from third'] });

        const wrapped = wrapLM(primary, fallback([second, third]));

        expect(await wrapped.chat([{ role: 'user', content: 'hi' }])).toBe('from third');
        expect(second.chat).toHaveBeenCalledTimes(1);
    });

    it('rethrows the last failure when every model fails', async () => {
        const primary = flaky([new Error('one')], []);
        const backup = flaky([new Error('two')], []);

        await expect(wrapLM(primary, fallback([backup])).generate('hi')).rejects.toThrow('two');
    });

    it('respects fallbackOn', async () => {
        const primary = flaky([new AuthError('mock', 'bad key')], []);
        const backup = new MockLM({ responses: ['backup'] });

        await expect(
            wrapLM(
                primary,
                fallback([backup], { fallbackOn: (error) => !(error instanceof AuthError) })
            ).generate('hi')
        ).rejects.toBeInstanceOf(AuthError);
        expect(backup.calls).toHaveLength(0);
    });

    it('serves a Predict end to end, with structured calls routed too', async () => {
        const primary = new MockLM({ capabilities: { supportsStructuredOutput: true } });
        vi.spyOn(primary, 'generateStructured').mockRejectedValue(new Error('down'));
        const backup = new MockLM({ structuredResponses: [{ answer: 'Paris' }] });

        const result = await new Predict(
            'question -> answer',
            wrapLM(primary, fallback([backup]))
        ).forward({ question: 'Capital?' });

        expect(result.answer).toBe('Paris');
    });
});

describe('wrapLM', () => {
    it('delegates everything that is not a call', () => {
        const lm = new MockLM();
        lm.setModel('special');
        const wrapped = wrapLM(lm);

        expect(wrapped.getModelName()).toBe('special');
        expect(wrapped.getCapabilities()).toBe(lm.getCapabilities());
        expect(wrapped.getUsage()).toEqual(lm.getUsage());
        wrapped.setModel!('other');
        expect(lm.getModelName()).toBe('other');
    });

    it('runs middleware outermost first and lets it rewrite requests', async () => {
        const order: string[] = [];
        const tag =
            (name: string): LMMiddleware =>
            async (request, next) => {
                order.push(`${name}:in`);
                const result = await next(request);
                order.push(`${name}:out`);
                return result;
            };
        const lm = new MockLM({ responses: ['ok'] });

        await wrapLM(
            lm,
            tag('a'),
            tag('b'),
            defaults({ temperature: 0, maxTokens: 5 })
        ).generate('hi', { maxTokens: 9 });

        expect(order).toEqual(['a:in', 'b:in', 'b:out', 'a:out']);
        expect(lm.calls[0].options).toEqual({ temperature: 0, maxTokens: 9 });
    });

    it('can short-circuit without calling the model', async () => {
        const lm = new MockLM();
        const wrapped = wrapLM(lm, async () => 'cached elsewhere');

        expect(await wrapped.generate('hi')).toBe('cached elsewhere');
        expect(lm.calls).toHaveLength(0);
    });

    it('keeps streaming available and retries a stream that fails to open', async () => {
        const lm = new MockLM({ responses: ['answer: streamed'] });
        const realStream = lm.chatStream.bind(lm);
        let opened = 0;
        vi.spyOn(lm, 'chatStream').mockImplementation(async function* (messages, options) {
            opened += 1;
            if (opened === 1) throw new RateLimitError('mock', '429');
            yield* realStream(messages, options);
        });
        const wrapped = wrapLM(lm, retry({ initialDelayMs: 0, jitter: false }));

        const chunks: string[] = [];
        for await (const partial of new Predict('question -> answer', wrapped).stream({
            question: 'Q',
        })) {
            chunks.push(String(partial.answer));
        }

        expect(opened).toBe(2);
        expect(chunks.at(-1)).toBe('streamed');
    });

    it('does not advertise optional methods the model lacks', () => {
        const lm = new MockLM();
        (lm as any).chatStream = undefined;
        (lm as any).generateStream = undefined;
        const wrapped = wrapLM(lm);

        expect(wrapped.chatStream).toBeUndefined();
        expect(wrapped.listModels).toBeUndefined();
    });

    it('keeps a model without native tool calling on the RespAct text loop', async () => {
        const lm = new MockLM({
            responses: ['Final Answer: answer: done'],
            capabilities: { supportsFunctionCalling: true },
        });
        const wrapped = wrapLM(lm);

        expect(wrapped.chatWithTools).toBe(BaseLM.prototype.chatWithTools);
        const result = await new RespAct('question -> answer', {
            lm: wrapped,
            tools: { noop: () => 'x' },
        }).forward({ question: 'Q' });
        expect(result.answer).toBe('done');
    });
});

describe('rateLimit', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('spaces calls evenly across the minute', async () => {
        const lm = new MockLM({ responses: ['1', '2', '3'] });
        const wrapped = wrapLM(lm, rateLimit({ requestsPerMinute: 60 }));

        const all = Promise.all([
            wrapped.generate('a'),
            wrapped.generate('b'),
            wrapped.generate('c'),
        ]);
        await vi.advanceTimersByTimeAsync(0);
        expect(lm.calls).toHaveLength(1);
        await vi.advanceTimersByTimeAsync(1000);
        expect(lm.calls).toHaveLength(2);
        await vi.advanceTimersByTimeAsync(1000);
        expect(await all).toEqual(['1', '2', '3']);
    });

    it('rejects a non-positive budget', () => {
        expect(() => rateLimit({ requestsPerMinute: 0 })).toThrow(RangeError);
    });
});
