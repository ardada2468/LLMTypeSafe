import {
    budget,
    BudgetExceededError,
    circuitBreaker,
    CircuitOpenError,
    concurrencyLimit,
    type CircuitState,
} from './resilience';
import { fallback, wrapLM } from './middleware';
import { AuthError, LMError } from './errors';
import { MockLM } from '../test-utils';

const down = () => new LMError('mock', '503', { status: 503 });

/** A MockLM whose chat calls throw whatever `failure()` returns, while it returns anything. */
function failing(failure: () => unknown, responses: string[] = []): MockLM {
    const lm = new MockLM({ responses, respond: () => 'ok' });
    const chat = lm.chat.bind(lm);
    vi.spyOn(lm, 'chat').mockImplementation(async (messages, options) => {
        const error = failure();
        if (error) throw error;
        return chat(messages, options);
    });
    return lm;
}

const hi = [{ role: 'user' as const, content: 'hi' }];

describe('circuitBreaker', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('opens after consecutive failures and rejects without calling the model', async () => {
        const lm = failing(down);
        const wrapped = wrapLM(lm, circuitBreaker({ failureThreshold: 2 }));

        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        const error = await wrapped.chat(hi).catch((e) => e);

        expect(error).toBeInstanceOf(CircuitOpenError);
        expect(error.retryAfterMs).toBe(30_000);
        expect(lm.chat).toHaveBeenCalledTimes(2);
    });

    it('lets one probe through after the reset timeout and closes on success', async () => {
        let broken = true;
        const lm = failing(() => (broken ? down() : undefined));
        const states: CircuitState[] = [];
        const wrapped = wrapLM(
            lm,
            circuitBreaker({
                failureThreshold: 1,
                resetTimeoutMs: 1000,
                onStateChange: (state) => states.push(state),
            })
        );

        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        vi.advanceTimersByTime(999);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(CircuitOpenError);

        vi.advanceTimersByTime(1);
        broken = false;
        await expect(wrapped.chat(hi)).resolves.toBe('ok');
        await expect(wrapped.chat(hi)).resolves.toBe('ok');
        expect(states).toEqual(['open', 'half-open', 'closed']);
    });

    it('re-opens when the probe fails, and admits only one probe at a time', async () => {
        let release!: (error: unknown) => void;
        let calls = 0;
        const lm = new MockLM({ respond: () => 'ok' });
        vi.spyOn(lm, 'chat').mockImplementation(() => {
            calls++;
            if (calls === 1) return Promise.reject(down());
            return new Promise((_, reject) => (release = reject));
        });
        const wrapped = wrapLM(
            lm,
            circuitBreaker({ failureThreshold: 1, resetTimeoutMs: 100 })
        );

        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        vi.advanceTimersByTime(100);

        const probe = wrapped.chat(hi).catch((e) => e);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(CircuitOpenError);
        release(down());
        expect(await probe).toBeInstanceOf(LMError);

        const error = await wrapped.chat(hi).catch((e) => e);
        expect(error).toBeInstanceOf(CircuitOpenError);
        expect(error.retryAfterMs).toBe(100);
        expect(calls).toBe(2);
    });

    it('does not count client errors or reset-interrupted streaks', async () => {
        const errors: unknown[] = [
            down(),
            new AuthError('mock', '401', { status: 401 }),
            down(),
        ];
        const lm = failing(() => errors.shift());
        const wrapped = wrapLM(lm, circuitBreaker({ failureThreshold: 2 }));

        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(AuthError);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        await expect(wrapped.chat(hi)).resolves.toBe('ok');
    });

    it('does not count a call the caller aborted', async () => {
        const controller = new AbortController();
        const lm = failing(() => {
            controller.abort();
            return down();
        });
        const wrapped = wrapLM(lm, circuitBreaker({ failureThreshold: 1 }));

        await expect(wrapped.chat(hi, { signal: controller.signal })).rejects.toBeInstanceOf(
            LMError
        );
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        expect(lm.chat).toHaveBeenCalledTimes(2);
    });

    it('hands an open circuit to fallback immediately', async () => {
        const primary = failing(down);
        const secondary = new MockLM({ respond: () => 'from secondary' });
        const wrapped = wrapLM(
            primary,
            fallback([secondary]),
            circuitBreaker({ failureThreshold: 1 })
        );

        await expect(wrapped.chat(hi)).resolves.toBe('from secondary');
        await expect(wrapped.chat(hi)).resolves.toBe('from secondary');
        expect(primary.chat).toHaveBeenCalledTimes(1);
    });

    it('survives a throwing observer and validates its options', async () => {
        const wrapped = wrapLM(
            failing(down),
            circuitBreaker({
                failureThreshold: 1,
                onStateChange: () => {
                    throw new Error('observer bug');
                },
            })
        );
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(CircuitOpenError);

        expect(() => circuitBreaker({ failureThreshold: 0 })).toThrow(RangeError);
        expect(() => circuitBreaker({ resetTimeoutMs: -1 })).toThrow(RangeError);
    });
});

describe('concurrencyLimit', () => {
    afterEach(() => vi.restoreAllMocks());

    /** A model whose chat calls block until released, tracking peak concurrency. */
    function gated() {
        const lm = new MockLM({ respond: () => 'ok' });
        const gates: Array<() => void> = [];
        const stats = { active: 0, peak: 0, started: [] as string[] };
        vi.spyOn(lm, 'chat').mockImplementation(async (messages) => {
            stats.active++;
            stats.peak = Math.max(stats.peak, stats.active);
            stats.started.push(String(messages[0].content));
            await new Promise<void>((resolve) => gates.push(resolve));
            stats.active--;
            return 'ok';
        });
        return { lm, gates, stats };
    }
    const tick = () => new Promise((resolve) => setImmediate(resolve));
    const say = (content: string) => [{ role: 'user' as const, content }];

    it('never exceeds the limit and serves waiters in arrival order', async () => {
        const { lm, gates, stats } = gated();
        const wrapped = wrapLM(lm, concurrencyLimit({ maxConcurrent: 2 }));

        const all = Promise.all(['a', 'b', 'c', 'd'].map((c) => wrapped.chat(say(c))));
        await tick();
        expect(stats.started).toEqual(['a', 'b']);

        while (stats.started.length < 4 || stats.active > 0) {
            gates.shift()?.();
            await tick();
        }
        await expect(all).resolves.toEqual(['ok', 'ok', 'ok', 'ok']);
        expect(stats.peak).toBe(2);
        expect(stats.started).toEqual(['a', 'b', 'c', 'd']);
    });

    it('frees the slot when a call fails', async () => {
        const lm = failing(() => down());
        const wrapped = wrapLM(lm, concurrencyLimit({ maxConcurrent: 1 }));
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        await expect(wrapped.chat(hi)).rejects.toBeInstanceOf(LMError);
        expect(lm.chat).toHaveBeenCalledTimes(2);
    });

    it('drops an aborted waiter without consuming a slot', async () => {
        const { lm, gates, stats } = gated();
        const wrapped = wrapLM(lm, concurrencyLimit({ maxConcurrent: 1 }));
        const controller = new AbortController();

        const first = wrapped.chat(say('a'));
        const cancelled = wrapped.chat(say('b'), { signal: controller.signal });
        const third = wrapped.chat(say('c'));
        await tick();
        controller.abort(new Error('gone'));
        await expect(cancelled).rejects.toThrow('gone');

        gates.shift()!();
        await first;
        await tick();
        gates.shift()!();
        await third;
        expect(stats.started).toEqual(['a', 'c']);

        const aborted = AbortSignal.abort(new Error('already'));
        await expect(wrapped.chat(say('d'), { signal: aborted })).rejects.toThrow('already');
    });

    it('holds a streaming slot until the stream is finished or abandoned', async () => {
        const lm = new MockLM({ respond: () => 'one two three' });
        const wrapped = wrapLM(lm, concurrencyLimit({ maxConcurrent: 1 }));

        const stream = wrapped.chatStream!(hi);
        await stream.next();
        let secondDone = false;
        const second = wrapped.chat(hi).then(() => (secondDone = true));
        await tick();
        expect(secondDone).toBe(false);

        await stream.return(undefined);
        await second;
        expect(secondDone).toBe(true);

        // A fully consumed stream releases too.
        for await (const _chunk of wrapped.chatStream!(hi)) void _chunk;
        await expect(wrapped.chat(hi)).resolves.toBe('one two three');
    });

    it('validates its options', () => {
        expect(() => concurrencyLimit({ maxConcurrent: 0 })).toThrow(RangeError);
        expect(() => concurrencyLimit({ maxConcurrent: 1.5 })).toThrow(RangeError);
    });
});

describe('budget', () => {
    it('caps the number of requests', async () => {
        const lm = new MockLM({ respond: () => 'ok' });
        const wrapped = wrapLM(lm, budget({ maxRequests: 2 }));

        await wrapped.chat(hi);
        await wrapped.chat(hi);
        const error = await wrapped.chat(hi).catch((e) => e);

        expect(error).toBeInstanceOf(BudgetExceededError);
        expect(error).toMatchObject({ limit: 'requests', max: 2, spent: 2 });
        expect(lm.calls).toHaveLength(2);
    });

    it('caps tokens and cost from the model usage, ignoring earlier spend and resets', async () => {
        const usage = { totalTokens: 500, totalCost: 1, promptTokens: 0, completionTokens: 0 };
        const lm = new MockLM({ respond: () => 'ok' });
        const source = { getUsage: () => ({ ...usage }) };
        const tokens = wrapLM(lm, budget({ usage: source, maxTokens: 100 }));
        const cost = wrapLM(lm, budget({ usage: source, maxCost: 0.5 }));

        await expect(tokens.chat(hi)).resolves.toBe('ok');
        usage.totalTokens += 60;
        usage.totalCost += 0.3;
        await expect(tokens.chat(hi)).resolves.toBe('ok');
        await expect(cost.chat(hi)).resolves.toBe('ok');

        // The model's usage is reset: spend so far must not be forgotten.
        usage.totalTokens = 40;
        usage.totalCost = 0.2;
        await expect(tokens.chat(hi)).rejects.toMatchObject({ limit: 'tokens', spent: 100 });
        await expect(cost.chat(hi)).rejects.toMatchObject({ limit: 'cost', max: 0.5 });
    });

    it('validates its options', () => {
        expect(() => budget({})).toThrow(RangeError);
        expect(() => budget({ maxTokens: 10 })).toThrow(/usage/);
        expect(() => budget({ maxRequests: -1 })).toThrow(RangeError);
        expect(() => budget({ maxRequests: Number.NaN })).toThrow(RangeError);
    });
});
