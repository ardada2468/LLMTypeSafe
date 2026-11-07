import { BestOfN } from './best-of-n';
import { Predict } from './predict';
import { ChainOfThought } from './chain-of-thought';
import { Module } from '../core/module';
import { MemoryCache, buildCacheKey } from '../core/cache';
import { context } from '../core/config';
import { MockLM } from '../test-utils';
import type { LLMCallOptions } from '../types/language-model';

const answer = (value: string) => `[[ ## answer ## ]]\n${value}\n\n[[ ## completed ## ]]`;

/** Replies with the next scripted answer, recording the options of every call. */
function scripted(answers: string[]): MockLM {
    let index = 0;
    return new MockLM({ respond: () => answer(answers[index++] ?? 'exhausted') });
}

const wordCount = (_inputs: Record<string, any>, prediction: any) =>
    String(prediction.answer).split(/\s+/).length === 1 ? 1 : 0;

describe('BestOfN', () => {
    it('returns the first prediction that reaches the threshold and stops there', async () => {
        const lm = scripted(['two words', 'Paris', 'never asked']);
        const best = new BestOfN(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: wordCount,
            threshold: 1,
        });

        const result = await best.forward({ question: 'Capital of France?' });

        expect(result.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(2);
    });

    it('returns the highest-rewarded prediction when none reaches the threshold', async () => {
        const lm = scripted(['a b c', 'a', 'a b']);
        const best = new BestOfN(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: (_inputs, prediction: any) => -String(prediction.answer).length,
            threshold: 100,
        });

        const result = await best.forward({ question: 'Q' });

        expect(result.answer).toBe('a');
        expect(lm.calls).toHaveLength(3);
    });

    it('stamps every call with the rollout temperature and a distinct rollout id', async () => {
        const lm = scripted(['x y', 'x y', 'x y']);
        await new BestOfN(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: wordCount,
            threshold: 1,
        }).forward({ question: 'Q' }, { temperature: 0 });

        expect(lm.calls.map((call) => call.options?.temperature)).toEqual([1, 1, 1]);
        expect(lm.calls.map((call) => call.options?.metadata?.rolloutId)).toEqual([0, 1, 2]);
    });

    it('honours a custom temperature', async () => {
        const lm = scripted(['Paris']);
        await new BestOfN(new Predict('question -> answer', lm), {
            n: 2,
            rewardFn: wordCount,
            threshold: 1,
            temperature: 0.7,
        }).forward({ question: 'Q' });

        expect(lm.calls[0].options?.temperature).toBe(0.7);
    });

    it('gets a fresh sample per rollout even with the response cache on', async () => {
        const lm = scripted(['first try', 'Paris']);
        const program = new BestOfN(new Predict('question -> answer', lm), {
            n: 2,
            rewardFn: wordCount,
            threshold: 1,
        });

        const result = await context({ cache: new MemoryCache() }, () =>
            program.forward({ question: 'Q' })
        );

        expect(result.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(2);
    });

    it('replays each rollout from the cache when the whole program runs again', async () => {
        const lm = scripted(['first try', 'Paris']);
        const cache = new MemoryCache();
        const program = new BestOfN(new Predict('question -> answer', lm), {
            n: 2,
            rewardFn: wordCount,
            threshold: 1,
        });

        await context({ cache }, () => program.forward({ question: 'Q' }));
        const again = await context({ cache }, () => program.forward({ question: 'Q' }));

        expect(again.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(2);
    });

    it('wraps the scope model, and a per-call model, as well as a module’s own', async () => {
        const scoped = scripted(['a b', 'Paris']);
        await context({ lm: scoped }, () =>
            new BestOfN(new Predict('question -> answer'), {
                n: 2,
                rewardFn: wordCount,
                threshold: 1,
            }).forward({ question: 'Q' })
        );
        expect(scoped.calls.map((call) => call.options?.metadata?.rolloutId)).toEqual([0, 1]);

        const perCall = scripted(['a b', 'Paris']);
        await new BestOfN(new Predict('question -> answer'), {
            n: 2,
            rewardFn: wordCount,
            threshold: 1,
        }).forward({ question: 'Q' }, { lm: perCall });
        expect(perCall.calls.map((call) => call.options?.metadata?.rolloutId)).toEqual([0, 1]);
    });

    it('reaches predictors inside a hand-written forward that drops its options', async () => {
        const lm = scripted(['a b', 'Paris']);
        class Pipeline extends Module {
            step = new Predict('question -> answer', lm);

            async forward(inputs: Record<string, any>, _options?: LLMCallOptions) {
                return this.step.forward(inputs);
            }
        }

        const result = await new BestOfN(new Pipeline(), {
            n: 2,
            rewardFn: wordCount,
            threshold: 1,
        }).forward({ question: 'Q' });

        expect(result.answer).toBe('Paris');
        expect(lm.calls.map((call) => call.options?.temperature)).toEqual([1, 1]);
    });

    it('leaves the wrapped module untouched', async () => {
        const lm = scripted(['Paris', 'Paris']);
        const inner = new Predict('question -> answer', lm);
        await new BestOfN(inner, { n: 1, rewardFn: wordCount, threshold: 1 }).forward({
            question: 'Q',
        });

        await inner.forward({ question: 'Q' });
        expect(lm.calls[1].options?.metadata).toBeUndefined();
    });

    it('tolerates failures up to failCount', async () => {
        let call = 0;
        const lm = new MockLM({
            respond: () => {
                call += 1;
                if (call === 1) throw new Error('provider down');
                return answer('Paris');
            },
        });

        const result = await new BestOfN(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: wordCount,
            threshold: 1,
            failCount: 1,
        }).forward({ question: 'Q' });

        expect(result.answer).toBe('Paris');
    });

    it('rethrows once more than failCount attempts have failed', async () => {
        const lm = new MockLM({
            respond: () => {
                throw new Error('provider down');
            },
        });

        await expect(
            new BestOfN(new Predict('question -> answer', lm), {
                n: 3,
                rewardFn: wordCount,
                threshold: 1,
                failCount: 1,
            }).forward({ question: 'Q' })
        ).rejects.toThrow('provider down');
        expect(lm.calls).toHaveLength(2);
    });

    it('rethrows the last error when every attempt failed within failCount', async () => {
        const lm = new MockLM({
            respond: () => {
                throw new Error('provider down');
            },
        });

        await expect(
            new BestOfN(new Predict('question -> answer', lm), {
                n: 2,
                rewardFn: wordCount,
                threshold: 1,
            }).forward({ question: 'Q' })
        ).rejects.toThrow('provider down');
        expect(lm.calls).toHaveLength(2);
    });

    it('counts a throwing reward function as a failed attempt', async () => {
        const lm = scripted(['a', 'b']);
        let calls = 0;
        const result = await new BestOfN(new Predict('question -> answer', lm), {
            n: 2,
            rewardFn: async () => {
                calls += 1;
                if (calls === 1) throw new Error('judge failed');
                return 1;
            },
            threshold: 1,
        }).forward({ question: 'Q' });

        expect(result.answer).toBe('b');
    });

    it('exposes the wrapped module’s predictors', () => {
        const best = new BestOfN(new ChainOfThought('question -> answer'), {
            n: 2,
            rewardFn: wordCount,
            threshold: 1,
        });

        expect(best.namedPredictors().map(([path]) => path)).toEqual(['module']);
    });

    it('validates its options', () => {
        const module = new Predict('question -> answer');
        expect(() => new BestOfN(module, { n: 0, rewardFn: wordCount, threshold: 1 })).toThrow(
            RangeError
        );
        expect(
            () => new BestOfN(module, { n: 1, rewardFn: undefined as never, threshold: 1 })
        ).toThrow(/rewardFn/);
    });
});

describe('buildCacheKey and rollout ids', () => {
    const base = {
        provider: 'mock',
        model: 'm',
        operation: 'chat' as const,
        messages: [{ role: 'user' as const, content: 'hi' }],
    };

    it('keeps the key unchanged when no rollout id is set', () => {
        expect(buildCacheKey({ ...base, options: { metadata: { traceId: 'x' } } })).toBe(
            buildCacheKey(base)
        );
    });

    it('distinguishes rollouts', () => {
        const a = buildCacheKey({ ...base, options: { metadata: { rolloutId: 1 } } });
        const b = buildCacheKey({ ...base, options: { metadata: { rolloutId: 2 } } });
        expect(a).not.toBe(b);
        expect(a).not.toBe(buildCacheKey(base));
    });
});
