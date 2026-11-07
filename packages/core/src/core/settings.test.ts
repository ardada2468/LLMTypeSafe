import {
    configure,
    context,
    getCache,
    getCallbacks,
    getDefaultLM,
    isTracingEnabled,
    isUsageTrackingEnabled,
    settings,
    withSettings,
} from './config';
import { MemoryCache } from './cache';
import { Module } from './module';
import { Prediction } from './prediction';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { ReAct } from '../modules/react';
import { MockLM } from '../test-utils';
import { EchoLM } from './test-lms';

// Must run before anything in this file calls configure({ lm }).
describe('before configure()', () => {
    it('constructs modules before any model is configured', async () => {
        // A fresh module graph per test file means nothing is configured yet.
        const predict = new Predict('question -> answer');
        const cot = new ChainOfThought('question -> answer');
        const agent = new ReAct('question -> answer', { tools: {} });

        await expect(predict.forward({ question: 'Q' })).rejects.toThrow(
            /No language model configured/
        );

        const lm = new MockLM({ responses: ['answer: A', 'reasoning: thinking\nanswer: B'] });
        configure({ lm });

        expect((await predict.forward({ question: 'Q' })).answer).toBe('A');
        expect((await cot.forward({ question: 'Q' })).answer).toBe('B');
        expect(agent).toBeInstanceOf(ReAct);
    });
});

describe('settings.context', () => {
    afterEach(() => {
        configure({ cache: false, tracing: false, callbacks: [], trackUsage: false });
    });

    it('overrides the globals inside the scope only', () => {
        const globalLM = new MockLM();
        const scopedLM = new MockLM();
        configure({ lm: globalLM });

        const inside = context({ lm: scopedLM, tracing: true }, () => ({
            lm: getDefaultLM(),
            tracing: isTracingEnabled(),
        }));

        expect(inside).toEqual({ lm: scopedLM, tracing: true });
        expect(getDefaultLM()).toBe(globalLM);
        expect(isTracingEnabled()).toBe(false);
    });

    it('keeps concurrent scopes isolated across awaits', async () => {
        const a = new MockLM();
        const b = new MockLM();
        const seen = await Promise.all(
            [a, b, a, b].map((lm, index) =>
                context({ lm }, async () => {
                    // Interleave the scopes: later ones resume first.
                    await new Promise((resolve) => setTimeout(resolve, 20 - index * 5));
                    return getDefaultLM();
                })
            )
        );

        expect(seen).toEqual([a, b, a, b]);
    });

    it('runs different models for concurrent invocations of one module', async () => {
        const fast = new EchoLM({}, 'fast');
        const slow = new EchoLM({ x: { delayMs: 20 } }, 'slow');
        const predict = new Predict('question -> answer');

        const [first, second] = await Promise.all([
            context({ lm: slow }, () => predict.forward({ question: 'x' })),
            context({ lm: fast }, () => predict.forward({ question: 'y' })),
        ]);

        expect(first.answer).toBe('x');
        expect(second.answer).toBe('y');
        expect(slow.prompts).toHaveLength(1);
        expect(fast.prompts).toHaveLength(1);
    });

    it('nests, inheriting the outer scope and ignoring undefined fields', () => {
        const outer = new MockLM();
        const result = context({ lm: outer, tracing: true }, () =>
            context({ lm: undefined, trackUsage: true }, () => ({
                lm: getDefaultLM(),
                tracing: isTracingEnabled(),
                usage: isUsageTrackingEnabled(),
            }))
        );

        expect(result).toEqual({ lm: outer, tracing: true, usage: true });
    });

    it('replaces rather than extends the callback list', () => {
        const global = {};
        configure({ callbacks: [global] });

        expect(context({ callbacks: [] }, () => getCallbacks())).toEqual([]);
        expect(getCallbacks()).toEqual([global]);
    });

    it('scopes a cache', () => {
        const cache = new MemoryCache();
        expect(context({ cache }, () => getCache())).toBe(cache);
        expect(getCache()).toBeUndefined();
    });

    it('returns whatever the function returns, sync or async', async () => {
        expect(context({}, () => 42)).toBe(42);
        await expect(withSettings({}, async () => 'done')).resolves.toBe('done');
    });

    it('exposes the resolved values through the settings object', () => {
        const lm = new MockLM();
        context({ lm, trackUsage: true }, () => {
            expect(settings.lm).toBe(lm);
            expect(settings.trackUsage).toBe(true);
            expect(settings.tracing).toBe(false);
        });
        expect(settings.context).toBe(context);
        expect(settings.configure).toBe(configure);
    });
});

describe('lazy language-model resolution', () => {
    it('resolves options.lm, then the module, then the scope, then the global', async () => {
        const globalLM = new MockLM({ responses: ['answer: global'] });
        const scopedLM = new MockLM({ responses: ['answer: scoped', 'answer: scoped'] });
        const ownLM = new MockLM({ responses: ['answer: own'] });
        const callLM = new MockLM({ responses: ['answer: call'] });
        configure({ lm: globalLM });

        const bare = new Predict('question -> answer');
        const owned = new Predict('question -> answer', ownLM);

        expect((await bare.forward({ question: 'Q' })).answer).toBe('global');
        expect(
            (await context({ lm: scopedLM }, () => bare.forward({ question: 'Q' }))).answer
        ).toBe('scoped');
        expect(
            (await context({ lm: scopedLM }, () => owned.forward({ question: 'Q' }))).answer
        ).toBe('own');
        expect(
            (
                await context({ lm: scopedLM }, () =>
                    owned.forward({ question: 'Q' }, { lm: callLM })
                )
            ).answer
        ).toBe('call');
        expect(
            (
                await context({ lm: scopedLM }, () =>
                    owned.withLM(scopedLM).forward({ question: 'Q' })
                )
            ).answer
        ).toBe('scoped');
    });

    it('never forwards options.lm to the provider', async () => {
        const lm = new MockLM({ responses: ['answer: A'] });
        await new Predict('question -> answer').forward(
            { question: 'Q' },
            { lm, temperature: 0 }
        );

        expect(lm.calls[0].options).toEqual({ temperature: 0 });
    });

    it('lets a custom module resolve its model at call time', async () => {
        class Raw extends Module {
            async forward(inputs: Record<string, any>) {
                const reply = await this.resolveLM().generate(String(inputs.text));
                return new Prediction({ reply });
            }
        }
        const lm = new MockLM({ responses: ['hi'] });
        const module = new Raw();

        const result = await context({ lm }, () => module.forward({ text: 'hello' }));
        expect(result.get('reply')).toBe('hi');
    });
});
