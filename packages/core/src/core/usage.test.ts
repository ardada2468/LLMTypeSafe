import { configure, context } from './config';
import { Module } from './module';
import { type Prediction } from './prediction';
import { clearHistory } from './trace';
import { collectUsage, recordLMUsage, sumUsage } from './usage';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { evaluate } from '../evaluate/evaluate';
import { Example } from './example';
import { MockLM } from '../test-utils';
import { EchoLM } from './test-lms';

class Pipeline extends Module {
    first = new Predict('question -> answer');
    second = new Predict('question -> answer');

    async forward(inputs: Record<string, any>): Promise<Prediction> {
        const a = await this.first.forward(inputs);
        return this.second.forward({ question: `${a.answer}2` });
    }
}

describe('per-call usage', () => {
    afterEach(() => {
        configure({ trackUsage: false, tracing: false, pricing: undefined });
        clearHistory();
    });

    it('is empty, and nothing is collected, while tracking is off', async () => {
        const lm = new MockLM({ responses: ['answer: A'] });

        const result = await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(result.getLMUsage()).toEqual({});
        // The model's lifetime counters keep working regardless.
        expect(lm.getUsage().totalTokens).toBe(15);
    });

    it('reports the invocation’s usage keyed by provider/model', async () => {
        configure({ trackUsage: true });
        // Two calls in one invocation: the reply and its repair round trip.
        const lm = new MockLM({
            responses: [
                'reasoning: r\nanswer: A\nscore: high',
                'reasoning: r\nanswer: A\nscore: 1',
            ],
        });

        const result = await new ChainOfThought(
            'question -> answer, score: number',
            lm
        ).forward({ question: 'Q' }, { repairAttempts: 1 });

        expect(result.getLMUsage()).toEqual({
            'mock/mock-model': {
                promptTokens: 20,
                completionTokens: 10,
                totalTokens: 30,
                requestCount: 2,
                errorCount: 0,
                cacheHits: 0,
                averageLatency: 1,
            },
        });
    });

    it('is exact for concurrent invocations against one shared model', async () => {
        configure({ trackUsage: true });
        const lm = new EchoLM({
            slow: { delayMs: 30, promptTokens: 100, completionTokens: 10 },
            fast: { delayMs: 1, promptTokens: 7, completionTokens: 3 },
        });
        const predict = new Predict('question -> answer', lm);

        const [slow, fast] = await Promise.all([
            predict.forward({ question: 'slow' }),
            predict.forward({ question: 'fast' }),
        ]);

        expect(slow.getLMUsage()['test/echo']).toMatchObject({
            promptTokens: 100,
            completionTokens: 10,
            requestCount: 1,
        });
        expect(fast.getLMUsage()['test/echo']).toMatchObject({
            promptTokens: 7,
            completionTokens: 3,
            requestCount: 1,
        });
    });

    it('includes nested modules in the composite’s figure, and only theirs in each child’s', async () => {
        configure({ trackUsage: true });
        const lm = new EchoLM({
            q: { promptTokens: 5 },
            q2: { promptTokens: 50 },
        });
        const pipeline = new Pipeline();

        const result = await context({ lm }, () => pipeline.forward({ question: 'q' }));

        expect(result.getLMUsage()['test/echo'].promptTokens).toBe(55);
        expect(result.getLMUsage()['test/echo'].requestCount).toBe(2);
    });

    it('keeps models apart and fills cost from the pricing table', async () => {
        configure({
            trackUsage: true,
            pricing: {
                'test/cheap': { input: 1, output: 2 },
                pricey: { input: 10, output: 20, cachedInput: 5 },
            },
        });
        const cheap = new EchoLM(
            { q: { promptTokens: 1_000_000, completionTokens: 500_000 } },
            'cheap'
        );
        const pricey = new EchoLM(
            { q2: { promptTokens: 1_000_000, completionTokens: 0, cachedTokens: 400_000 } },
            'pricey'
        );
        const pipeline = new Pipeline();
        pipeline.first = new Predict('question -> answer', cheap);
        pipeline.second = new Predict('question -> answer', pricey);

        const usage = (await pipeline.forward({ question: 'q' })).getLMUsage();

        expect(usage['test/cheap'].totalCost).toBeCloseTo(1 + 1);
        // 600k uncached at $10/M + 400k cached at $5/M.
        expect(usage['test/pricey'].totalCost).toBeCloseTo(6 + 2);
        expect(usage['test/pricey'].cachedTokens).toBe(400_000);
    });

    it('gives each traced LM call its exact usage under concurrency', async () => {
        configure({ tracing: true });
        const lm = new EchoLM({
            slow: { delayMs: 25, promptTokens: 40 },
            fast: { delayMs: 1, promptTokens: 4 },
        });
        const predict = new Predict('question -> answer', lm);

        const [slow, fast] = await Promise.all([
            predict.forward({ question: 'slow' }),
            predict.forward({ question: 'fast' }),
        ]);

        expect(slow.trace?.calls[0].usage.promptTokens).toBe(40);
        expect(fast.trace?.calls[0].usage.promptTokens).toBe(4);
        expect(slow.trace?.usage.promptTokens).toBe(40);
    });

    it('counts response-cache hits without counting requests', async () => {
        configure({ trackUsage: true, cache: true });
        try {
            const lm = new MockLM({ responses: ['answer: A'] });
            const predict = new Predict('question -> answer', lm);

            await predict.forward({ question: 'Q' });
            const second = await predict.forward({ question: 'Q' });

            expect(second.getLMUsage()['mock/mock-model']).toMatchObject({
                requestCount: 0,
                cacheHits: 1,
                totalTokens: 0,
            });
        } finally {
            configure({ cache: false });
        }
    });

    it('lets a hand-written model report usage with recordLMUsage', async () => {
        const { usage } = await collectUsage(async () => {
            recordLMUsage('custom/model', { promptTokens: 3, completionTokens: 4 });
        });

        expect(usage.total()).toMatchObject({
            promptTokens: 3,
            totalTokens: 7,
            requestCount: 1,
        });
        // Outside any collecting scope it is a no-op.
        expect(() => recordLMUsage('custom/model', { promptTokens: 1 })).not.toThrow();
    });

    it('sums usage across models', () => {
        expect(
            sumUsage([
                { promptTokens: 1, completionTokens: 2, totalTokens: 3, totalCost: 0.5 },
                { promptTokens: 4, completionTokens: 5, totalTokens: 9 },
            ])
        ).toMatchObject({
            promptTokens: 5,
            completionTokens: 7,
            totalTokens: 12,
            totalCost: 0.5,
        });
    });
});

describe('evaluate usage', () => {
    it('counts only the run’s own calls, even with concurrent traffic on the same model', async () => {
        const lm = new EchoLM({
            a: { promptTokens: 2, delayMs: 5 },
            b: { promptTokens: 3, delayMs: 5 },
            noise: { promptTokens: 1000, delayMs: 2 },
        });
        const predict = new Predict('question -> answer', lm);
        const data = ['a', 'b'].map((q) =>
            new Example({ question: q, answer: q }).withInputs('question')
        );

        const [report] = await Promise.all([
            evaluate(predict, data, (ex, pred) => ex.get('answer') === pred.get('answer')),
            predict.forward({ question: 'noise' }),
        ]);

        expect(report.score).toBe(1);
        expect(report.usage.promptTokens).toBe(5);
        expect(report.usage.requestCount).toBe(2);
        expect(report.usage.byModel['test/echo'].promptTokens).toBe(5);
    });
});
