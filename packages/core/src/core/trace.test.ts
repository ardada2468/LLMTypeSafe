import { clearHistory, inspectHistory } from './trace';
import { configure } from './config';
import { ChainOfThought } from '../modules/chain-of-thought';
import { Predict } from '../modules/predict';
import { RespAct } from '../modules/respact';
import { MockLM } from '../test-utils';
import { ValidationError } from './errors';
import { type TraceEntry } from '../types/module';

describe('tracing', () => {
    beforeEach(() => {
        clearHistory();
        configure({ tracing: false, onTrace: undefined, traceHistorySize: 100 });
    });

    afterEach(() => {
        clearHistory();
        configure({ tracing: false, onTrace: undefined, traceHistorySize: 100 });
    });

    it('records nothing while tracing is off', async () => {
        const lm = new MockLM({ responses: ['answer: Paris'] });

        const result = await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(result.trace).toBeUndefined();
        expect(inspectHistory()).toEqual([]);
    });

    it('attaches a trace to the prediction when tracing is on', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: Paris'] });

        const result = await new Predict('question -> answer', lm).forward({
            question: 'Capital of France?',
        });

        expect(result.trace?.moduleId).toMatch(/^Predict#\d+$/);
        expect(result.trace?.input).toEqual({ question: 'Capital of France?' });
        expect(result.trace?.output).toEqual({ answer: 'Paris' });
        expect(result.trace?.rawLMOutput).toBe('answer: Paris');
    });

    it('records the prompt actually sent to the model', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: Paris'] });

        await new Predict('question -> answer', lm).forward({ question: 'Capital of France?' });

        expect(inspectHistory(1)[0].rawLMInput).toBe(lm.lastPrompt());
    });

    it('measures the usage delta for the call rather than lifetime totals', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: one', 'answer: two'] });
        const predict = new Predict('question -> answer', lm);

        await predict.forward({ question: 'first' });
        await predict.forward({ question: 'second' });

        const [first, second] = inspectHistory(2);
        expect(first.usage.totalTokens).toBe(15);
        expect(second.usage.totalTokens).toBe(15);
        expect(lm.getUsage().totalTokens).toBe(30);
    });

    it('records one entry per language-model call in a multi-step module', async () => {
        configure({ tracing: true });
        const lm = new MockLM({
            responses: ['Paris is the capital.', 'answer: Paris'],
        });

        const result = await new ChainOfThought('question -> answer', lm).forward({
            question: 'Capital of France?',
        });

        expect(result.trace?.calls).toHaveLength(2);
        expect(result.trace?.calls[0].rawOutput).toBe('Paris is the capital.');
        expect(result.trace?.rawLMOutput).toBe('answer: Paris');
    });

    it('traces every step of a tool loop', async () => {
        configure({ tracing: true });
        const lm = new MockLM({
            responses: [
                'I need the weather.\nAction: weather\nAction Input: Paris',
                'Final Answer: answer: sunny',
            ],
        });

        await new RespAct('question -> answer', {
            lm,
            tools: { weather: async () => 'sunny' },
        }).forward({ question: 'Weather in Paris?' });

        const [entry] = inspectHistory(1);
        expect(entry.moduleId).toMatch(/^RespAct#\d+$/);
        expect(entry.calls).toHaveLength(2);
        expect(entry.output).toEqual({ answer: 'sunny', steps: 2 });
    });

    it('records the prompt behind a failed call and rethrows', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: Paris'] });

        await expect(
            new Predict('question -> answer, confidence', lm).forward({ question: 'Q' })
        ).rejects.toThrow(ValidationError);

        const [entry] = inspectHistory(1);
        expect(entry.error).toBeInstanceOf(ValidationError);
        expect(entry.rawLMOutput).toBe('answer: Paris');
        expect(entry.output).toEqual({});
    });

    it('records the prompt even when the provider itself throws', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: [] });

        await expect(
            new Predict('question -> answer', lm).forward({ question: 'Q' })
        ).rejects.toThrow('no more scripted responses');

        const [entry] = inspectHistory(1);
        expect(entry.calls).toHaveLength(1);
        expect(entry.rawLMInput).toContain('question: Q');
        expect(entry.rawLMOutput).toBe('');
    });

    it('sums per-call usage into the entry total', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['reasoning', 'answer: Paris'] });

        const result = await new ChainOfThought('question -> answer', lm).forward({
            question: 'Q',
        });

        const calls = result.trace?.calls ?? [];
        expect(calls.map((call) => call.usage.totalTokens)).toEqual([15, 15]);
        expect(result.trace?.usage.totalTokens).toBe(30);
    });

    it('falls back to the default bound when given a non-finite history size', async () => {
        configure({ tracing: true, traceHistorySize: Number('nope') });
        const lm = new MockLM({ responses: ['answer: a', 'answer: b'] });
        const predict = new Predict('question -> answer', lm);

        await predict.forward({ question: '1' });
        await predict.forward({ question: '2' });

        expect(inspectHistory()).toHaveLength(2);
    });

    it('returns the last n entries, oldest first', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: a', 'answer: b', 'answer: c'] });
        const predict = new Predict('question -> answer', lm);

        await predict.forward({ question: '1' });
        await predict.forward({ question: '2' });
        await predict.forward({ question: '3' });

        expect(inspectHistory(2).map((entry) => entry.output.answer)).toEqual(['b', 'c']);
        expect(inspectHistory()).toHaveLength(3);
        expect(inspectHistory(0)).toEqual([]);
    });

    it('drops the oldest entries once the history is full', async () => {
        configure({ tracing: true, traceHistorySize: 2 });
        const lm = new MockLM({ responses: ['answer: a', 'answer: b', 'answer: c'] });
        const predict = new Predict('question -> answer', lm);

        await predict.forward({ question: '1' });
        await predict.forward({ question: '2' });
        await predict.forward({ question: '3' });

        expect(inspectHistory().map((entry) => entry.output.answer)).toEqual(['b', 'c']);
    });

    it('hands every entry to the onTrace handler', async () => {
        const seen: TraceEntry[] = [];
        configure({ tracing: true, onTrace: (entry) => seen.push(entry) });
        const lm = new MockLM({ responses: ['answer: Paris'] });

        await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(seen).toHaveLength(1);
        expect(seen[0].output).toEqual({ answer: 'Paris' });
    });

    it('does not call the handler while tracing is off', async () => {
        const onTrace = vi.fn();
        configure({ tracing: false, onTrace });
        const lm = new MockLM({ responses: ['answer: Paris'] });

        await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(onTrace).not.toHaveBeenCalled();
    });

    it('survives a handler that throws', async () => {
        configure({
            tracing: true,
            onTrace: () => {
                throw new Error('exporter is down');
            },
        });
        const lm = new MockLM({ responses: ['answer: Paris'] });

        const result = await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(result.answer).toBe('Paris');
        expect(inspectHistory()).toHaveLength(1);
    });

    it('gives each module instance its own id', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: a', 'answer: b'] });

        await new Predict('question -> answer', lm).forward({ question: '1' });
        await new Predict('question -> answer', lm).forward({ question: '2' });

        const [first, second] = inspectHistory(2);
        expect(first.moduleId).not.toBe(second.moduleId);
    });

    it('snapshots the top-level inputs so later reassignment cannot rewrite the trace', async () => {
        configure({ tracing: true });
        const lm = new MockLM({ responses: ['answer: Paris'] });
        const inputs = { question: 'Q' };

        await new Predict('question -> answer', lm).forward(inputs);
        inputs.question = 'mutated';

        expect(inspectHistory(1)[0].input).toEqual({ question: 'Q' });
    });
});
