import { Predict } from './predict';
import { ChainOfThought } from './chain-of-thought';
import { Signature, InputField, OutputField } from '../core/signature';
import { Example } from '../core/example';
import { buildPrompt, renderDemos } from '../utils/parsing';
import { MockLM } from '../test-utils';

class QA extends Signature {
    static description = 'Answer a question';

    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;

    @OutputField({ description: 'confidence 0-1', type: 'number' })
    confidence!: number;
}

const paris = new Example({
    question: 'Capital of France?',
    answer: 'Paris',
    confidence: 0.99,
}).withInputs('question');

const tokyo = new Example({
    question: 'Capital of Japan?',
    answer: 'Tokyo',
    confidence: 0.98,
}).withInputs('question');

describe('demos in the prompt', () => {
    it('renders nothing when a module has no demos', () => {
        const withoutDemos = buildPrompt(QA, { question: 'Capital of Peru?' });
        const withEmptyDemos = buildPrompt(QA, { question: 'Capital of Peru?' }, []);

        expect(withEmptyDemos).toBe(withoutDemos);
        expect(withoutDemos).not.toContain('worked example');
    });

    it('renders demos in the labelled format the parser reads back', () => {
        const block = renderDemos(QA, [paris]);

        expect(block).toContain('question: Capital of France?');
        expect(block).toContain('answer: Paris');
        expect(block).toContain('confidence: 0.99');
    });

    it('numbers demos and keeps them in the order given', () => {
        const block = renderDemos(QA, [paris, tokyo]);

        expect(block).toContain('Here are 2 worked examples of this task:');
        expect(block.indexOf('Example 1:')).toBeLessThan(block.indexOf('Example 2:'));
        expect(block.indexOf('Paris')).toBeLessThan(block.indexOf('Tokyo'));
    });

    it('renders demos for a string signature', () => {
        const prompt = buildPrompt('question -> answer', { question: 'Capital of Peru?' }, [
            new Example({ question: 'Capital of France?', answer: 'Paris' }),
        ]);

        expect(prompt).toContain('question: Capital of France?');
        expect(prompt).toContain('answer: Paris');
        expect(prompt).toContain('question: Capital of Peru?');
    });

    it('splits a demo by the signature when the example declares no input keys', () => {
        const block = renderDemos(QA, [
            new Example({ question: 'Capital of Italy?', answer: 'Rome', confidence: 0.9 }),
        ]);

        expect(block).toContain('question: Capital of Italy?');
        expect(block).toContain('answer: Rome');
    });

    it('places demos after the task description and before the real input', () => {
        const prompt = buildPrompt(QA, { question: 'Capital of Peru?' }, [paris]);

        expect(prompt.indexOf('Answer a question')).toBeLessThan(prompt.indexOf('Example 1:'));
        expect(prompt.indexOf('Example 1:')).toBeLessThan(
            prompt.indexOf('question: Capital of Peru?')
        );
    });

    it('honours a custom input prefix but labels outputs plainly', () => {
        class Prefixed extends Signature {
            @InputField({ description: 'the text', prefix: 'Text:' })
            text!: string;

            @OutputField({ description: 'the label' })
            label!: string;
        }

        const block = renderDemos(Prefixed, [
            new Example({ text: 'hello', label: 'greeting' }).withInputs('text'),
        ]);

        expect(block).toContain('Text: hello');
        expect(block).toContain('label: greeting');
    });

    it('serialises non-string demo values as JSON', () => {
        class Tagged extends Signature {
            @InputField({ description: 'the text' })
            text!: string;

            @OutputField({ description: 'tags', type: 'string[]' })
            tags!: string[];
        }

        const block = renderDemos(Tagged, [
            new Example({ text: 'hello', tags: ['a', 'b'] }).withInputs('text'),
        ]);

        expect(block).toContain('tags: ["a","b"]');
    });

    it('drops a demo that shares no fields with the signature', () => {
        const block = renderDemos(QA, [new Example({ unrelated: 'nothing to teach' })]);

        expect(block).toBe('');
    });

    it('keeps output fields the signature never declared', () => {
        const block = renderDemos(QA, [
            new Example({
                question: 'Capital of France?',
                answer: 'Paris',
                confidence: 0.99,
                reasoning: 'France is in Europe and Paris is its seat of government.',
            }).withInputs('question'),
        ]);

        expect(block).toContain('reasoning: France is in Europe');
        // Declared fields still lead, in signature order.
        expect(block.indexOf('answer: Paris')).toBeLessThan(block.indexOf('reasoning:'));
    });

    it('ignores undeclared fields when the example declares no input keys', () => {
        const block = renderDemos(QA, [
            new Example({ question: 'Capital of France?', answer: 'Paris', id: 'row-7' }),
        ]);

        expect(block).not.toContain('id:');
    });

    it('renders demos as JSON for the structured-output path', () => {
        const block = renderDemos(QA, [paris], { format: 'json' });

        expect(block).toContain('input: {"question":"Capital of France?"}');
        expect(block).toContain('output: {"answer":"Paris","confidence":0.99}');
        // The JSON schema, not the demo, dictates the shape on this path.
        expect(block).not.toContain('in the same format');
    });
});

describe('Predict demo configuration', () => {
    it('sends configured demos to the model', async () => {
        const lm = new MockLM({ responses: ['answer: Lima\nconfidence: 0.9'] });
        const predict = new Predict(QA, { lm, demos: [paris] });

        await predict.forward({ question: 'Capital of Peru?' });

        expect(lm.lastPrompt()).toContain('answer: Paris');
    });

    it('keeps the two-argument (signature, lm) form working', async () => {
        const lm = new MockLM({ responses: ['answer: Lima\nconfidence: 0.9'] });

        await new Predict(QA, lm).forward({ question: 'Capital of Peru?' });

        expect(lm.lastPrompt()).not.toContain('worked example');
    });

    it('exposes its demos as a copy', () => {
        const predict = new Predict(QA, { lm: new MockLM(), demos: [paris] });

        const demos = predict.getDemos();
        demos.push(tokyo);

        expect(predict.getDemos()).toHaveLength(1);
    });

    it('withDemos returns a new module and leaves the original unchanged', () => {
        const original = new Predict(QA, new MockLM());

        const compiled = original.withDemos([paris, tokyo]);

        expect(original.getDemos()).toHaveLength(0);
        expect(compiled.getDemos()).toHaveLength(2);
        expect(compiled).not.toBe(original);
    });

    it('withDemos preserves the module subclass', async () => {
        const lm = new MockLM({
            responses: ['because Peru', 'answer: Lima\nconfidence: 0.9'],
        });
        const compiled = new ChainOfThought(QA, lm).withDemos([paris]);

        const result = await compiled.forward({ question: 'Capital of Peru?' });

        expect(compiled).toBeInstanceOf(ChainOfThought);
        expect(result.reasoning).toBe('because Peru');
        expect(lm.calls[0].messages[0].content).toContain('answer: Paris');
    });

    it('sends JSON demos to a provider with native structured output', async () => {
        const lm = new MockLM({
            structuredResponses: [{ answer: 'Lima', confidence: 0.9 }],
            capabilities: { supportsStructuredOutput: true },
        });

        await new Predict(QA, { lm, demos: [paris] }).forward({ question: 'Capital of Peru?' });

        expect(lm.structuredCalls[0].prompt).toContain('output: {"answer":"Paris"');
        expect(lm.structuredCalls[0].prompt).not.toContain('answer: Paris');
    });

    it('rejects an object carrying only half a language model', () => {
        const halfALM = { chat: async () => 'answer: Paris' };

        expect(() => new Predict(QA, halfALM as any)).toThrow(/generate\(\)/);
    });

    it('gives each copy its own demo array', () => {
        const original = new Predict(QA, { lm: new MockLM(), demos: [paris] });

        const copy = original.withLM(new MockLM());
        (copy as any).demos.push(tokyo);

        expect((copy as any).demos).not.toBe((original as any).demos);
        expect(original.getDemos()).toHaveLength(1);
    });

    it('withLM swaps the model without touching the original', async () => {
        const student = new MockLM({ responses: ['answer: Lima\nconfidence: 0.9'] });
        const teacher = new MockLM({ responses: ['answer: Lima\nconfidence: 1.0'] });
        const predict = new Predict(QA, student);

        await predict.withLM(teacher).forward({ question: 'Capital of Peru?' });

        expect(teacher.calls).toHaveLength(1);
        expect(student.calls).toHaveLength(0);
    });
});
