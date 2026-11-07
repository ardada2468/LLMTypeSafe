import { Predict } from './predict';
import { ChainOfThought } from './chain-of-thought';
import { Signature, InputField, OutputField } from '../core/signature';
import { Example } from '../core/example';
import { type SignatureLike } from '../core/signature';
import { ChatAdapter } from '../adapters/chat';
import { JSONAdapter } from '../adapters/json';
import { contentToText } from '../utils/content';
import type { ChatMessage } from '../types/language-model';
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

// Rewritten for 0.7: demos are formatted by the adapter as user/assistant
// message pairs, not as a numbered "worked examples" block inside one prompt.

/** The messages ChatAdapter builds for one call with `demos`. */
function format(
    signature: SignatureLike,
    demos: Example[],
    inputs = { question: 'Capital of Peru?' }
) {
    return new ChatAdapter().format({ signature, demos, inputs });
}
const text = (messages: ChatMessage[]) => messages.map((m) => contentToText(m.content));

describe('demos in the prompt', () => {
    it('renders nothing extra when a module has no demos', () => {
        expect(format(QA, []).map((m) => m.role)).toEqual(['system', 'user']);
    });

    it('renders each demo as a user turn and an assistant turn', () => {
        const messages = format(QA, [paris]);

        expect(messages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user']);
        expect(text(messages)[1]).toBe('[[ ## question ## ]]\nCapital of France?');
        expect(text(messages)[2]).toBe(
            '[[ ## answer ## ]]\nParis\n\n[[ ## confidence ## ]]\n0.99\n\n[[ ## completed ## ]]'
        );
    });

    it('keeps demos in the order given, before the real input', () => {
        const all = text(format(QA, [paris, tokyo])).join('\n');

        expect(all.indexOf('Paris')).toBeLessThan(all.indexOf('Tokyo'));
        expect(all.indexOf('Tokyo')).toBeLessThan(all.indexOf('Capital of Peru?'));
    });

    it('renders demos for a string signature', () => {
        const all = text(
            format('question -> answer', [
                new Example({ question: 'Capital of France?', answer: 'Paris' }),
            ])
        ).join('\n');

        expect(all).toContain('[[ ## question ## ]]\nCapital of France?');
        expect(all).toContain('[[ ## answer ## ]]\nParis');
    });

    it('splits a demo by the signature when the example declares no input keys', () => {
        const messages = format(QA, [
            new Example({ question: 'Capital of Italy?', answer: 'Rome', confidence: 0.9 }),
        ]);

        expect(text(messages)[1]).toContain('Capital of Italy?');
        expect(text(messages)[2]).toContain('[[ ## answer ## ]]\nRome');
    });

    it('serialises non-string demo values as JSON', () => {
        class Tagged extends Signature {
            @InputField({ description: 'the text' })
            text!: string;

            @OutputField({ description: 'tags', type: 'string[]' })
            tags!: string[];
        }

        const messages = format(
            Tagged,
            [new Example({ text: 'hello', tags: ['a', 'b'] }).withInputs('text')],
            { text: 'x' } as any
        );

        expect(text(messages)[2]).toContain('[[ ## tags ## ]]\n["a","b"]');
    });

    it('drops a demo that shares no fields with the signature', () => {
        const messages = format(QA, [new Example({ unrelated: 'nothing to teach' })]);

        expect(messages).toHaveLength(2);
    });

    it('ignores fields the signature does not declare', () => {
        const all = text(
            format(QA, [
                new Example({ question: 'Capital of France?', answer: 'Paris', id: 'row-7' }),
            ])
        ).join('\n');

        expect(all).not.toContain('row-7');
    });

    it('renders demo replies as JSON for the JSON adapter', () => {
        const messages = new JSONAdapter().format({
            signature: QA,
            demos: [paris],
            inputs: { question: 'Capital of Peru?' },
        });

        expect(messages[2]).toEqual({
            role: 'assistant',
            content: '{"answer":"Paris","confidence":0.99}',
        });
    });
});

describe('Predict demo configuration', () => {
    it('sends configured demos to the model', async () => {
        const lm = new MockLM({ responses: ['answer: Lima\nconfidence: 0.9'] });
        const predict = new Predict(QA, { lm, demos: [paris] });

        await predict.forward({ question: 'Capital of Peru?' });

        expect(lm.lastPrompt()).toContain('[[ ## answer ## ]]\nParis');
    });

    it('keeps the two-argument (signature, lm) form working', async () => {
        const lm = new MockLM({ responses: ['answer: Lima\nconfidence: 0.9'] });

        await new Predict(QA, lm).forward({ question: 'Capital of Peru?' });

        expect(lm.calls[0].messages.map((m) => m.role)).toEqual(['system', 'user']);
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
            responses: ['reasoning: because Peru\nanswer: Lima\nconfidence: 0.9'],
        });
        const compiled = new ChainOfThought(QA, lm).withDemos([paris]);

        const result = await compiled.forward({ question: 'Capital of Peru?' });

        expect(compiled).toBeInstanceOf(ChainOfThought);
        expect(result.reasoning).toBe('because Peru');
        expect(lm.lastPrompt()).toContain('[[ ## answer ## ]]\nParis');
    });

    it('sends JSON demos to a provider with native structured output', async () => {
        const lm = new MockLM({
            structuredResponses: [{ answer: 'Lima', confidence: 0.9 }],
            capabilities: { supportsStructuredOutput: true },
        });

        await new Predict(QA, { lm, demos: [paris] }).forward({ question: 'Capital of Peru?' });

        expect(lm.structuredCalls[0].prompt).toContain('Assistant:\n{"answer":"Paris"');
        expect(lm.structuredCalls[0].prompt).not.toContain('[[ ## answer ## ]]\nParis');
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
