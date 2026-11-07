import { z } from 'zod';
import { ChainOfThought, REASONING_DESCRIPTION } from './chain-of-thought';
import { Predict } from './predict';
import { Signature, InputField, OutputField, ImageField, signature } from '../core/signature';
import { Example } from '../core/example';
import { ValidationError } from '../core/errors';
import { MockLM } from '../test-utils';

// Rewritten for 0.7: ChainOfThought is a single call whose signature gains a
// leading `reasoning` output, as in DSPy. It used to make a free-text
// reasoning call and then an answer call.

class QA extends Signature {
    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;
}

const reply = (reasoning: string, answer: string) =>
    `[[ ## reasoning ## ]]\n${reasoning}\n\n[[ ## answer ## ]]\n${answer}\n\n[[ ## completed ## ]]`;

describe('ChainOfThought', () => {
    it('makes exactly one call and returns the reasoning with the outputs', async () => {
        const lm = new MockLM({
            responses: [reply('Paris is the capital of France.', 'Paris')],
        });

        const result = await new ChainOfThought(QA, lm).forward({
            question: 'Capital of France?',
        });

        expect(lm.calls).toHaveLength(1);
        expect(result.reasoning).toBe('Paris is the capital of France.');
        expect(result.answer).toBe('Paris');
    });

    it('asks for a leading reasoning field in the prompt', async () => {
        const lm = new MockLM({ responses: [reply('r', 'Paris')] });
        await new ChainOfThought(QA, lm).forward({ question: 'Q' });

        const system = String(lm.calls[0].messages[0].content);
        expect(system).toContain(`1. \`reasoning\` (string): ${REASONING_DESCRIPTION}`);
        expect(system.indexOf('[[ ## reasoning ## ]]')).toBeLessThan(
            system.indexOf('[[ ## answer ## ]]')
        );
    });

    it('works with a string signature', async () => {
        const lm = new MockLM({ responses: ['reasoning: 6 times 7\nanswer: 42'] });

        const result = await new ChainOfThought('question -> answer: int', lm).forward({
            question: 'What is 6 times 7?',
        });

        expect(result.answer).toBe(42);
        expect(result.reasoning).toBe('6 times 7');
    });

    it('works with a zod signature and keeps its types', async () => {
        const Classify = signature({
            input: z.object({ text: z.string() }),
            output: z.object({ label: z.enum(['spam', 'ham']) }),
        });
        const lm = new MockLM({ responses: ['reasoning: looks fine\nlabel: ham'] });

        const result = await new ChainOfThought(Classify, lm).forward({ text: 'hi mum' });

        const label: 'spam' | 'ham' = result.label;
        const reasoning: string = result.reasoning;
        expect(label).toBe('ham');
        expect(reasoning).toBe('looks fine');
    });

    it('sends image inputs as content parts', async () => {
        class Caption extends Signature {
            @ImageField({ description: 'the photo' })
            photo!: string;

            @OutputField({ description: 'a caption' })
            caption!: string;
        }
        const lm = new MockLM({ responses: ['reasoning: a cat is visible\ncaption: a cat'] });

        const result = await new ChainOfThought(Caption, lm).forward({
            photo: 'https://example.com/cat.png',
        });

        expect(result.caption).toBe('a cat');
        const content = lm.calls[0].messages.at(-1)?.content;
        expect(Array.isArray(content)).toBe(true);
        expect(content).toContainEqual({
            type: 'image',
            source: { kind: 'url', url: 'https://example.com/cat.png' },
        });
    });

    it('renders demos with or without reasoning', async () => {
        const lm = new MockLM({ responses: [reply('r', 'Lima')] });
        const cot = new ChainOfThought(QA, {
            lm,
            demos: [
                new Example({
                    question: 'Capital of France?',
                    reasoning: 'France is in Europe.',
                    answer: 'Paris',
                }).withInputs('question'),
                new Example({ question: 'Capital of Japan?', answer: 'Tokyo' }).withInputs(
                    'question'
                ),
            ],
        });

        await cot.forward({ question: 'Capital of Peru?' });

        const assistant = lm.calls[0].messages
            .filter((message) => message.role === 'assistant')
            .map((message) => String(message.content));
        expect(assistant[0]).toContain('[[ ## reasoning ## ]]\nFrance is in Europe.');
        expect(assistant[1]).not.toContain('reasoning');
        expect(assistant[1]).toContain('[[ ## answer ## ]]\nTokyo');
    });

    it('passes options through to the call', async () => {
        const lm = new MockLM({ responses: [reply('r', 'Paris')] });
        await new ChainOfThought(QA, lm).forward({ question: 'Q' }, { temperature: 0.3 });

        expect(lm.calls[0].options).toEqual({ temperature: 0.3 });
    });

    it('validates the outputs against the signature', async () => {
        class Scored extends Signature {
            @OutputField({ description: 'score', type: 'number' })
            score!: number;
        }

        const lm = new MockLM({ responses: ['reasoning: r\nscore: not-a-number'] });

        await expect(new ChainOfThought(Scored, lm).forward({})).rejects.toThrow(
            ValidationError
        );
    });

    it('uses the structured path when the provider supports it', async () => {
        const lm = new MockLM({
            structuredResponses: [{ reasoning: 'some reasoning', answer: 'Paris' }],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await new ChainOfThought(QA, lm).forward({ question: 'Q' });

        expect(lm.calls).toHaveLength(0);
        expect(lm.structuredCalls).toHaveLength(1);
        expect(lm.structuredCalls[0].schema).toMatchObject({
            required: ['reasoning', 'answer'],
        });
        expect(result.answer).toBe('Paris');
        expect(result.reasoning).toBe('some reasoning');
    });

    it('is a predictor, reported as self', () => {
        const cot = new ChainOfThought(QA, new MockLM());

        expect(cot).toBeInstanceOf(Predict);
        expect(cot.namedPredictors()).toEqual([['self', cot]]);
    });

    it('repairs a failed reply in a second call', async () => {
        class Scored extends Signature {
            @OutputField({ description: 'score', type: 'number' })
            score!: number;
        }

        const lm = new MockLM({
            responses: ['reasoning: r\nscore: not-a-number', 'reasoning: r\nscore: 0.4'],
        });

        const result = await new ChainOfThought(Scored, lm).forward({}, { repairAttempts: 1 });

        expect(lm.calls).toHaveLength(2);
        expect(result.score).toBe(0.4);
        expect(lm.lastPrompt()).toContain('score: expected number');
    });
});
