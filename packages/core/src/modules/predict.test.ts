import { Predict } from './predict';
import { Signature, InputField, OutputField } from '../core/signature';
import { ValidationError } from '../core/errors';
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

describe('Predict', () => {
    describe('text path (provider without structured output)', () => {
        it('parses and validates labelled text output', async () => {
            const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.95'] });
            const predict = new Predict(QA, lm);

            const result = await predict.forward({ question: 'Capital of France?' });

            expect(result.answer).toBe('Paris');
            expect(result.confidence).toBe(0.95);
            expect(typeof result.confidence).toBe('number');
        });

        it('includes the signature description and inputs in the prompt', async () => {
            const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.9'] });
            await new Predict(QA, lm).forward({ question: 'Capital of France?' });

            expect(lm.lastPrompt()).toContain('Answer a question');
            expect(lm.lastPrompt()).toContain('question: Capital of France?');
        });

        it('throws ValidationError when a typed field cannot be coerced', async () => {
            const lm = new MockLM({ responses: ['answer: Paris\nconfidence: very high'] });
            const predict = new Predict(QA, lm);

            await expect(predict.forward({ question: 'Q' })).rejects.toThrow(ValidationError);
        });

        it('throws ValidationError when a required field is absent', async () => {
            const lm = new MockLM({ responses: ['answer: Paris'] });
            const predict = new Predict(QA, lm);

            await expect(predict.forward({ question: 'Q' })).rejects.toThrow(ValidationError);
        });

        it('passes call options through to the model', async () => {
            const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.9'] });
            await new Predict(QA, lm).forward(
                { question: 'Q' },
                { temperature: 0.2, maxTokens: 128 }
            );

            expect(lm.calls[0].options).toEqual({ temperature: 0.2, maxTokens: 128 });
        });

        it('works with a string signature', async () => {
            const lm = new MockLM({ responses: ['answer: 42'] });
            const result = await new Predict('question -> answer: int', lm).forward({
                question: 'Q',
            });

            expect(result.answer).toBe(42);
        });
    });

    describe('structured path (provider with native structured output)', () => {
        const structuredLM = (responses: unknown[]) =>
            new MockLM({
                structuredResponses: responses,
                capabilities: { supportsStructuredOutput: true },
            });

        it('uses generateStructured and validates the result', async () => {
            const lm = structuredLM([{ answer: 'Paris', confidence: 0.99 }]);
            const result = await new Predict(QA, lm).forward({ question: 'Q' });

            expect(lm.structuredCalls).toHaveLength(1);
            expect(lm.calls).toHaveLength(0);
            expect(result.answer).toBe('Paris');
            expect(result.confidence).toBe(0.99);
        });

        it('sends a JSON Schema describing every output field', async () => {
            const lm = structuredLM([{ answer: 'Paris', confidence: 0.9 }]);
            await new Predict(QA, lm).forward({ question: 'Q' });

            expect(lm.structuredCalls[0].schema).toMatchObject({
                type: 'object',
                required: ['answer', 'confidence'],
                additionalProperties: false,
                properties: {
                    answer: { type: 'string' },
                    confidence: { type: 'number' },
                },
            });
        });

        it('still validates when the provider returns the wrong type', async () => {
            const lm = structuredLM([{ answer: 'Paris', confidence: 'high' }]);

            await expect(new Predict(QA, lm).forward({ question: 'Q' })).rejects.toThrow(
                ValidationError
            );
        });

        it('treats null as absent so optional fields can be omitted', async () => {
            class Opt extends Signature {
                @OutputField({ description: 'answer' })
                answer!: string;

                @OutputField({ description: 'note', required: false })
                note?: string;
            }

            const lm = new MockLM({
                structuredResponses: [{ answer: 'done', note: null }],
                capabilities: { supportsStructuredOutput: true },
            });

            const result = await new Predict(Opt, lm).forward({});
            expect(result.answer).toBe('done');
            expect(result.note).toBeUndefined();
        });

        it('passes options through on the structured path', async () => {
            const lm = structuredLM([{ answer: 'Paris', confidence: 0.9 }]);
            await new Predict(QA, lm).forward({ question: 'Q' }, { timeout: 1234 });

            expect(lm.structuredCalls[0].options).toEqual({ timeout: 1234 });
        });
    });

    it('throws a clear error when constructed without a signature', async () => {
        const lm = new MockLM({ responses: ['x'] });
        const predict = new Predict(undefined as any, lm);

        await expect(predict.forward({})).rejects.toThrow('No signature provided');
    });
});
