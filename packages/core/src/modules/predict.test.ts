import { Predict } from './predict';
import { Signature, InputField, OutputField, ImageField } from '../core/signature';
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

    describe('self-repair', () => {
        it('re-prompts and succeeds when repairAttempts allows one more round-trip', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: very high',
                    'answer: Paris\nconfidence: 0.9',
                ],
            });

            const result = await new Predict(QA, lm).forward(
                { question: 'Q' },
                { repairAttempts: 1 }
            );

            expect(lm.calls).toHaveLength(2);
            expect(result.confidence).toBe(0.9);
        });

        it('names the failing field, its declared type and the received value', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: very high',
                    'answer: Paris\nconfidence: 0.9',
                ],
            });

            await new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 1 });

            const repairPrompt = lm.lastPrompt();
            expect(repairPrompt).toContain('failed validation for: confidence');
            expect(repairPrompt).toContain('expected number');
            expect(repairPrompt).toContain('"very high"');
        });

        it('reports a missing field as received nothing', async () => {
            const lm = new MockLM({
                responses: ['answer: Paris', 'answer: Paris\nconfidence: 0.5'],
            });

            await new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 1 });

            expect(lm.lastPrompt()).toContain('confidence: expected number, received nothing');
        });

        it('keeps the original prompt in the repair prompt', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: very high',
                    'answer: Paris\nconfidence: 0.9',
                ],
            });

            await new Predict(QA, lm).forward(
                { question: 'Capital of France?' },
                { repairAttempts: 1 }
            );

            expect(lm.lastPrompt()).toContain('question: Capital of France?');
        });

        it('rethrows the last ValidationError once the attempts are spent', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: high',
                    'answer: Paris\nconfidence: higher',
                ],
            });

            await expect(
                new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 1 })
            ).rejects.toThrow(ValidationError);
            expect(lm.calls).toHaveLength(2);
        });

        it('makes no extra call when the first response validates', async () => {
            const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.9'] });

            await new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 3 });

            expect(lm.calls).toHaveLength(1);
        });

        it('stops early when an attempt reproduces the previous failure exactly', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: high',
                    'answer: Paris\nconfidence: high',
                    'answer: Paris\nconfidence: 0.9',
                ],
            });

            await expect(
                new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 5 })
            ).rejects.toThrow(ValidationError);
            expect(lm.calls).toHaveLength(2);
        });

        it('does not stack earlier corrections onto later repair prompts', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: very high',
                    'answer: Paris\nconfidence: quite high',
                    'answer: Paris\nconfidence: 0.7',
                ],
            });

            await new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 2 });

            const occurrences = lm.lastPrompt().split('failed validation for:').length - 1;
            expect(lm.calls).toHaveLength(3);
            expect(occurrences).toBe(1);
        });

        it('repairs on the native structured path too', async () => {
            const lm = new MockLM({
                structuredResponses: [
                    { answer: 'Paris', confidence: 'high' },
                    { answer: 'Paris', confidence: 0.9 },
                ],
                capabilities: { supportsStructuredOutput: true },
            });

            const result = await new Predict(QA, lm).forward(
                { question: 'Q' },
                { repairAttempts: 1 }
            );

            expect(lm.structuredCalls).toHaveLength(2);
            expect(result.confidence).toBe(0.9);
            expect(lm.structuredCalls[1].prompt).toContain('failed validation for: confidence');
            expect(lm.structuredCalls[1].prompt).toContain('Return the corrected object');
        });

        it('treats a repairAttempts of zero as no repair at all', async () => {
            const lm = new MockLM({
                responses: [
                    'answer: Paris\nconfidence: very high',
                    'answer: Paris\nconfidence: 0.9',
                ],
            });

            await expect(
                new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 0 })
            ).rejects.toThrow(ValidationError);
            expect(lm.calls).toHaveLength(1);
        });

        it('does not repair errors that are not validation failures', async () => {
            const lm = new MockLM({ responses: [] });

            await expect(
                new Predict(QA, lm).forward({ question: 'Q' }, { repairAttempts: 2 })
            ).rejects.toThrow('no more scripted responses');
            expect(lm.calls).toHaveLength(1);
        });
    });

    it('throws a clear error when constructed without a signature', async () => {
        const lm = new MockLM({ responses: ['x'] });
        const predict = new Predict(undefined as any, lm);

        await expect(predict.forward({})).rejects.toThrow('No signature provided');
    });
});

describe('image inputs', () => {
    const PNG = 'iVBORw0KGgo=';

    class DescribeImage extends Signature {
        static description = 'Describe the picture.';

        @ImageField({ description: 'the picture' })
        picture!: string;

        @OutputField({ description: 'what it shows' })
        caption!: string;
    }

    it('sends the image as content rather than a placeholder', async () => {
        const lm = new MockLM({ responses: ['caption: a cat'] });

        const result = await new Predict(DescribeImage, lm).forward({
            picture: `data:image/png;base64,${PNG}`,
        });

        expect(result.caption).toBe('a cat');

        // Predict used to flatten the image to "[image: image/png]", so the
        // model never actually saw it.
        const content = lm.calls.at(-1)?.messages[0]?.content;
        expect(Array.isArray(content)).toBe(true);
        expect(content).toContainEqual({
            type: 'image',
            source: { kind: 'base64', mediaType: 'image/png', data: PNG },
        });
    });

    it('leaves a text-only prompt as a plain string', async () => {
        const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.9'] });

        await new Predict(QA, lm).forward({ question: 'Capital of France?' });

        expect(typeof lm.calls.at(-1)?.messages[0]?.content).toBe('string');
    });
});
