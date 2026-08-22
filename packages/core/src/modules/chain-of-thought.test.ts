import { ChainOfThought } from './chain-of-thought';
import { Signature, InputField, OutputField } from '../core/signature';
import { ValidationError } from '../core/errors';
import { MockLM } from '../test-utils';

class QA extends Signature {
    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;
}

describe('ChainOfThought', () => {
    it('reasons first, then answers, and returns both', async () => {
        const lm = new MockLM({
            responses: ['Paris is the capital of France.', 'answer: Paris'],
        });

        const result = await new ChainOfThought(QA, lm).forward({
            question: 'Capital of France?',
        });

        expect(lm.calls).toHaveLength(2);
        expect(result.reasoning).toBe('Paris is the capital of France.');
        expect(result.answer).toBe('Paris');
    });

    it('asks for step-by-step reasoning in the first call', async () => {
        const lm = new MockLM({ responses: ['reasoning', 'answer: Paris'] });
        await new ChainOfThought(QA, lm).forward({ question: 'Q' });

        expect(lm.calls[0].messages[0].content).toContain("Let's think step by step");
    });

    it('feeds the reasoning into the final prompt', async () => {
        const lm = new MockLM({ responses: ['because of X', 'answer: Paris'] });
        await new ChainOfThought(QA, lm).forward({ question: 'Q' });

        expect(lm.calls[1].messages[0].content).toContain('Reasoning: because of X');
    });

    it('passes options through to both calls', async () => {
        const lm = new MockLM({ responses: ['reasoning', 'answer: Paris'] });
        await new ChainOfThought(QA, lm).forward({ question: 'Q' }, { temperature: 0.3 });

        expect(lm.calls[0].options).toEqual({ temperature: 0.3 });
        expect(lm.calls[1].options).toEqual({ temperature: 0.3 });
    });

    it('validates the final answer against the signature', async () => {
        class Scored extends Signature {
            @OutputField({ description: 'score', type: 'number' })
            score!: number;
        }

        const lm = new MockLM({ responses: ['reasoning', 'score: not-a-number'] });

        await expect(new ChainOfThought(Scored, lm).forward({})).rejects.toThrow(
            ValidationError
        );
    });

    it('uses the structured path when the provider supports it', async () => {
        const lm = new MockLM({
            responses: ['some reasoning'],
            structuredResponses: [{ answer: 'Paris' }],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await new ChainOfThought(QA, lm).forward({ question: 'Q' });

        // One free-text call for reasoning, one structured call for the answer.
        expect(lm.calls).toHaveLength(1);
        expect(lm.structuredCalls).toHaveLength(1);
        expect(result.answer).toBe('Paris');
        expect(result.reasoning).toBe('some reasoning');
    });

    describe('self-repair', () => {
        it('retries the answering step without reasoning again', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'score', type: 'number' })
                score!: number;
            }

            const lm = new MockLM({
                responses: ['reasoning', 'score: not-a-number', 'score: 0.4'],
            });

            const result = await new ChainOfThought(Scored, lm).forward(
                {},
                { repairAttempts: 1 }
            );

            expect(lm.calls).toHaveLength(3);
            expect(result.score).toBe(0.4);
            expect(result.reasoning).toBe('reasoning');
        });

        it('carries the reasoning into the repair prompt', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'score', type: 'number' })
                score!: number;
            }

            const lm = new MockLM({
                responses: ['because of X', 'score: not-a-number', 'score: 0.4'],
            });

            await new ChainOfThought(Scored, lm).forward({}, { repairAttempts: 1 });

            expect(lm.lastPrompt()).toContain('Reasoning: because of X');
            expect(lm.lastPrompt()).toContain('score: expected number');
        });

        it('rethrows the ValidationError when repair is not requested', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'score', type: 'number' })
                score!: number;
            }

            const lm = new MockLM({ responses: ['reasoning', 'score: not-a-number'] });

            await expect(new ChainOfThought(Scored, lm).forward({})).rejects.toThrow(
                ValidationError
            );
            expect(lm.calls).toHaveLength(2);
        });
    });
});
