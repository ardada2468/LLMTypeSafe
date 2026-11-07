import { Refine, parseAdvice } from './refine';
import { Predict } from './predict';
import { Module } from '../core/module';
import { context } from '../core/config';
import { MockLM } from '../test-utils';
import { contentToText } from '../utils/content';
import type { ChatMessage, LLMCallOptions } from '../types/language-model';

const answer = (value: string) => `[[ ## answer ## ]]\n${value}\n\n[[ ## completed ## ]]`;
const feedback = (advice: string) =>
    `[[ ## discussion ## ]]\nThe answer was too long.\n\n[[ ## advice ## ]]\n${advice}\n\n[[ ## completed ## ]]`;

const text = (messages: ChatMessage[]) =>
    messages.map((message) => contentToText(message.content)).join('\n');
const isFeedback = (messages: ChatMessage[]) => text(messages).includes('program_trajectory');

const oneWord = (_inputs: Record<string, any>, prediction: any) =>
    String(prediction.answer).trim().split(/\s+/).length === 1 ? 1 : 0;

/**
 * Plays both the program's model and the feedback model: answers program
 * calls from `answers` in order and feedback calls with `advice`.
 */
function refineLM(answers: string[], advice = '{"self": "Reply with a single word."}') {
    let index = 0;
    return new MockLM({
        respond: (messages) =>
            isFeedback(messages) ? feedback(advice) : answer(answers[index++] ?? 'exhausted'),
    });
}

describe('Refine', () => {
    it('asks for advice after a failed attempt and passes it to the next as hint_', async () => {
        const lm = refineLM(['The city of Paris', 'Paris']);
        const refine = new Refine(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: oneWord,
            threshold: 1,
        });

        const result = await refine.forward({ question: 'Capital of France?' });

        expect(result.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(3);
        const [first, advice, second] = lm.calls;
        expect(text(first.messages)).not.toContain('hint_');
        expect(isFeedback(advice.messages)).toBe(true);
        expect(text(second.messages)).toContain('hint_');
        expect(text(second.messages)).toContain('Reply with a single word.');
    });

    it('shows the feedback model the inputs, trajectory, outputs and reward', async () => {
        const lm = refineLM(['The city of Paris', 'Paris']);
        await new Refine(new Predict('question -> answer', lm), {
            n: 2,
            rewardFn: oneWord,
            threshold: 1,
        }).forward({ question: 'Capital of France?' });

        const prompt = text(lm.calls[1].messages);
        expect(prompt).toContain('Capital of France?');
        expect(prompt).toContain('"module_name": "self"');
        expect(prompt).toContain('The city of Paris');
        expect(prompt).toContain('Module self');
        expect(prompt).toContain('prediction.answer');
    });

    it('does not ask for advice after the last attempt or a passing one', async () => {
        const lm = refineLM(['Paris']);
        await new Refine(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: oneWord,
            threshold: 1,
        }).forward({ question: 'Q' });
        expect(lm.calls).toHaveLength(1);

        const failing = refineLM(['a b', 'c d']);
        const result = await new Refine(new Predict('question -> answer', failing), {
            n: 2,
            rewardFn: (_inputs, prediction: any) => -String(prediction.answer).length,
            threshold: 100,
        }).forward({ question: 'Q' });
        // attempt, advice, attempt — no advice after the final attempt.
        expect(failing.calls).toHaveLength(3);
        expect(result.answer).toBe('a b');
    });

    it('gives each predictor of a composite its own advice, N/A when it has none', async () => {
        const lm = new MockLM({
            respond: (messages) => {
                const prompt = text(messages);
                if (isFeedback(messages)) return feedback('{"draft": "Be brief."}');
                if (prompt.includes('[[ ## question ## ]]')) {
                    return `[[ ## draft ## ]]\nlong draft\n\n[[ ## completed ## ]]`;
                }
                return answer(prompt.includes('hint_') ? 'short' : 'too long');
            },
        });
        class Pipeline extends Module {
            draft = new Predict('question -> draft', lm);
            polish = new Predict('draft -> answer', lm);

            async forward(inputs: Record<string, any>, _options?: LLMCallOptions) {
                const { draft } = await this.draft.forward(inputs);
                return this.polish.forward({ draft });
            }
        }

        const result = await new Refine(new Pipeline(), {
            n: 2,
            rewardFn: oneWord,
            threshold: 1,
        }).forward({ question: 'Q' });

        expect(result.answer).toBe('short');
        const hinted = lm.calls.slice(3).map((call) => text(call.messages));
        expect(hinted[0]).toContain('Be brief.');
        expect(hinted[1]).toContain('N/A');
    });

    it('keeps the feedback predictor out of the program’s predictors and saved state', () => {
        const refine = new Refine(new Predict('question -> answer'), {
            n: 2,
            rewardFn: oneWord,
            threshold: 1,
        });

        expect(refine.namedPredictors().map(([path]) => path)).toEqual(['module']);
        expect(Object.keys(refine.dumpState()).sort()).toEqual(['metadata', 'module']);
        expect(refine.deepCopy().namedPredictors()).toHaveLength(1);
    });

    it('uses the scope model for feedback when the program has none of its own', async () => {
        const lm = refineLM(['a b', 'Paris']);
        const result = await context({ lm }, () =>
            new Refine(new Predict('question -> answer'), {
                n: 2,
                rewardFn: oneWord,
                threshold: 1,
            }).forward({ question: 'Q' })
        );

        expect(result.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(3);
    });

    it('counts a failed feedback call against failCount and carries on', async () => {
        let index = 0;
        const answers = ['a b', 'Paris'];
        const lm = new MockLM({
            respond: (messages) => {
                if (isFeedback(messages)) throw new Error('feedback model down');
                return answer(answers[index++]!);
            },
        });

        const result = await new Refine(new Predict('question -> answer', lm), {
            n: 3,
            rewardFn: oneWord,
            threshold: 1,
        }).forward({ question: 'Q' });
        expect(result.answer).toBe('Paris');

        await expect(
            new Refine(new Predict('question -> answer', lm), {
                n: 3,
                rewardFn: () => 0,
                threshold: 1,
                failCount: 0,
            }).forward({ question: 'Q' })
        ).rejects.toThrow('feedback model down');
    });
});

describe('parseAdvice', () => {
    it('reads a JSON object, fenced or not', () => {
        expect(parseAdvice('{"a": "x", "b": {"k": 1}}', ['a', 'b'])).toEqual({
            a: 'x',
            b: '{\n  "k": 1\n}',
        });
        expect(parseAdvice('```json\n{"a": "x"}\n```', ['a'])).toEqual({ a: 'x' });
    });

    it('applies free text to every predictor', () => {
        expect(parseAdvice('Be brief.', ['a', 'b'])).toEqual({
            a: 'Be brief.',
            b: 'Be brief.',
        });
    });
});
