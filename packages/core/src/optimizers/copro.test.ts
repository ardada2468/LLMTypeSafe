import { describe, expect, it } from 'vitest';
import { COPRO } from './copro';
import { Module } from '../core/module';
import { context } from '../core/config';
import { Predict } from '../modules/predict';
import { MockLM } from '../testing/mock-lm';
import type { LLMCallOptions } from '../types/language-model';
import type { OptimizerProgressEvent } from './utils';
import { doublingLM, doublingSet, exactAnswer, reply, viewPrompt } from './test-support';

/** Gets the answer right only when its instructions say the magic words. */
const magicWordLM = () => doublingLM((view) => view.system.includes('DOUBLE IT'));

/**
 * A proposer that flounders until it is shown scored attempts, then finds the
 * magic words — so only a search that feeds back scores can succeed.
 */
function learningProposer() {
    let fresh = 0;
    const lm = new MockLM({
        respond: (messages) => {
            const view = viewPrompt(messages);
            const previous = view.inputs.previous_instructions ?? '';
            if (previous.startsWith('None')) {
                fresh += 1;
                return reply({ proposed_instruction: `Attempt number ${fresh}.` });
            }
            return reply({ proposed_instruction: 'DOUBLE IT, carefully.' });
        },
    });
    return lm;
}

class Pipeline extends Module {
    hint = new Predict('question -> hint');
    solve = new Predict('question, hint -> answer');

    async forward(inputs: Record<string, any>, options?: LLMCallOptions) {
        const { hint } = await this.hint.forward(inputs, options);
        return this.solve.forward({ question: inputs.question, hint }, options);
    }
}

const trainset = doublingSet([1, 2, 3, 4]);

describe('COPRO', () => {
    it('iterates on instructions using the scores of earlier attempts', async () => {
        const proposer = learningProposer();
        const student = new Predict('question -> answer', magicWordLM());
        const events: OptimizerProgressEvent[] = [];

        const optimizer = new COPRO({
            metric: exactAnswer,
            promptModel: proposer,
            breadth: 3,
            depth: 3,
            onProgress: (event) => events.push(event),
        });
        const compiled = await optimizer.compile(student, { trainset });

        expect(compiled.getInstructions()).toBe('DOUBLE IT, carefully.');
        expect(student.getInstructions()).toBeUndefined();

        const results = optimizer.results!;
        expect(results.bestScore).toBe(1);
        // Round 0: the original plus two fresh proposals. Round 1: the proposer,
        // now shown scores, finds the answer (repeats are not re-scored).
        expect(results.history.map((a) => [a.depth, a.score])).toEqual([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ]);
        expect(compiled.coproResults).toBe(results);
        expect(events.map((e) => e.step)).toEqual([1, 2, 3]);

        // Later rounds condition on what was tried, with scores.
        const later = proposer.calls
            .map((call) => viewPrompt(call.messages).inputs.previous_instructions ?? '')
            .find((text) => !text.startsWith('None'))!;
        expect(later).toContain('Instruction: Attempt number 1.');
        expect(later).toContain('Score: 0.000');
        // Proposals sample hot.
        expect(proposer.calls[0].options?.temperature).toBe(1.4);
    });

    it('tunes each predictor of a composite in turn, keeping what does not help', async () => {
        const lm = doublingLM(
            (view) => view.system.includes('DOUBLE IT'),
            (view) => (view.outputs.includes('hint') ? reply({ hint: 'think' }) : undefined)
        );
        const student = new Pipeline();
        // The pipeline's predictors have no model of their own; a scope supplies one.
        const compiled = await context({ lm }, () =>
            new COPRO({
                metric: exactAnswer,
                promptModel: learningProposer(),
                breadth: 2,
                depth: 2,
            }).compile(student, { trainset: doublingSet([1, 2]) })
        );

        // Nothing helps the hint step, so it keeps its signature's instructions.
        expect(compiled.hint.getInstructions()).toBeUndefined();
        expect(compiled.solve.getInstructions()).toBe('DOUBLE IT, carefully.');
    });

    it('validates its options', () => {
        expect(() => new COPRO({} as never)).toThrow(/metric/);
        expect(() => new COPRO({ metric: exactAnswer, breadth: 1 })).toThrow(RangeError);
        expect(() => new COPRO({ metric: exactAnswer, depth: 0 })).toThrow(RangeError);
    });
});
