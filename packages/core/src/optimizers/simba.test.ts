import { describe, expect, it } from 'vitest';
import { SIMBA } from './simba';
import { Predict } from '../modules/predict';
import { MockLM } from '../testing/mock-lm';
import type { OptimizerProgressEvent } from './utils';
import { doublingLM, doublingSet, exactAnswer, reply, viewPrompt } from './test-support';

/**
 * Right with a demo or the magic word in the instructions; otherwise right
 * only when sampled hot. Hot and cold runs disagree, which is SIMBA's signal.
 */
const taskLM = () =>
    doublingLM(
        (view) =>
            view.demos.length > 0 ||
            view.system.includes('DOUBLE') ||
            (view.options?.temperature ?? 0) >= 0.9
    );

function advisor() {
    return new MockLM({
        respond: () =>
            reply({
                discussion: 'The better run doubled the number.',
                module_advice: JSON.stringify({ self: 'Always DOUBLE the number.' }),
            }),
    });
}

const trainset = doublingSet([1, 2, 3, 4, 5, 6, 7, 8]);
const valset = doublingSet([11, 12, 13, 14]);

describe('SIMBA', () => {
    it('climbs from variable examples to a program that is always right', async () => {
        const events: OptimizerProgressEvent[] = [];
        const student = new Predict('question -> answer', taskLM());
        const optimizer = new SIMBA({
            metric: exactAnswer,
            promptModel: advisor(),
            bsize: 4,
            numCandidates: 3,
            maxSteps: 2,
            maxDemos: 2,
            seed: 5,
            onProgress: (event) => events.push(event),
        });

        const compiled = await optimizer.compile(student, { trainset, valset });
        const results = optimizer.results!;

        // Cold runs fail and the hot run succeeds, so every example is variable.
        expect(results.steps[0].gaps).toEqual([1, 1, 1, 1]);
        expect(results.steps[0].candidates.length).toBeGreaterThan(0);
        expect(results.steps[0].candidates.every((c) => c.score === 1)).toBe(true);

        expect(results.bestScore).toBe(1);
        expect(results.pool[0].fullScore).toBe(0);
        const improved =
            compiled.getDemos().length > 0 ||
            (compiled.getInstructions() ?? '').includes('DOUBLE');
        expect(improved).toBe(true);
        expect(compiled.getDemos().length).toBeLessThanOrEqual(2);
        expect(compiled.simbaResults).toBe(results);

        expect(student.getDemos()).toEqual([]);
        expect(student.getInstructions()).toBeUndefined();
        expect(events.map((e) => e.phase)).toEqual(['step', 'step', 'evaluate']);
    });

    it('appends a self-reflective rule comparing a better and a worse run', async () => {
        const promptModel = advisor();
        const compiled = await new SIMBA({
            metric: exactAnswer,
            promptModel,
            bsize: 2,
            numCandidates: 2,
            maxSteps: 1,
            strategies: ['rule'],
        }).compile(new Predict('question -> answer', taskLM()), { trainset, valset });

        expect(compiled.getInstructions()).toBe(
            'Given the fields `question`, produce the fields `answer`.\n\nAlways DOUBLE the number.'
        );
        const asked = viewPrompt(promptModel.calls[0].messages).inputs;
        expect(asked.better_score).toBe('1');
        expect(asked.worse_score).toBe('0');
        expect(asked.better_trajectory).toContain('Step 1: self');
        expect(asked.module_names).toBe('self');
    });

    it('builds nothing when every run agrees', async () => {
        const optimizer = new SIMBA({
            metric: exactAnswer,
            bsize: 2,
            numCandidates: 2,
            maxSteps: 1,
            maxDemos: 0,
            strategies: ['demo', 'rule'],
        });
        const compiled = await optimizer.compile(
            new Predict(
                'question -> answer',
                doublingLM(() => true)
            ),
            { trainset }
        );
        expect(optimizer.results!.steps[0].candidates).toEqual([]);
        expect(optimizer.results!.pool).toHaveLength(1);
        expect(compiled.getDemos()).toEqual([]);
    });

    it('validates its options', () => {
        expect(() => new SIMBA({} as never)).toThrow(/metric/);
        return expect(
            new SIMBA({ metric: exactAnswer, maxDemos: 0, strategies: ['demo'] }).compile(
                new Predict('question -> answer'),
                { trainset }
            )
        ).rejects.toThrow(/strategy/);
    });
});
