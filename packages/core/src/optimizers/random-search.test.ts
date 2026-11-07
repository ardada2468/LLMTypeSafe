import { describe, expect, it } from 'vitest';
import { BootstrapFewShotWithRandomSearch } from './random-search';
import { Predict } from '../modules/predict';
import type { PredictorState } from '../core/module';
import type { OptimizerProgressEvent } from './utils';
import {
    demoQuestions,
    doublingLM,
    doublingSet,
    exactAnswer,
    viewPrompt,
} from './test-support';

/** Evens are easy; odds come out right only when the demo for 8 is in the prompt. */
const goldenDemoLM = () =>
    doublingLM((view, n) => n % 2 === 0 || demoQuestions(view).includes('double 8'));

const trainset = doublingSet([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const valset = doublingSet([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);

describe('BootstrapFewShotWithRandomSearch', () => {
    it('scores zero-shot, labelled, unshuffled and shuffled candidates and keeps the best', async () => {
        const lm = goldenDemoLM();
        const student = new Predict('question -> answer', lm);
        const events: OptimizerProgressEvent[] = [];

        const optimizer = new BootstrapFewShotWithRandomSearch({
            metric: exactAnswer,
            numCandidatePrograms: 8,
            maxBootstrappedDemos: 2,
            maxLabeledDemos: 0,
            seed: 1,
            onProgress: (event) => events.push(event),
        });
        const compiled = await optimizer.compile(student, { trainset, valset });

        const results = optimizer.results!;
        expect(results.candidates).toHaveLength(11);
        const kinds = results.candidates.map((c) => c.kind);
        expect(kinds.filter((k) => k === 'shuffled-bootstrap')).toHaveLength(8);
        expect(kinds).toEqual(expect.arrayContaining(['zero-shot', 'labeled', 'bootstrap']));

        const zeroShot = results.candidates.find((c) => c.kind === 'zero-shot')!;
        expect(zeroShot.score).toBe(0.5);
        // The unshuffled bootstrap keeps the first two evens, so it misses the golden demo.
        const unshuffled = results.candidates.find((c) => c.kind === 'bootstrap')!;
        expect(unshuffled.score).toBe(0.5);

        // Best first, and the best is the golden-demo program.
        const scores = results.candidates.map((c) => c.score);
        expect(scores).toEqual([...scores].sort((a, b) => b - a));
        expect(results.bestScore).toBe(1);
        expect(compiled.getDemos().map((d) => d.get('question'))).toContain('double 8');

        // Candidate scores ride along on the program without becoming sub-modules.
        expect(compiled.candidatePrograms).toEqual(results.candidates);
        expect(Object.keys(compiled)).not.toContain('candidatePrograms');
        expect(compiled.namedPredictors()).toHaveLength(1);

        expect(student.getDemos()).toEqual([]);
        expect(events).toHaveLength(11);
        expect(events.at(-1)?.bestScore).toBe(1);

        await compiled.forward({ question: 'double 21' });
        expect(viewPrompt(lm.calls.at(-1)!.messages).demos.length).toBeGreaterThan(0);
    });

    it('is deterministic for a seed', async () => {
        const run = async () => {
            const optimizer = new BootstrapFewShotWithRandomSearch({
                metric: exactAnswer,
                numCandidatePrograms: 4,
                maxBootstrappedDemos: 3,
                maxLabeledDemos: 1,
                seed: 7,
            });
            await optimizer.compile(new Predict('question -> answer', goldenDemoLM()), {
                trainset,
                valset,
            });
            return optimizer.results!.candidates.map((c) => [c.kind, c.seed, c.score]);
        };
        expect(await run()).toEqual(await run());
    });

    it('stops early at stopAtScore', async () => {
        const optimizer = new BootstrapFewShotWithRandomSearch({
            metric: exactAnswer,
            numCandidatePrograms: 4,
            maxLabeledDemos: 16,
            stopAtScore: 1,
        });
        await optimizer.compile(new Predict('question -> answer', goldenDemoLM()), {
            trainset,
            valset,
        });
        // Labelled-only includes the golden demo, so the search stops there.
        expect(optimizer.results!.candidates.map((c) => c.kind)).toEqual([
            'labeled',
            'zero-shot',
        ]);
    });

    it('keeps the student instructions on every candidate, clearing only its demos', async () => {
        const lm = goldenDemoLM();
        const student = new Predict('question -> answer', lm)
            .withInstructions('Double the number.')
            .withDemos(doublingSet([99]));
        const optimizer = new BootstrapFewShotWithRandomSearch({
            metric: exactAnswer,
            numCandidatePrograms: 1,
            maxBootstrappedDemos: 2,
            maxLabeledDemos: 0,
        });
        const compiled = await optimizer.compile(student, { trainset, valset });

        for (const candidate of optimizer.results!.candidates) {
            expect((candidate.state.self as PredictorState).instructions).toBe(
                'Double the number.'
            );
        }
        const zeroShot = optimizer.results!.candidates.find((c) => c.kind === 'zero-shot')!;
        expect((zeroShot.state.self as PredictorState).demos).toEqual([]);
        expect(compiled.getInstructions()).toBe('Double the number.');
        expect(student.getDemos().map((d) => d.get('question'))).toEqual(['double 99']);
    });

    it('requires a metric', () => {
        expect(() => new BootstrapFewShotWithRandomSearch({} as never)).toThrow(/metric/);
    });
});
