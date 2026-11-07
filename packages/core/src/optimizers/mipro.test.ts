import { describe, expect, it } from 'vitest';
import { MIPROv2, MIPRO_TIPS, trialsFor } from './mipro';
import { Predict } from '../modules/predict';
import { MockLM } from '../testing/mock-lm';
import type { OptimizerProgressEvent } from './utils';
import { doublingLM, doublingSet, exactAnswer, reply, viewPrompt } from './test-support';

/**
 * Right only with the magic word in the instructions — and, for odd numbers,
 * only with demos as well. So the best program needs both halves of the
 * search: the right instruction *and* a demo set.
 */
const taskLM = () =>
    doublingLM(
        (view, n) => view.system.includes('DOUBLE') && (n % 2 === 0 || view.demos.length > 0)
    );

/** Summarises the dataset when asked; every third proposal has the magic word. */
function proposer() {
    let proposals = 0;
    return new MockLM({
        respond: (messages) => {
            const view = viewPrompt(messages);
            if (view.outputs.includes('observations')) {
                return reply({ observations: 'Each question asks to double a number.' });
            }
            proposals += 1;
            return reply({
                proposed_instruction:
                    proposals % 3 === 2
                        ? `DOUBLE the number (variant ${proposals}).`
                        : `Answer the question (variant ${proposals}).`,
            });
        },
    });
}

const trainset = doublingSet([1, 2, 3, 4, 5, 6]);
const valset = doublingSet([11, 12, 13, 14, 15, 16, 17, 18]);

describe('MIPROv2', () => {
    it('finds the instruction and demo set that work together', async () => {
        const promptModel = proposer();
        const events: OptimizerProgressEvent[] = [];
        const student = new Predict('question -> answer', taskLM());

        const optimizer = new MIPROv2({
            metric: exactAnswer,
            promptModel,
            auto: null,
            numCandidates: 4,
            numTrials: 16,
            minibatchSize: 4,
            minibatchFullEvalSteps: 4,
            maxBootstrappedDemos: 2,
            maxLabeledDemos: 2,
            seed: 3,
            onProgress: (event) => events.push(event),
        });
        const compiled = await optimizer.compile(student, { trainset, valset });
        const results = optimizer.results!;

        expect(results.bestScore).toBe(1);
        expect(compiled.getInstructions()).toMatch(/^DOUBLE/);
        expect(compiled.getDemos().length).toBeGreaterThan(0);
        expect(results.best.self.instruction).toBe(compiled.getInstructions());
        expect(compiled.miproResults).toBe(results);

        // Candidate 0 is the original instruction; proposals follow.
        expect(results.instructionCandidates.self[0]).toBe(
            'Given the fields `question`, produce the fields `answer`.'
        );
        expect(results.instructionCandidates.self).toHaveLength(4);
        expect(results.demoSetCandidates).toBe(4);
        expect(results.datasetSummary).toBe('Each question asks to double a number.');

        // Trial 0 is the baseline on the full valset; the rest are minibatches,
        // with periodic full evaluations of the most promising configuration.
        expect(results.trials).toHaveLength(17);
        expect(results.trials[0]).toMatchObject({ trial: 0, minibatch: false, score: 0 });
        expect(results.trials.slice(1).every((t) => t.minibatch)).toBe(true);
        expect(results.fullEvaluations.map((e) => e.trial)).toEqual([0, 4, 8, 12, 16]);
        expect(events.some((e) => e.phase === 'evaluate')).toBe(true);

        // Proposals are grounded in the summary, the program, demos and a tip.
        const proposal = viewPrompt(
            promptModel.calls.find((call) =>
                viewPrompt(call.messages).outputs.includes('proposed_instruction')
            )!.messages
        );
        expect(proposal.inputs.dataset_summary).toBe('Each question asks to double a number.');
        expect(proposal.inputs.program_description).toContain(
            'self (Predict): question -> answer'
        );
        expect(MIPRO_TIPS).toContain(proposal.inputs.tip);

        expect(student.getInstructions()).toBeUndefined();
        expect(student.getDemos()).toEqual([]);
    });

    it('scores every trial on the full valset without minibatching, and is deterministic', async () => {
        const run = async () => {
            const optimizer = new MIPROv2({
                metric: exactAnswer,
                promptModel: proposer(),
                taskModel: taskLM(),
                auto: null,
                numCandidates: 3,
                numTrials: 6,
                minibatch: false,
                maxBootstrappedDemos: 0,
                maxLabeledDemos: 0,
                seed: 1,
            });
            // No model on the student: the task model serves the search.
            const compiled = await optimizer.compile(new Predict('question -> answer'), {
                trainset,
                valset,
            });
            return { optimizer, compiled };
        };
        const { optimizer, compiled } = await run();
        const results = optimizer.results!;
        expect(results.trials.every((t) => !t.minibatch)).toBe(true);
        expect(results.demoSetCandidates).toBe(0);
        // Instructions only, no demos: the magic word gets the evens right.
        expect(results.bestScore).toBe(0.5);
        expect(compiled.getInstructions()).toMatch(/^DOUBLE/);
        expect(compiled.getDemos()).toEqual([]);

        const again = (await run()).optimizer.results!;
        expect(again.trials.map((t) => [t.params, t.score])).toEqual(
            results.trials.map((t) => [t.params, t.score])
        );
    });

    it('derives the trial count from an auto preset', async () => {
        const optimizer = new MIPROv2({
            metric: exactAnswer,
            promptModel: proposer(),
            auto: 'light',
            maxBootstrappedDemos: 0,
            maxLabeledDemos: 0,
            dataAwareProposer: false,
        });
        await optimizer.compile(new Predict('question -> answer', taskLM()), { trainset });
        expect(optimizer.results!.numCandidates).toBe(6);
        expect(optimizer.results!.numTrials).toBe(trialsFor(6, 1, false));
        expect(optimizer.results!.datasetSummary).toBeUndefined();
    });

    it('computes DSPy trial counts', () => {
        expect(trialsFor(6, 1, true)).toBe(10);
        expect(trialsFor(12, 1, true)).toBe(18);
        expect(trialsFor(6, 3, true)).toBe(31);
    });

    it('validates its options', () => {
        expect(() => new MIPROv2({} as never)).toThrow(/metric/);
        expect(() => new MIPROv2({ metric: exactAnswer, auto: null })).toThrow(/numCandidates/);
        expect(() => new MIPROv2({ metric: exactAnswer, auto: 'huge' as never })).toThrow(
            /auto/
        );
    });
});
