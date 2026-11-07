import { afterEach, describe, expect, it } from 'vitest';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { GEPA, buildReflectionPrompt, extractInstruction } from './gepa';
import { Module } from '../core/module';
import { context } from '../core/config';
import { Predict } from '../modules/predict';
import { MockLM } from '../testing/mock-lm';
import { contentToText } from '../utils/content';
import type { Example } from '../core/example';
import type { Prediction } from '../core/prediction';
import type { LLMCallOptions } from '../types/language-model';
import type { MetricTraceStep } from '../evaluate/types';
import { doublingLM, doublingSet, numberIn, reply } from './test-support';

/** Right only with the magic word in the instructions. */
const magicWordLM = () => doublingLM((view) => view.system.includes('DOUBLE'));

/** Exact match, with feedback that tells the reflection model what went wrong. */
const feedbackMetric = (example: Example, prediction: Prediction) => {
    const right = String(example.get('answer')) === String(prediction.get('answer'));
    return {
        score: right ? 1 : 0,
        feedback: right
            ? 'Correct.'
            : `Wrong: the answer should be twice the number, ${example.get('answer')}.`,
    };
};

/** Proposes the magic word once the feedback says what the task is. */
function reflector() {
    let calls = 0;
    return new MockLM({
        respond: (messages) => {
            calls += 1;
            const prompt = contentToText(messages[0].content);
            return prompt.includes('twice the number')
                ? 'Here you go:\n```\nDOUBLE the number in the question.\n```'
                : `\`\`\`\nTry harder, attempt ${calls}.\n\`\`\``;
        },
    });
}

const tempDirs: string[] = [];
afterEach(async () => {
    await Promise.all(
        tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true }))
    );
});

describe('GEPA', () => {
    it('evolves an instruction from reflective feedback', async () => {
        const reflectionLM = reflector();
        const seen: Array<{ predName?: string; predTrace?: MetricTraceStep[] }> = [];
        const metric = (
            example: Example,
            prediction: Prediction,
            _trace?: MetricTraceStep[],
            predName?: string,
            predTrace?: MetricTraceStep[]
        ) => {
            if (predName !== undefined) seen.push({ predName, predTrace });
            return feedbackMetric(example, prediction);
        };
        const student = new Predict('question -> answer', magicWordLM());

        const optimizer = new GEPA({
            metric,
            reflectionLM,
            maxMetricCalls: 20,
            reflectionMinibatchSize: 2,
            trackStats: true,
            seed: 1,
        });
        const compiled = await optimizer.compile(student, {
            trainset: doublingSet([1, 2, 3, 4]),
            valset: doublingSet([5, 6, 7, 8]),
        });
        const results = optimizer.detailedResults!;

        expect(compiled.getInstructions()).toBe('DOUBLE the number in the question.');
        expect(student.getInstructions()).toBeUndefined();
        expect(results.candidates[0]).toMatchObject({ origin: 'seed', score: 0, parents: [] });
        expect(results.candidates[1]).toMatchObject({
            origin: 'reflection',
            score: 1,
            parents: [0],
        });
        expect(results.bestIndex).toBe(1);
        expect(results.scores).toEqual([0, 1]);
        expect(results.lineage).toEqual({ 0: [], 1: [0] });
        expect(results.paretoFront.every((front) => front.candidates.includes(1))).toBe(true);
        expect(results.totalMetricCalls).toBeGreaterThanOrEqual(20);
        expect(results.iterationLog?.[0]).toMatchObject({
            kind: 'reflection',
            accepted: true,
            predictor: 'self',
            oldScore: 0,
            newScore: 1,
        });
        expect(compiled.detailedResults).toBe(results);

        // The reflection prompt shows the instruction, the outputs and the feedback.
        const prompt = contentToText(reflectionLM.calls[0].messages[0].content);
        expect(prompt).toContain('Given the fields `question`, produce the fields `answer`.');
        expect(prompt).toContain('## Generated Outputs');
        expect(prompt).toContain('## Feedback\nWrong: the answer should be twice the number');

        // The metric was asked about the predictor being improved, with its own steps.
        expect(seen[0].predName).toBe('self');
        expect(seen[0].predTrace).toHaveLength(1);
        expect(seen[0].predTrace?.[0].path).toBe('self');
    });

    it('merges two frontier candidates that improved different predictors', async () => {
        /**
         * Small evens need the hint step's "Write X"; odds need the solve step's
         * "Use Y"; and 100 is broken by X. So improving either predictor alone
         * trades one example for another, keeping several candidates on the
         * Pareto front — and the merge of the two branches beats both.
         */
        class Pipeline extends Module {
            hint = new Predict('question -> hint');
            solve = new Predict('question, hint -> answer');
            async forward(inputs: Record<string, any>, options?: LLMCallOptions) {
                const { hint } = await this.hint.forward(inputs, options);
                return this.solve.forward({ question: inputs.question, hint }, options);
            }
        }
        const lm = doublingLM(
            (view, n) => {
                const usedX = view.inputs.hint === 'use-x';
                if (n === 100) return !usedX;
                return n % 2 === 0 ? usedX : view.system.includes('Use Y');
            },
            (view) =>
                view.outputs.includes('hint')
                    ? reply({ hint: view.system.includes('Write X') ? 'use-x' : 'none' })
                    : undefined
        );
        let proposals = 0;
        const reflectionLM = new MockLM({
            respond: (messages) => {
                proposals += 1;
                const prompt = contentToText(messages[0].content);
                const forHint = /produce the fields `hint`|Write X/.test(
                    prompt.split('```')[1]
                );
                return `\`\`\`\n${forHint ? 'Write X' : 'Use Y'} (${proposals})\n\`\`\``;
            },
        });

        const optimizer = new GEPA({
            metric: feedbackMetric,
            reflectionLM,
            maxMetricCalls: 60,
            reflectionMinibatchSize: 2,
            seed: 2,
            trackStats: true,
        });
        const compiled = await context({ lm }, () =>
            optimizer.compile(new Pipeline(), {
                trainset: doublingSet([2, 3]),
                valset: [...doublingSet([2, 3]), ...doublingSet([100])],
            })
        );
        const results = optimizer.detailedResults!;

        const merge = results.candidates.find((c) => c.origin === 'merge');
        expect(merge).toBeDefined();
        expect(merge!.parents).toHaveLength(2);
        expect(merge!.instructions.hint).toMatch(/^Write X/);
        expect(merge!.instructions.solve).toMatch(/^Use Y/);
        // Each parent improved a different predictor of the common seed.
        const [p1, p2] = merge!.parents.map((id) => results.candidates[id]);
        expect(p1.parents).toEqual([0]);
        expect(p2.parents).toEqual([0]);
        expect(results.lineage[merge!.id]).toEqual(merge!.parents);
        expect(results.iterationLog?.some((entry) => entry.kind === 'merge')).toBe(true);

        // Several candidates share the front: each wins some example.
        const onFront = new Set(results.paretoFront.flatMap((front) => front.candidates));
        expect(onFront.size).toBeGreaterThan(1);
        expect(compiled.namedPredictors().map(([path]) => path)).toEqual(['hint', 'solve']);
        expect(results.scores[results.bestIndex]).toBeCloseTo(2 / 3);
    });

    it('skips minibatches the candidate already aces', async () => {
        const reflectionLM = reflector();
        const optimizer = new GEPA({
            metric: feedbackMetric,
            reflectionLM,
            maxFullEvals: 3,
            reflectionMinibatchSize: 2,
            trackStats: true,
        });
        const compiled = await optimizer.compile(
            new Predict(
                'question -> answer',
                doublingLM(() => true)
            ),
            { trainset: doublingSet([1, 2]) }
        );
        expect(reflectionLM.calls).toHaveLength(0);
        expect(optimizer.detailedResults!.candidates).toHaveLength(1);
        expect(
            optimizer.detailedResults!.iterationLog?.every((e) => e.kind === 'skipped')
        ).toBe(true);
        expect(optimizer.detailedResults!.budget).toBe(6);
        expect(compiled.getInstructions()).toBeUndefined();
    });

    it('selects the current best when asked to', async () => {
        const optimizer = new GEPA({
            metric: feedbackMetric,
            reflectionLM: reflector(),
            maxMetricCalls: 30,
            reflectionMinibatchSize: 2,
            candidateSelectionStrategy: 'current_best',
            useMerge: false,
        });
        const compiled = await optimizer.compile(
            new Predict('question -> answer', magicWordLM()),
            { trainset: doublingSet([1, 2, 3]) }
        );
        expect(compiled.getInstructions()).toBe('DOUBLE the number in the question.');
        expect(optimizer.detailedResults!.candidates.some((c) => c.origin === 'merge')).toBe(
            false
        );
    });

    it('checkpoints its state to logDir and resumes from it', async () => {
        const logDir = await mkdtemp(join(tmpdir(), 'gepa-'));
        tempDirs.push(logDir);
        const make = (budget: number, reflectionLM = reflector()) =>
            new GEPA({
                metric: feedbackMetric,
                reflectionLM,
                maxMetricCalls: budget,
                reflectionMinibatchSize: 2,
                logDir,
                seed: 4,
            });
        const data = { trainset: doublingSet([1, 2, 3, 4]), valset: doublingSet([5, 6]) };

        // A budget that only covers the seed evaluation.
        const first = make(2);
        await first.compile(new Predict('question -> answer', magicWordLM()), data);
        const saved = JSON.parse(await readFile(join(logDir, 'gepa_state.json'), 'utf8'));
        expect(saved.candidates).toHaveLength(1);
        expect(saved.metricCalls).toBe(2);
        expect(saved.paths).toEqual(['self']);

        // A bigger budget picks up where the first run stopped: no seed re-run.
        const taskLM = magicWordLM();
        const second = make(14);
        const compiled = await second.compile(new Predict('question -> answer', taskLM), data);
        expect(compiled.getInstructions()).toBe('DOUBLE the number in the question.');
        expect(second.detailedResults!.candidates[0].discoveredAtCalls).toBe(2);
        expect(second.detailedResults!.logDir).toBe(logDir);
        const resumed = JSON.parse(await readFile(join(logDir, 'gepa_state.json'), 'utf8'));
        expect(resumed.metricCalls).toBe(second.detailedResults!.totalMetricCalls);

        // A run over a different valset cannot resume from it.
        await expect(
            make(20).compile(new Predict('question -> answer'), {
                trainset: data.trainset,
                valset: doublingSet([5, 6, 7]),
            })
        ).rejects.toThrow(/different program or valset/);
    });

    it('validates its options and computes budgets', () => {
        const reflectionLM = reflector();
        expect(() => new GEPA({ reflectionLM, maxMetricCalls: 1 } as never)).toThrow(/metric/);
        expect(() => new GEPA({ metric: feedbackMetric, maxMetricCalls: 1 } as never)).toThrow(
            /reflectionLM/
        );
        expect(() => new GEPA({ metric: feedbackMetric, reflectionLM })).toThrow(/budget/);
        expect(
            () =>
                new GEPA({
                    metric: feedbackMetric,
                    reflectionLM,
                    auto: 'light',
                    maxFullEvals: 2,
                })
        ).toThrow(/budget/);
        const light = new GEPA({ metric: feedbackMetric, reflectionLM, auto: 'light' });
        expect(light.budgetFor(10, 2)).toBe(7 * 10 + 6 * 2 * 8 * 3);
        expect(
            new GEPA({ metric: feedbackMetric, reflectionLM, maxFullEvals: 4 }).budgetFor(10, 1)
        ).toBe(40);
    });
});

describe('GEPA prompt helpers', () => {
    it('extracts the fenced instruction, dropping a language tag', () => {
        expect(extractInstruction('Sure.\n```text\nDo the thing.\n```\nDone.')).toBe(
            'Do the thing.'
        );
        expect(extractInstruction('```\nfirst\n```\nand\n```\nsecond\n```')).toContain('first');
        expect(extractInstruction('No fences here.')).toBe('No fences here.');
    });

    it('builds the reflection prompt around the instruction and examples', () => {
        const prompt = buildReflectionPrompt('Old instruction.', '# Example 1');
        expect(prompt).toContain('```\nOld instruction.\n```');
        expect(prompt).toContain('```\n# Example 1\n```');
        expect(prompt).toContain('within ``` blocks');
        expect(numberIn('x 12')).toBe(12);
    });
});
