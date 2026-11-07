import { describe, expect, it } from 'vitest';
import {
    attachResults,
    cleanInstruction,
    demoSetPlans,
    describeProgram,
    evaluateProgram,
    lanes,
    metricPasses,
    predictorInstruction,
    proposeInstructions,
    reportProgress,
    runWithTrace,
    sampleMinibatch,
    summarizeDataset,
} from './utils';
import { Module } from '../core/module';
import { configure } from '../core/config';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { MockLM } from '../testing/mock-lm';
import { createRng } from './random';
import type { Callback } from '../core/callbacks';
import type { LLMCallOptions } from '../types/language-model';
import { doublingLM, doublingSet, reply, viewPrompt } from './test-support';

class Pipeline extends Module {
    hint = new Predict('question -> hint');
    solve = new ChainOfThought('question, hint -> answer');

    async forward(inputs: Record<string, any>, options?: LLMCallOptions) {
        const { hint } = await this.hint.forward(inputs, options);
        return this.solve.forward({ question: inputs.question, hint }, options);
    }
}

const pipelineLM = () =>
    doublingLM(
        () => true,
        (view) =>
            view.outputs.includes('hint')
                ? reply({ hint: `hint for ${view.inputs.question}` })
                : undefined
    );

describe('runWithTrace', () => {
    it('records each predictor call by path, isolated per concurrent run', async () => {
        const lm = pipelineLM();
        const program = new Pipeline();
        const observed: string[] = [];
        const userCallback: Callback = { onModuleEnd: (e) => observed.push(e.moduleId) };
        configure({ callbacks: [userCallback] });
        try {
            const [a, b] = await Promise.all([
                runWithTrace(program, { question: 'double 1' }, { lm }),
                runWithTrace(program, { question: 'double 2' }, { lm }),
            ]);
            expect(a.trace.map((s) => s.path)).toEqual(['hint', 'solve']);
            expect(a.trace[0]).toEqual({
                path: 'hint',
                inputs: { question: 'double 1' },
                outputs: { hint: 'hint for double 1' },
            });
            expect(b.trace[1].inputs).toEqual({
                question: 'double 2',
                hint: 'hint for double 2',
            });
            expect(b.prediction?.get('answer')).toBe('4');
            // The user's own callbacks still fire.
            expect(observed.length).toBeGreaterThanOrEqual(6);
        } finally {
            configure({ callbacks: [] });
        }
    });

    it('captures a failure instead of throwing', async () => {
        const lm = new MockLM({
            respond: () => {
                throw new Error('boom');
            },
        });
        const run = await runWithTrace(new Predict('question -> answer', lm), {
            question: 'q',
        });
        expect(run.prediction).toBeUndefined();
        expect((run.error as Error).message).toBe('boom');
        expect(run.trace).toEqual([]);
    });
});

describe('optimizer utils', () => {
    it('evaluates a program and keeps per-example scores and feedback', async () => {
        const program = new Predict(
            'question -> answer',
            doublingLM((_v, n) => n === 2)
        );
        const result = await evaluateProgram(
            program,
            doublingSet([1, 2]),
            (example, prediction) => ({
                score: String(prediction.get('answer')) === example.get('answer') ? 1 : 0,
                feedback: `checked ${example.get('question')}`,
            })
        );
        expect(result.score).toBe(0.5);
        expect(result.scores).toEqual([0, 1]);
        expect(result.feedback).toEqual(['checked double 1', 'checked double 2']);
        expect(await evaluateProgram(program, [], () => 1)).toMatchObject({
            score: 0,
            scores: [],
        });
    });

    it('judges passes for every metric result shape', () => {
        expect(metricPasses(true, 0.5)).toBe(true);
        expect(metricPasses(0.5, 0.5)).toBe(true);
        expect(metricPasses(0.4, 0.5)).toBe(false);
        expect(metricPasses({ score: 0.9, feedback: 'ok' }, 0.5)).toBe(true);
        expect(metricPasses(NaN, 0)).toBe(false);
    });

    it('samples seeded minibatches without replacement', () => {
        const items = [1, 2, 3, 4, 5, 6];
        const a = sampleMinibatch(items, 3, createRng(1));
        expect(a).toHaveLength(3);
        expect(new Set(a).size).toBe(3);
        expect(sampleMinibatch(items, 3, createRng(1))).toEqual(a);
        expect(sampleMinibatch(items, 10, createRng(1)).sort()).toEqual(items);
        expect(sampleMinibatch(items, -1, createRng(1))).toEqual([]);
    });

    it('describes a program from its structure', () => {
        const program = new Pipeline();
        program.hint.setInstructions('Give a hint.');
        const text = describeProgram(program);
        expect(text).toContain('Pipeline with 2 step(s)');
        expect(text).toContain('1. hint (Predict): question -> hint');
        expect(text).toContain('Instructions: Give a hint.');
        expect(text).toContain(
            '2. solve (ChainOfThought): question, hint -> reasoning, answer'
        );
        expect(predictorInstruction(program.solve)).toContain('`reasoning`, `answer`');
    });

    it('proposes distinct, cleaned instructions and survives a failing proposer', async () => {
        let n = 0;
        const lm = new MockLM({
            respond: (messages) => {
                n += 1;
                if (n === 3) throw new Error('flaky');
                const view = viewPrompt(messages);
                expect(view.inputs.tip).toBeTruthy();
                return reply({ proposed_instruction: n === 2 ? '"Same."' : 'Same.' });
            },
        });
        const proposals = await proposeInstructions({
            lm,
            n: 3,
            basicInstruction: 'Answer.',
            previous: [{ instruction: 'Old.', score: 0.25 }],
            tips: ['Be brief.'],
        });
        expect(proposals).toEqual(['Same.']);
        expect(viewPrompt(lm.calls[0].messages).inputs.previous_instructions).toBe(
            'Instruction: Old.\nScore: 0.250'
        );
        expect(cleanInstruction('  `Do it.`  ')).toBe('Do it.');
    });

    it('summarises a dataset, or gives up quietly', async () => {
        const lm = new MockLM({ respond: () => reply({ observations: 'Doubling numbers.' }) });
        expect(await summarizeDataset(doublingSet([1, 2]), { lm, rng: createRng(1) })).toBe(
            'Doubling numbers.'
        );
        const broken = new MockLM({
            respond: () => {
                throw new Error('down');
            },
        });
        expect(await summarizeDataset(doublingSet([1]), { lm: broken })).toBeUndefined();
        expect(await summarizeDataset([], { lm })).toBeUndefined();
    });

    it('plans demo sets by DSPy seed convention', () => {
        expect(demoSetPlans(5, 10)).toEqual([
            { kind: 'zero-shot', seed: -3 },
            { kind: 'labeled', seed: -2 },
            { kind: 'bootstrap', seed: -1 },
            { kind: 'shuffled-bootstrap', seed: 10 },
            { kind: 'shuffled-bootstrap', seed: 11 },
        ]);
        expect(demoSetPlans(2, 0).map((p) => p.kind)).toEqual(['zero-shot', 'labeled']);
    });

    it('attaches results without making them part of the module tree', () => {
        const program = attachResults(new Predict('q -> a'), 'extra', {
            programs: [new Predict('q -> a')],
        });
        expect((program as unknown as { extra: unknown }).extra).toBeDefined();
        expect(program.namedPredictors()).toHaveLength(1);
        expect(Object.keys(program)).not.toContain('extra');
    });

    it('swallows progress callback errors and clamps lanes', () => {
        expect(() =>
            reportProgress(
                () => {
                    throw new Error('observer');
                },
                { optimizer: 'x', phase: 'y' }
            )
        ).not.toThrow();
        expect(lanes(undefined)).toBe(4);
        expect(lanes(0)).toBe(1);
        expect(lanes(3.7)).toBe(3);
    });
});
