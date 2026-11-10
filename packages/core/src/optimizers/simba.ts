import { type Module, type ProgramState } from '../core/module';
import { signature } from '../core/signature';
import { Predict } from '../modules/predict';
import type { ILanguageModel } from '../types/language-model';
import { type MetricTraceStep } from '../evaluate/types';
import { type Metric } from './types';
import { createRng, type Rng } from './random';
import { sampleCategorical } from './tpe';
import {
    attachResults,
    describeProgram,
    evaluateProgram,
    lanes,
    mean,
    metricScore,
    predictorInstruction,
    randomInt,
    reportProgress,
    runWithTrace,
    sampleMinibatch,
    stepToDemo,
    type BaseOptimizerOptions,
    type CompileConfig,
    transportOptions,
} from './utils';
import { resolveExampleInputs } from './types';
import { mapWithConcurrency } from '../utils/pool';

/** How SIMBA turns a variable example into a new candidate. */
export type SIMBAStrategy = 'demo' | 'rule';

export interface SIMBAOptions extends BaseOptimizerOptions {
    /** Scores every run. */
    metric: Metric;
    /** Examples per minibatch. Default 32. */
    bsize?: number;
    /** Runs per example per step (one per temperature), and new candidates per step. Default 6. */
    numCandidates?: number;
    /** Optimisation steps. Default 8. */
    maxSteps?: number;
    /** Most demos a predictor may carry; older ones are dropped. `0` disables the demo strategy. Default 4. */
    maxDemos?: number;
    /** Writes the self-reflective rules. Defaults to the configured model. */
    promptModel?: ILanguageModel;
    /** Strategies to choose between. Default both (`'demo'` only when `maxDemos > 0`). */
    strategies?: SIMBAStrategy[];
    /** Softmax temperature for picking which pool program to build on. Default 0.2. */
    temperatureForSampling?: number;
}

/** What one step did. */
export interface SIMBAStep {
    step: number;
    /** Per example in the minibatch: best minus worst score across the runs. */
    gaps: number[];
    /** The candidates built this step, with the strategy and minibatch score of each. */
    candidates: Array<{ strategy: SIMBAStrategy; score: number; poolIndex: number }>;
}

export interface SIMBAResults {
    bestScore: number;
    /** Every program the run considered; index 0 is the student. */
    pool: Array<{ state: ProgramState; batchScores: number[]; fullScore?: number }>;
    steps: SIMBAStep[];
}

const OfferFeedback = signature(
    [
        'program_description "the program\'s steps and their current instructions"',
        'worse_trajectory "a run that scored lower: each step\'s inputs and outputs"',
        'worse_score: number',
        'better_trajectory "a run of the same input that scored higher"',
        'better_score: number',
        'module_names "the steps to advise, one per line"',
    ].join(', ') +
        ' -> discussion "what the better run did differently", ' +
        'module_advice: json "an object mapping each step name to one concise, general rule ' +
        'that would make that step behave like the better run"',
    'Two runs of the same program on the same input scored differently. Compare them, ' +
        'work out what made the better run better, and write advice each step can follow ' +
        'on future inputs to get the higher score. Advice must be general, not specific to ' +
        'this input.'
);

/**
 * DSPy's `SIMBA` (Stochastic Introspective Mini-Batch Ascent).
 *
 * Each step samples a minibatch and runs programs from a growing pool on it
 * several times, at a spread of temperatures. Examples whose runs disagree —
 * high output variability, so the program *can* get them right — are the
 * signal. For the most variable ones SIMBA builds a new candidate from a pool
 * program by one of two strategies:
 *
 * - `'demo'` — append the successful run's steps as demos to the predictors
 *   that took them (at most `maxDemos` each, oldest dropped);
 * - `'rule'` — ask the prompt model to compare the best and worst runs and
 *   write one general rule per predictor, appended to its instructions.
 *
 * New candidates are scored on the same minibatch and join the pool; pool
 * programs are later sampled by a softmax over their mean minibatch scores.
 * At the end the student and each step's winner are scored on the valset (the
 * trainset when none is given) and the best is returned.
 *
 * Temperatures span 0.2 to 1.0 across `numCandidates` runs. The student is not
 * modified; {@link results} and the returned program's non-enumerable
 * `simbaResults` hold the pool and per-step history.
 */
export class SIMBA {
    private readonly options: SIMBAOptions;

    results?: SIMBAResults;

    constructor(options: SIMBAOptions) {
        if (typeof options?.metric !== 'function') {
            throw new Error('SIMBA requires a metric function.');
        }
        this.options = options;
    }

    async compile<M extends Module>(
        student: M,
        config: CompileConfig
    ): Promise<M & { simbaResults: SIMBAResults }> {
        const o = this.options;
        const rng = createRng(o.seed ?? 0);
        const bsize = Math.max(1, o.bsize ?? 32);
        const numCandidates = Math.max(1, o.numCandidates ?? 6);
        const maxSteps = Math.max(0, o.maxSteps ?? 8);
        const maxDemos = o.maxDemos ?? 4;
        const strategies = (o.strategies ?? ['demo', 'rule']).filter(
            (s) => s !== 'demo' || maxDemos > 0
        );
        if (strategies.length === 0) throw new Error('SIMBA has no strategy left to use.');
        const temperatures = Array.from({ length: numCandidates }, (_, i) =>
            numCandidates === 1 ? 0.2 : round(0.2 + (0.8 * i) / (numCandidates - 1))
        );

        const pool: Array<{ program: M; batchScores: number[] }> = [
            { program: student.deepCopy(), batchScores: [] },
        ];
        const winners = new Set<number>([0]);
        const steps: SIMBAStep[] = [];

        for (let step = 0; step < maxSteps; step++) {
            const batch = sampleMinibatch(config.trainset, bsize, rng);
            const inputs = batch.map((example, index) =>
                resolveExampleInputs(example, o.inputKeys, index)
            );

            // Run sampled pool programs at each temperature over the minibatch.
            const runs: Array<
                Array<{ poolIndex: number; trace: MetricTraceStep[]; score: number }>
            > = batch.map(() => []);
            const usedPrograms = new Map<number, number[]>();
            for (const temperature of temperatures) {
                const poolIndex = this.pick(pool, rng);
                const program = pool[poolIndex].program;
                const settled = await mapWithConcurrency(
                    batch,
                    async (example, index) => {
                        const run = await runWithTrace(program, inputs[index], {
                            ...o.callOptions,
                            temperature,
                        });
                        let score = 0;
                        if (run.prediction) {
                            try {
                                score = metricScore(
                                    await o.metric(example, run.prediction, run.trace)
                                );
                            } catch {
                                score = 0;
                            }
                        }
                        runs[index].push({ poolIndex, trace: run.trace, score });
                        return score;
                    },
                    { concurrency: lanes(o.concurrency) }
                );
                const scores = settled.map((r) => (r.status === 'fulfilled' ? r.value : 0));
                usedPrograms.set(poolIndex, [
                    ...(usedPrograms.get(poolIndex) ?? []),
                    ...scores,
                ]);
            }
            for (const [poolIndex, scores] of usedPrograms) {
                pool[poolIndex].batchScores.push(mean(scores));
            }

            // The most variable examples first; among equals, the ones with a higher best.
            const buckets = runs
                .map((bucket, index) => {
                    const sorted = [...bucket].sort((a, b) => b.score - a.score);
                    return {
                        index,
                        best: sorted[0],
                        worst: sorted[sorted.length - 1],
                        gap: sorted[0].score - sorted[sorted.length - 1].score,
                    };
                })
                .sort(
                    (a, b) => b.gap - a.gap || b.best.score - a.best.score || a.index - b.index
                );
            const gaps = runs.map((bucket) => {
                const scores = bucket.map((r) => r.score);
                return Math.max(...scores) - Math.min(...scores);
            });

            const record: SIMBAStep = { step: step + 1, gaps, candidates: [] };
            const useful = buckets.filter((bucket) => bucket.gap > 0).slice(0, numCandidates);
            const built: Array<{ program: M; strategy: SIMBAStrategy }> = [];
            for (const bucket of useful) {
                const source = pool[this.pick(pool, rng)].program;
                const strategy = strategies[randomInt(rng, strategies.length)];
                const candidate =
                    strategy === 'demo'
                        ? appendDemos(source, bucket.best.trace, maxDemos)
                        : await this.appendRule(source, bucket.worst, bucket.best);
                if (candidate) built.push({ program: candidate, strategy });
            }

            let stepBest: { poolIndex: number; score: number } | undefined;
            for (const { program, strategy } of built) {
                const { score } = await evaluateProgram(program, batch, o.metric, o);
                pool.push({ program, batchScores: [score] });
                const poolIndex = pool.length - 1;
                record.candidates.push({ strategy, score, poolIndex });
                if (!stepBest || score > stepBest.score) stepBest = { poolIndex, score };
            }
            if (stepBest) winners.add(stepBest.poolIndex);
            steps.push(record);
            reportProgress(o.onProgress, {
                optimizer: 'SIMBA',
                phase: 'step',
                step: step + 1,
                total: maxSteps,
                score: stepBest?.score,
                message: `${built.length} candidate(s) from ${useful.length} variable example(s)`,
            });
        }

        // Score the student and every step's winner on held-out data.
        const valset = config.valset ?? config.trainset;
        const fullScores = new Map<number, number>();
        let bestIndex = 0;
        for (const poolIndex of [...winners].sort((a, b) => a - b)) {
            const { score } = await evaluateProgram(
                pool[poolIndex].program,
                valset,
                o.metric,
                o
            );
            fullScores.set(poolIndex, score);
            if (score > fullScores.get(bestIndex)!) bestIndex = poolIndex;
        }
        reportProgress(o.onProgress, {
            optimizer: 'SIMBA',
            phase: 'evaluate',
            bestScore: fullScores.get(bestIndex),
        });

        this.results = {
            bestScore: fullScores.get(bestIndex) ?? 0,
            pool: pool.map((entry, index) => ({
                state: entry.program.dumpState(),
                batchScores: entry.batchScores,
                ...(fullScores.has(index) ? { fullScore: fullScores.get(index) } : {}),
            })),
            steps,
        };
        return attachResults(
            pool[bestIndex].program.deepCopy(),
            'simbaResults',
            this.results
        ) as M & {
            simbaResults: SIMBAResults;
        };
    }

    /** A pool index, sampled by softmax over mean minibatch scores. */
    private pick(pool: Array<{ batchScores: number[] }>, rng: Rng): number {
        const temperature = Math.max(1e-6, this.options.temperatureForSampling ?? 0.2);
        const means = pool.map((entry) => mean(entry.batchScores));
        const top = Math.max(...means);
        return sampleCategorical(
            means.map((m) => Math.exp((m - top) / temperature)),
            rng
        );
    }

    /**
     * Ask the prompt model for a rule per predictor, comparing a worse and a
     * better run, and append each rule to that predictor's instructions.
     * Returns `undefined` when the model gives no usable advice.
     */
    private async appendRule<M extends Module>(
        source: M,
        worse: { trace: MetricTraceStep[]; score: number },
        better: { trace: MetricTraceStep[]; score: number }
    ): Promise<M | undefined> {
        const paths = source.namedPredictors().map(([path]) => path);
        let advice: unknown;
        try {
            const prediction = await new Predict(OfferFeedback).forward(
                {
                    program_description: describeProgram(source),
                    worse_trajectory: renderTrajectory(worse.trace),
                    worse_score: worse.score,
                    better_trajectory: renderTrajectory(better.trace),
                    better_score: better.score,
                    module_names: paths.join('\n'),
                },
                {
                    ...transportOptions(this.options.callOptions),
                    ...(this.options.promptModel ? { lm: this.options.promptModel } : {}),
                }
            );
            advice = prediction.get('module_advice');
        } catch {
            return undefined;
        }
        if (typeof advice !== 'object' || advice === null) return undefined;

        const candidate = source.deepCopy();
        let changed = false;
        for (const [path, predictor] of candidate.namedPredictors()) {
            const rule = (advice as Record<string, unknown>)[path];
            if (typeof rule !== 'string' || rule.trim() === '') continue;
            predictor.setInstructions(`${predictorInstruction(predictor)}\n\n${rule.trim()}`);
            changed = true;
        }
        return changed ? candidate : undefined;
    }
}

/**
 * A copy of `source` where each predictor the successful run called gains a
 * demo of that call (its last, if it ran several times). Predictors over
 * `maxDemos` drop their oldest demos.
 */
function appendDemos<M extends Module>(
    source: M,
    trace: MetricTraceStep[],
    maxDemos: number
): M | undefined {
    if (trace.length === 0) return undefined;
    const lastByPath = new Map<string, MetricTraceStep>();
    for (const step of trace) lastByPath.set(step.path, step);
    const candidate = source.deepCopy();
    for (const [path, predictor] of candidate.namedPredictors()) {
        const step = lastByPath.get(path);
        if (!step) continue;
        predictor.setDemos([...predictor.getDemos(), stepToDemo(step)].slice(-maxDemos));
    }
    return candidate;
}

/** A trace as text for a prompt: each step's path, inputs and outputs. */
function renderTrajectory(trace: MetricTraceStep[]): string {
    if (trace.length === 0) return '(the run failed before any step finished)';
    return trace
        .map(
            (step, index) =>
                `Step ${index + 1}: ${step.path}\n  inputs: ${JSON.stringify(step.inputs)}\n` +
                `  outputs: ${JSON.stringify(step.outputs)}`
        )
        .join('\n');
}

function round(value: number): number {
    return Math.round(value * 100) / 100;
}
