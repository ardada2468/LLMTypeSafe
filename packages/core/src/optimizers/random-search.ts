import { type Module, type ProgramState } from '../core/module';
import type { ILanguageModel } from '../types/language-model';
import { type Metric } from './types';
import { createRng } from './random';
import {
    attachResults,
    buildDemoSet,
    demoSetPlans,
    evaluateProgram,
    reportProgress,
    withDemosByPath,
    withoutDemos,
    type BaseOptimizerOptions,
    type CompileConfig,
    type DemoSetPlan,
} from './utils';

export interface BootstrapFewShotWithRandomSearchOptions extends BaseOptimizerOptions {
    /** Scores every candidate, and judges bootstrap attempts. */
    metric: Metric;
    /**
     * Candidates beyond the three fixed ones (zero-shot, labelled-only,
     * unshuffled bootstrap). Default 8, as in DSPy — so up to 11 programs are
     * scored in all.
     */
    numCandidatePrograms?: number;
    /** Cap on bootstrapped demos per predictor. Default 4. */
    maxBootstrappedDemos?: number;
    /** Cap on labelled demos per predictor. Default 16. */
    maxLabeledDemos?: number;
    /** Score at which a numeric metric passes during bootstrapping. Default 0.5. */
    threshold?: number;
    /** Model that generates the bootstrapped traces. Defaults to the student's. */
    teacher?: ILanguageModel;
    /**
     * Stop early once a candidate's validation score reaches this. Default
     * none — every candidate is scored.
     */
    stopAtScore?: number;
}

/** One scored candidate. */
export interface RandomSearchCandidate {
    /** How it was built. */
    kind: DemoSetPlan['kind'];
    /** The seed that built it (DSPy's convention: -3, -2, -1, then 0, 1, …). */
    seed: number;
    /** Validation score. */
    score: number;
    /** Per-example validation scores. */
    scores: number[];
    /** The candidate's learned state; load it into a copy of the student to rebuild it. */
    state: ProgramState;
}

/** What a random search produced, best first. */
export interface RandomSearchResults {
    bestScore: number;
    candidates: RandomSearchCandidate[];
}

/**
 * DSPy's `BootstrapFewShotWithRandomSearch`: build many few-shot programs,
 * score each on the validation set, keep the best.
 *
 * The candidates, by seed as in DSPy:
 *
 * - `-3` — zero-shot: the student with no demos (its instructions are kept),
 *   the baseline to beat.
 * - `-2` — labelled demos only, no model calls to build.
 * - `-1` — bootstrapped demos from the trainset in its given order.
 * - `seed, seed + 1, …` — bootstrapped demos from a shuffled trainset, each
 *   with its own seed and a demo budget drawn between 1 and
 *   `maxBootstrappedDemos`.
 *
 * Demos are assigned per predictor from traced runs, so multi-step programs
 * work. The returned program is a copy; it carries `candidatePrograms` (every
 * candidate's score and state, best first) as a non-enumerable property, and
 * {@link results} holds the same for inspection.
 */
export class BootstrapFewShotWithRandomSearch {
    private readonly options: BootstrapFewShotWithRandomSearchOptions;

    /** What the last compile produced. */
    results?: RandomSearchResults;

    constructor(options: BootstrapFewShotWithRandomSearchOptions) {
        if (typeof options?.metric !== 'function') {
            throw new Error('BootstrapFewShotWithRandomSearch requires a metric function.');
        }
        this.options = options;
    }

    async compile<M extends Module>(
        student: M,
        config: CompileConfig
    ): Promise<M & { candidatePrograms: RandomSearchCandidate[] }> {
        const o = this.options;
        const valset = config.valset ?? config.trainset;
        const seed = o.seed ?? 0;
        const plans = demoSetPlans(3 + (o.numCandidatePrograms ?? 8), seed);
        const base = withoutDemos(student);
        const sizeRng = createRng(seed);

        const candidates: Array<RandomSearchCandidate & { program: M }> = [];
        for (const [index, plan] of plans.entries()) {
            const demos = await buildDemoSet(base, config.trainset, o.metric, plan, {
                maxBootstrappedDemos: o.maxBootstrappedDemos ?? 4,
                maxLabeledDemos: o.maxLabeledDemos ?? 16,
                seed,
                sizeRng,
                concurrency: o.concurrency,
                threshold: o.threshold,
                teacher: o.teacher,
                inputKeys: o.inputKeys,
                callOptions: o.callOptions,
            });
            const program = withDemosByPath(base, demos);
            const evaluation = await evaluateProgram(program, valset, o.metric, o);
            candidates.push({
                ...plan,
                score: evaluation.score,
                scores: evaluation.scores,
                state: program.dumpState(),
                program,
            });
            reportProgress(o.onProgress, {
                optimizer: 'BootstrapFewShotWithRandomSearch',
                phase: 'candidate',
                step: index + 1,
                total: plans.length,
                score: evaluation.score,
                bestScore: Math.max(...candidates.map((c) => c.score)),
                message: `${plan.kind} (seed ${plan.seed})`,
            });
            if (o.stopAtScore !== undefined && evaluation.score >= o.stopAtScore) break;
        }

        // Stable sort: on a tie the earlier (simpler) candidate wins.
        const ranked = [...candidates].sort((a, b) => b.score - a.score);
        const summary: RandomSearchCandidate[] = ranked.map((candidate) => ({
            kind: candidate.kind,
            seed: candidate.seed,
            score: candidate.score,
            scores: candidate.scores,
            state: candidate.state,
        }));
        this.results = { bestScore: ranked[0].score, candidates: summary };
        return attachResults(ranked[0].program, 'candidatePrograms', summary) as M & {
            candidatePrograms: RandomSearchCandidate[];
        };
    }
}
