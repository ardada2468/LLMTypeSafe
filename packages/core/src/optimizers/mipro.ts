import { type Example } from '../core/example';
import { type Module } from '../core/module';
import { context, findDefaultLM } from '../core/config';
import type { ILanguageModel } from '../types/language-model';
import { type Metric } from './types';
import { createRng } from './random';
import { TPESampler } from './tpe';
import {
    attachResults,
    buildDemoSet,
    demoSetPlans,
    describeProgram,
    evaluateProgram,
    formatExamples,
    mean,
    predictorInstruction,
    proposeInstructions,
    randomInt,
    reportProgress,
    sampleMinibatch,
    summarizeDataset,
    type BaseOptimizerOptions,
    type CompileConfig,
    withoutDemos,
} from './utils';

/** Budget presets, as in DSPy: instruction/demo candidates and validation-set cap. */
export const MIPRO_AUTO_PRESETS = {
    light: { numCandidates: 6, valsetSize: 100 },
    medium: { numCandidates: 12, valsetSize: 300 },
    heavy: { numCandidates: 18, valsetSize: 1000 },
} as const;

/**
 * Tips shown to the proposer, one drawn at random per proposal, so candidate
 * instructions differ in style as well as content — DSPy's tip pool, reworded.
 */
export const MIPRO_TIPS: readonly string[] = [
    'Write the instruction in your own words; be creative.',
    'Keep the instruction short and simple.',
    'Make the instruction descriptive: spell out exactly what a good output looks like.',
    'Frame the task as high-stakes, where a wrong answer has real consequences.',
    'Open the instruction by giving the model a fitting persona or role.',
    'Write whatever instruction you believe works best.',
];

export interface MIPROv2Options extends BaseOptimizerOptions {
    /** Scores trials, and judges bootstrap attempts. */
    metric: Metric;
    /** Writes the dataset summary and the candidate instructions. Defaults to the configured model. */
    promptModel?: ILanguageModel;
    /**
     * Runs the program during the search, as the default model for every
     * predictor without its own. The returned program is not pinned to it.
     */
    taskModel?: ILanguageModel;
    /**
     * Budget preset. Default `'light'`. `null` uses `numCandidates` and
     * `numTrials` exactly as given (both required then).
     */
    auto?: 'light' | 'medium' | 'heavy' | null;
    /** Instruction and demo-set candidates per predictor. Set by `auto`. */
    numCandidates?: number;
    /** Search trials. Derived from the candidate count when `auto` is set. */
    numTrials?: number;
    /** Score trials on a minibatch of the valset, as DSPy does. Default true. */
    minibatch?: boolean;
    /** Examples per minibatch. Default 35. */
    minibatchSize?: number;
    /** Fully evaluate the most promising configuration every this many trials. Default 5. */
    minibatchFullEvalSteps?: number;
    /** Cap on bootstrapped demos per predictor. Default 4. `0` with no labelled demos searches instructions only. */
    maxBootstrappedDemos?: number;
    /** Cap on labelled demos per predictor. Default 4. */
    maxLabeledDemos?: number;
    /** Score at which a numeric metric passes during bootstrapping. Default 0.5. */
    threshold?: number;
    /** Model that generates the bootstrapped traces. */
    teacher?: ILanguageModel;
    /** Ground proposals in an LM-written dataset summary. Default true. */
    dataAwareProposer?: boolean;
    /** Ground proposals in a description of the program's structure. Default true. */
    programAwareProposer?: boolean;
    /** Show the proposer a random tip. Default true. */
    tipAwareProposer?: boolean;
    /** Show the proposer demos from a candidate demo set. Default true. */
    fewshotAwareProposer?: boolean;
    /** Proposer temperature. Default 1. */
    initTemperature?: number;
}

/** One search trial. */
export interface MIPROTrial {
    /** 0 is the unmodified program. */
    trial: number;
    /** Instruction and demo-set index per predictor path. */
    params: Record<string, { instruction: number; demos: number }>;
    score: number;
    /** Whether `score` is a minibatch score (true) or a full-valset score. */
    minibatch: boolean;
}

export interface MIPROResults {
    bestScore: number;
    /** The winning configuration per predictor path. */
    best: Record<
        string,
        { instruction: string; instructionIndex: number; demoSetIndex: number }
    >;
    /** The instruction candidates per predictor path; index 0 is the original. */
    instructionCandidates: Record<string, string[]>;
    /** The demo-set candidates, by plan. */
    demoSetCandidates: number;
    /** The LM-written dataset summary, when one was made. */
    datasetSummary?: string;
    numTrials: number;
    numCandidates: number;
    trials: MIPROTrial[];
    /** Full-valset evaluations, in order: the configurations that were promoted. */
    fullEvaluations: Array<{ trial: number; params: MIPROTrial['params']; score: number }>;
}

type Config = Record<string, { instruction: number; demos: number }>;

/**
 * DSPy's `MIPROv2`: jointly optimise every predictor's instruction and demos.
 *
 * 1. **Bootstrap** `numCandidates` demo sets per predictor (zero-shot,
 *    labelled, unshuffled and shuffled bootstraps), from traced runs.
 * 2. **Propose** `numCandidates - 1` instructions per predictor with the
 *    prompt model, grounded in an LM-written summary of the trainset, a
 *    description of the program's structure, demos from a candidate set, and
 *    a random tip. The current instruction is candidate 0.
 * 3. **Search** with a TPE sampler over (instruction, demo set) per predictor.
 *    Each trial is scored on a seeded minibatch of the valset; every
 *    `minibatchFullEvalSteps` trials, and at the end, the configuration with
 *    the best mean minibatch score is scored on the full valset. The best
 *    full-valset score wins.
 *
 * The student is not modified. {@link results} — and the returned program's
 * non-enumerable `miproResults` — hold every trial, the candidates, and the
 * winning configuration.
 */
export class MIPROv2 {
    private readonly options: MIPROv2Options;

    results?: MIPROResults;

    constructor(options: MIPROv2Options) {
        if (typeof options?.metric !== 'function') {
            throw new Error('MIPROv2 requires a metric function.');
        }
        const auto = options.auto === undefined ? 'light' : options.auto;
        if (
            auto === null &&
            (options.numCandidates === undefined || options.numTrials === undefined)
        ) {
            throw new Error('MIPROv2 with auto: null needs both numCandidates and numTrials.');
        }
        if (auto !== null && !(auto in MIPRO_AUTO_PRESETS)) {
            throw new Error(
                `MIPROv2 auto must be 'light', 'medium', 'heavy' or null, got ${String(auto)}.`
            );
        }
        this.options = { ...options, auto };
    }

    async compile<M extends Module>(
        student: M,
        config: CompileConfig
    ): Promise<M & { miproResults: MIPROResults }> {
        const o = this.options;
        // Resolve the prompt model *before* the task model becomes the scoped
        // default: like DSPy, an unset prompt model means the configured model,
        // not the task model the search happens to run under.
        const promptModel = o.promptModel ?? (o.taskModel ? findDefaultLM() : undefined);
        const run = <T>(fn: () => Promise<T>) =>
            o.taskModel ? context({ lm: o.taskModel }, fn) : fn();
        return run(() => this.search(student, config, promptModel));
    }

    private async search<M extends Module>(
        student: M,
        config: CompileConfig,
        promptModel: ILanguageModel | undefined
    ): Promise<M & { miproResults: MIPROResults }> {
        const o = this.options;
        const seed = o.seed ?? 0;
        const rng = createRng(seed);
        const preset = o.auto ? MIPRO_AUTO_PRESETS[o.auto] : undefined;
        const numCandidates = Math.max(1, o.numCandidates ?? preset!.numCandidates);
        const base = student.deepCopy();
        const predictors = base.namedPredictors();
        const paths = predictors.map(([path]) => path);
        const maxBoot = o.maxBootstrappedDemos ?? 4;
        const maxLabeled = o.maxLabeledDemos ?? 4;
        const searchDemos = maxBoot > 0 || maxLabeled > 0;

        let valset = config.valset ?? config.trainset;
        if (preset && valset.length > preset.valsetSize) {
            valset = sampleMinibatch(valset, preset.valsetSize, rng);
        }
        const numTrials =
            o.numTrials ?? trialsFor(numCandidates, predictors.length, searchDemos);
        const progress = (phase: string, extra: Record<string, unknown> = {}) =>
            reportProgress(o.onProgress, { optimizer: 'MIPROv2', phase, ...extra });

        // (a) Demo-set candidates, per predictor path.
        const demoSets: Array<Record<string, Example[]>> = [];
        if (searchDemos) {
            const sizeRng = createRng(seed + 1);
            const plans = demoSetPlans(numCandidates, seed);
            for (const [index, plan] of plans.entries()) {
                demoSets.push(
                    await buildDemoSet(withoutDemos(base), config.trainset, o.metric, plan, {
                        maxBootstrappedDemos: maxBoot,
                        maxLabeledDemos: maxLabeled,
                        seed,
                        sizeRng,
                        concurrency: o.concurrency,
                        threshold: o.threshold,
                        teacher: o.teacher,
                        inputKeys: o.inputKeys,
                        callOptions: o.callOptions,
                    })
                );
                progress('bootstrap', { step: index + 1, total: plans.length });
            }
        }

        // (b) Grounded instruction proposals, per predictor path.
        const datasetSummary =
            o.dataAwareProposer === false
                ? undefined
                : await summarizeDataset(config.trainset, {
                      lm: promptModel,
                      rng: createRng(seed + 2),
                  });
        const programDescription =
            o.programAwareProposer === false ? undefined : describeProgram(base);
        const instructionCandidates: Record<string, string[]> = {};
        for (const [path, predictor] of predictors) {
            const original = predictorInstruction(predictor);
            const candidates = [original];
            for (let i = 1; i < numCandidates; i++) {
                const demoSet = demoSets.length > 0 ? demoSets[i % demoSets.length][path] : [];
                const tip =
                    o.tipAwareProposer === false
                        ? undefined
                        : MIPRO_TIPS[randomInt(rng, MIPRO_TIPS.length)];
                const [proposal] = await proposeInstructions({
                    lm: promptModel,
                    n: 1,
                    index: i - 1,
                    total: numCandidates - 1,
                    basicInstruction: original,
                    programDescription,
                    datasetSummary,
                    demos:
                        o.fewshotAwareProposer === false || !demoSet || demoSet.length === 0
                            ? undefined
                            : formatExamples(demoSet),
                    tips: tip ? [tip] : undefined,
                    temperature: o.initTemperature ?? 1,
                    callOptions: o.callOptions,
                });
                if (proposal && !candidates.includes(proposal)) candidates.push(proposal);
            }
            instructionCandidates[path] = candidates;
            progress('propose', { message: `${path}: ${candidates.length} instruction(s)` });
        }

        // (c) Bayesian search over (instruction, demo set) per predictor.
        const space: Record<string, number> = {};
        for (const path of paths) {
            space[`${path}:instruction`] = instructionCandidates[path].length;
            space[`${path}:demos`] = Math.max(1, demoSets.length);
        }
        const tpe = new TPESampler(space, { seed });
        const toConfig = (params: Record<string, number>): Config =>
            Object.fromEntries(
                paths.map((path) => [
                    path,
                    {
                        instruction: params[`${path}:instruction`],
                        demos: params[`${path}:demos`],
                    },
                ])
            );
        const toParams = (cfg: Config): Record<string, number> =>
            Object.fromEntries(
                paths.flatMap((path) => [
                    [`${path}:instruction`, cfg[path].instruction],
                    [`${path}:demos`, cfg[path].demos],
                ])
            );
        const build = (cfg: Config): M => {
            const program = base.deepCopy();
            for (const [path, predictor] of program.namedPredictors()) {
                const choice = cfg[path];
                if (choice.instruction > 0) {
                    predictor.setInstructions(instructionCandidates[path][choice.instruction]);
                }
                if (demoSets.length > 0) predictor.setDemos(demoSets[choice.demos][path] ?? []);
            }
            return program;
        };
        const fullEval = (program: M) => evaluateProgram(program, valset, o.metric, o);

        const trials: MIPROTrial[] = [];
        const fullEvaluations: MIPROResults['fullEvaluations'] = [];
        const batchScores = new Map<string, { cfg: Config; scores: number[] }>();
        const fullyEvaluated = new Map<string, number>();
        const useMinibatch = (o.minibatch ?? true) && valset.length > (o.minibatchSize ?? 35);
        const minibatchSize = o.minibatchSize ?? 35;
        const fullEvalSteps = Math.max(1, o.minibatchFullEvalSteps ?? 5);

        let bestCfg: Config = toConfig(
            Object.fromEntries(Object.keys(space).map((k) => [k, 0]))
        );
        const baseline = await fullEval(build(bestCfg));
        let bestScore = baseline.score;
        fullyEvaluated.set(JSON.stringify(bestCfg), bestScore);
        tpe.tell(toParams(bestCfg), bestScore);
        trials.push({ trial: 0, params: bestCfg, score: bestScore, minibatch: false });
        fullEvaluations.push({ trial: 0, params: bestCfg, score: bestScore });
        progress('trial', { step: 0, total: numTrials, score: bestScore, bestScore });

        const promote = async (trial: number) => {
            let candidate: { key: string; cfg: Config; mean: number } | undefined;
            for (const [key, entry] of batchScores) {
                if (fullyEvaluated.has(key)) continue;
                const avg = mean(entry.scores);
                if (!candidate || avg > candidate.mean)
                    candidate = { key, cfg: entry.cfg, mean: avg };
            }
            if (!candidate) return;
            const { score } = await fullEval(build(candidate.cfg));
            fullyEvaluated.set(candidate.key, score);
            fullEvaluations.push({ trial, params: candidate.cfg, score });
            if (score > bestScore) {
                bestScore = score;
                bestCfg = candidate.cfg;
            }
            progress('evaluate', { step: trial, total: numTrials, score, bestScore });
        };

        for (let trial = 1; trial <= numTrials; trial++) {
            const params = tpe.suggest();
            const cfg = toConfig(params);
            const key = JSON.stringify(cfg);
            const program = build(cfg);
            let score: number;
            if (useMinibatch) {
                const batch = sampleMinibatch(valset, minibatchSize, rng);
                score = (await evaluateProgram(program, batch, o.metric, o)).score;
                const entry = batchScores.get(key) ?? { cfg, scores: [] };
                entry.scores.push(score);
                batchScores.set(key, entry);
            } else {
                score = fullyEvaluated.get(key) ?? (await fullEval(program)).score;
                if (!fullyEvaluated.has(key)) {
                    fullyEvaluated.set(key, score);
                    fullEvaluations.push({ trial, params: cfg, score });
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestCfg = cfg;
                }
            }
            tpe.tell(params, score);
            trials.push({ trial, params: cfg, score, minibatch: useMinibatch });
            progress('trial', { step: trial, total: numTrials, score, bestScore });

            if (useMinibatch && (trial % fullEvalSteps === 0 || trial === numTrials)) {
                await promote(trial);
            }
        }

        const best = build(bestCfg);
        this.results = {
            bestScore,
            best: Object.fromEntries(
                paths.map((path) => [
                    path,
                    {
                        instruction: instructionCandidates[path][bestCfg[path].instruction],
                        instructionIndex: bestCfg[path].instruction,
                        demoSetIndex: bestCfg[path].demos,
                    },
                ])
            ),
            instructionCandidates,
            demoSetCandidates: demoSets.length,
            ...(datasetSummary !== undefined ? { datasetSummary } : {}),
            numTrials,
            numCandidates,
            trials,
            fullEvaluations,
        };
        return attachResults(best, 'miproResults', this.results) as M & {
            miproResults: MIPROResults;
        };
    }
}

/**
 * DSPy's trial count for a candidate budget: enough for TPE to see each
 * variable's options a few times — `max(2 · vars · log2(n), 1.5 · n)`, where
 * each predictor contributes one variable for instructions and one for demos.
 */
export function trialsFor(
    numCandidates: number,
    numPredictors: number,
    demos: boolean
): number {
    const vars = numPredictors * (demos ? 2 : 1);
    return Math.max(
        1,
        Math.floor(
            Math.max(2 * vars * Math.log2(Math.max(2, numCandidates)), 1.5 * numCandidates)
        )
    );
}
