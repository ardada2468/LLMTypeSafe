import { type Module } from '../core/module';
import { invokeLM, providerOptions } from '../core/lm-call';
import { TsDspyError } from '../core/errors';
import type { ChatMessage, ILanguageModel } from '../types/language-model';
import { metricFeedback, type MetricTraceStep } from '../evaluate/types';
import { type Metric } from './types';
import { createRng, shuffled, type Rng } from './random';
import { sampleCategorical } from './tpe';
import {
    attachResults,
    evaluateProgram,
    mean,
    metricScore,
    predictorInstruction,
    reportProgress,
    runWithTrace,
    sampleMinibatch,
    type BaseOptimizerOptions,
    type CompileConfig,
    transportOptions,
} from './utils';
import { resolveExampleInputs } from './types';

/** Candidate counts behind the `auto` budgets, as in DSPy's GEPA. */
export const GEPA_AUTO_CANDIDATES = { light: 6, medium: 12, heavy: 18 } as const;

export interface GEPAOptions extends BaseOptimizerOptions {
    /**
     * Scores runs. May return `{ score, feedback }`; the feedback is what the
     * reflection model learns from. GEPA calls it with
     * `(example, prediction, trace, predName, predTrace)` while reflecting, so
     * the feedback can target the predictor being improved.
     */
    metric: Metric;
    /** Reads runs and feedback and writes new instructions. Use a strong model. */
    reflectionLM: ILanguageModel;
    /** Budget preset. Exactly one of `auto`, `maxMetricCalls`, `maxFullEvals` is required. */
    auto?: 'light' | 'medium' | 'heavy';
    /** Budget as a number of metric calls. */
    maxMetricCalls?: number;
    /** Budget as a number of full valset evaluations. */
    maxFullEvals?: number;
    /**
     * How the candidate to improve is picked. `'pareto'` (default) samples
     * from the Pareto front over per-example valset scores, weighted by how
     * many examples each candidate is best on — keeping specialists alive.
     * `'current_best'` always improves the best mean score.
     */
    candidateSelectionStrategy?: 'pareto' | 'current_best';
    /** Also try merging two frontier candidates that share an ancestor. Default true. */
    useMerge?: boolean;
    /** Most merge attempts. Default 5. */
    maxMergeInvocations?: number;
    /** Train examples per reflection. Default 3. */
    reflectionMinibatchSize?: number;
    /** Skip reflecting on a minibatch the candidate already aces. Default true. */
    skipPerfectScore?: boolean;
    /** The score that counts as perfect. Default 1. */
    perfectScore?: number;
    /** Score for a run that throws. Default 0. */
    failureScore?: number;
    /**
     * Directory for `gepa_state.json`, written after every iteration. A run
     * started with a directory that already holds a state for the same
     * program and valset resumes from it. Node only.
     */
    logDir?: string;
    /** Record a per-iteration log in the detailed results. Default false. */
    trackStats?: boolean;
}

/** One program GEPA discovered. */
export interface GEPACandidate {
    id: number;
    /** Instructions per predictor path. */
    instructions: Record<string, string>;
    /** Parent ids: none for the seed, one for a reflection, two for a merge. */
    parents: number[];
    /** How it was made. */
    origin: 'seed' | 'reflection' | 'merge';
    /** Per-example valset scores. */
    valScores: number[];
    /** Mean valset score. */
    score: number;
    /** Metric calls spent when it was found. */
    discoveredAtCalls: number;
}

/** One iteration, when `trackStats` is on. */
export interface GEPAIterationLog {
    iteration: number;
    kind: 'reflection' | 'merge' | 'skipped';
    parent?: number;
    predictor?: string;
    oldScore?: number;
    newScore?: number;
    accepted: boolean;
    metricCalls: number;
}

export interface GEPADetailedResults {
    candidates: GEPACandidate[];
    /** Mean valset score per candidate, by id. */
    scores: number[];
    bestIndex: number;
    /** Per valset example: the best score and the candidates that reach it. */
    paretoFront: Array<{ instance: number; score: number; candidates: number[] }>;
    /** Parent ids per candidate id. */
    lineage: Record<number, number[]>;
    totalMetricCalls: number;
    budget: number;
    iterations: number;
    /** Present when `trackStats` is on. */
    iterationLog?: GEPAIterationLog[];
    /** Where state was checkpointed, if anywhere. */
    logDir?: string;
}

/** What `logDir/gepa_state.json` holds. */
interface GEPAState {
    version: 1;
    paths: string[];
    valsetSize: number;
    iteration: number;
    metricCalls: number;
    rngDraws: number;
    candidates: GEPACandidate[];
    nextPredictor: Record<number, number>;
    mergesDone: number;
    mergesDue: number;
    triedMerges: string[];
    iterationLog: GEPAIterationLog[];
}

const STATE_FILE = 'gepa_state.json';

/**
 * Build the reflection prompt. Mirrors the instruction-proposal prompt of the
 * GEPA paper and library: current instruction, then examples with the
 * assistant's outputs and feedback, then the request for a new instruction in
 * a fenced block.
 */
export function buildReflectionPrompt(currentInstruction: string, examples: string): string {
    return [
        'I gave an assistant the following instructions to perform a task for me:',
        '```',
        currentInstruction,
        '```',
        '',
        "The following are examples of task inputs given to the assistant, the assistant's " +
            'response to each, and feedback on how each response could have been better:',
        '```',
        examples,
        '```',
        '',
        'Your task is to write a new instruction for the assistant.',
        '',
        'Read the inputs carefully to identify the input format, and infer a detailed ' +
            'description of the task I want the assistant to solve.',
        '',
        'Read all the assistant responses and the corresponding feedback. Identify all niche ' +
            'and domain-specific factual information about the task and include it in the ' +
            'instruction, since much of it may not be available to the assistant later. If the ' +
            'assistant used a generalizable strategy to solve the task, include that strategy ' +
            'in the instruction as well.',
        '',
        'Provide the new instruction within ``` blocks.',
    ].join('\n');
}

/** The text between the first and last ``` fences, or the whole reply without them. */
export function extractInstruction(reply: string): string {
    const start = reply.indexOf('```');
    const end = reply.lastIndexOf('```');
    if (start === -1 || end <= start) return reply.trim();
    let inner = reply.slice(start + 3, end);
    // Drop a language tag on the opening fence line, e.g. ```text.
    const newline = inner.indexOf('\n');
    if (newline !== -1 && /^[\w-]*$/.test(inner.slice(0, newline).trim())) {
        inner = inner.slice(newline + 1);
    }
    return inner.trim();
}

/** One record of a reflective dataset, rendered as GEPA does. */
interface ReflectiveRecord {
    inputs: Record<string, any>;
    outputs: Record<string, any> | string;
    feedback: string;
}

function renderRecords(records: ReflectiveRecord[]): string {
    const value = (v: unknown) => (typeof v === 'string' ? v : JSON.stringify(v, null, 2));
    return records
        .map((record, index) => {
            const lines = [`# Example ${index + 1}`, '## Inputs'];
            for (const [key, v] of Object.entries(record.inputs))
                lines.push(`### ${key}`, value(v));
            lines.push('## Generated Outputs');
            if (typeof record.outputs === 'string') lines.push(record.outputs);
            else
                for (const [key, v] of Object.entries(record.outputs))
                    lines.push(`### ${key}`, value(v));
            lines.push('## Feedback', record.feedback);
            return lines.join('\n');
        })
        .join('\n\n');
}

/** A seeded RNG that counts its draws, so a resumed run can fast-forward to the same state. */
function countingRng(seed: number, skip: number): Rng & { draws: () => number } {
    const inner = createRng(seed);
    let draws = 0;
    for (; draws < skip; draws++) inner();
    const rng = (() => {
        draws += 1;
        return inner();
    }) as Rng & { draws: () => number };
    rng.draws = () => draws;
    return rng;
}

/**
 * GEPA: reflective prompt evolution (Agrawal et al., 2025), as in DSPy.
 *
 * Starting from the student, GEPA repeatedly:
 *
 * 1. picks a candidate — from the Pareto front over per-example valset
 *    scores (sampled by how many examples each is best on), or the best mean;
 * 2. picks one of its predictors, round-robin;
 * 3. runs it on a small train minibatch, capturing traces and the metric's
 *    feedback for that predictor;
 * 4. shows the reflection model the predictor's current instruction with
 *    those inputs, outputs and feedback, and asks for a better instruction;
 * 5. keeps the new candidate if it beats its parent on the same minibatch,
 *    then scores it on the full valset and updates the front.
 *
 * With `useMerge`, after a success it may also combine two frontier
 * candidates that descend from a common ancestor, taking each predictor's
 * instruction from whichever side changed it. The run stops when the metric
 * call budget is spent.
 *
 * Returns the candidate with the best mean valset score, carrying
 * `detailedResults` (every candidate, scores, the Pareto front, lineage) as a
 * non-enumerable property; {@link detailedResults} holds the same. The student
 * is not modified. With `logDir`, state is checkpointed every iteration and a
 * later run resumes from it.
 */
export class GEPA {
    private readonly options: GEPAOptions;

    detailedResults?: GEPADetailedResults;

    constructor(options: GEPAOptions) {
        if (typeof options?.metric !== 'function') {
            throw new Error('GEPA requires a metric function.');
        }
        if (!options.reflectionLM) {
            throw new Error('GEPA requires a reflectionLM.');
        }
        const budgets = [options.auto, options.maxMetricCalls, options.maxFullEvals].filter(
            (b) => b !== undefined
        );
        if (budgets.length !== 1) {
            throw new Error(
                'GEPA needs exactly one budget: auto, maxMetricCalls or maxFullEvals.'
            );
        }
        if (options.auto !== undefined && !(options.auto in GEPA_AUTO_CANDIDATES)) {
            throw new Error(
                `GEPA auto must be 'light', 'medium' or 'heavy', got ${String(options.auto)}.`
            );
        }
        this.options = options;
    }

    /**
     * The metric-call budget for a run: as given, `maxFullEvals` valset
     * passes, or for `auto` enough to discover its candidate count — a full
     * valset pass per candidate plus the seed, and room for four two-minibatch
     * reflection attempts per candidate per predictor.
     */
    budgetFor(valsetSize: number, numPredictors: number): number {
        const o = this.options;
        if (o.maxMetricCalls !== undefined) return o.maxMetricCalls;
        if (o.maxFullEvals !== undefined) return o.maxFullEvals * valsetSize;
        const n = GEPA_AUTO_CANDIDATES[o.auto!];
        const mb = o.reflectionMinibatchSize ?? 3;
        return (n + 1) * valsetSize + n * Math.max(1, numPredictors) * 8 * mb;
    }

    async compile<M extends Module>(
        student: M,
        config: CompileConfig
    ): Promise<M & { detailedResults: GEPADetailedResults }> {
        const o = this.options;
        const trainset = config.trainset;
        const valset = config.valset ?? trainset;
        if (valset.length === 0)
            throw new Error('GEPA needs a non-empty valset (or trainset).');
        // Reflection minibatches come from the trainset. With none, no
        // iteration would spend any budget and the loop would never end.
        if (trainset.length === 0) throw new Error('GEPA needs a non-empty trainset.');
        const base = student.deepCopy();
        const paths = base.namedPredictors().map(([path]) => path);
        if (paths.length === 0) throw new Error('GEPA found no predictors to optimize.');
        const original = Object.fromEntries(
            base
                .namedPredictors()
                .map(([path, predictor]) => [path, predictorInstruction(predictor)])
        );
        const budget = this.budgetFor(valset.length, paths.length);
        const mb = Math.max(1, o.reflectionMinibatchSize ?? 3);
        const perfect = o.perfectScore ?? 1;
        const maxMerges = o.useMerge === false ? 0 : (o.maxMergeInvocations ?? 5);

        const build = (instructions: Record<string, string>): M => {
            const program = base.deepCopy();
            for (const [path, predictor] of program.namedPredictors()) {
                if (instructions[path] !== original[path]) {
                    predictor.setInstructions(instructions[path]);
                }
            }
            return program;
        };

        let state = await this.loadState(paths, valset.length);
        // A checkpoint is only this run's if it started from this program:
        // otherwise a re-run after editing the student would silently return
        // the old run's instructions.
        const seedInstructions = state?.candidates?.[0]?.instructions;
        if (seedInstructions && JSON.stringify(seedInstructions) !== JSON.stringify(original)) {
            throw new TsDspyError(
                `GEPA state in ${o.logDir} was written for a program with different ` +
                    'instructions. Use a fresh logDir.'
            );
        }
        const resumed = state !== undefined;
        const rng = countingRng(o.seed ?? 0, state?.rngDraws ?? 0);
        state ??= {
            version: 1,
            paths,
            valsetSize: valset.length,
            iteration: 0,
            metricCalls: 0,
            rngDraws: 0,
            candidates: [],
            nextPredictor: {},
            mergesDone: 0,
            mergesDue: 0,
            triedMerges: [],
            iterationLog: [],
        };
        const s = state;
        const progress = (phase: string, extra: Record<string, unknown> = {}) =>
            reportProgress(o.onProgress, {
                optimizer: 'GEPA',
                phase,
                step: s.iteration,
                bestScore: Math.max(...s.candidates.map((c) => c.score), 0),
                ...extra,
            });
        const log = (entry: Omit<GEPAIterationLog, 'iteration' | 'metricCalls'>) => {
            if (o.trackStats) {
                s.iterationLog.push({
                    iteration: s.iteration,
                    metricCalls: s.metricCalls,
                    ...entry,
                });
            }
        };
        const evalVal = async (instructions: Record<string, string>) => {
            const evaluation = await evaluateProgram(build(instructions), valset, o.metric, o);
            s.metricCalls += valset.length;
            return evaluation.scores;
        };
        const addCandidate = (
            instructions: Record<string, string>,
            parents: number[],
            origin: GEPACandidate['origin'],
            valScores: number[]
        ): GEPACandidate => {
            const candidate: GEPACandidate = {
                id: s.candidates.length,
                instructions,
                parents,
                origin,
                valScores,
                score: mean(valScores),
                discoveredAtCalls: s.metricCalls,
            };
            s.candidates.push(candidate);
            return candidate;
        };

        if (!resumed) {
            addCandidate({ ...original }, [], 'seed', await evalVal(original));
            progress('seed', { score: s.candidates[0].score });
            await this.saveState(s, rng);
        }

        while (s.metricCalls < budget) {
            s.iteration += 1;

            // Merge, when one is due and there are candidates to merge.
            if (s.mergesDue > 0 && s.mergesDone < maxMerges) {
                s.mergesDue -= 1;
                const merged = await this.tryMerge(
                    s,
                    valset.length,
                    rng,
                    evalVal,
                    addCandidate
                );
                if (merged !== undefined) {
                    s.mergesDone += 1;
                    log({ kind: 'merge', accepted: merged });
                    progress('merge', { message: merged ? 'accepted' : 'rejected' });
                    await this.saveState(s, rng);
                    continue;
                }
            }

            const parent = this.selectCandidate(s, rng);
            const next = s.nextPredictor[parent.id] ?? 0;
            s.nextPredictor[parent.id] = next + 1;
            const path = paths[next % paths.length];
            const batch = sampleMinibatch(trainset, mb, rng);

            // Run the parent on the minibatch, capturing traces and feedback.
            const program = build(parent.instructions);
            const records: ReflectiveRecord[] = [];
            const oldScores: number[] = [];
            for (const [index, example] of batch.entries()) {
                const inputs = resolveExampleInputs(example, o.inputKeys, index);
                const run = await runWithTrace(program, inputs, o.callOptions);
                const predTrace = run.trace.filter((step) => step.path === path);
                let score = o.failureScore ?? 0;
                let feedback: string | undefined;
                if (run.prediction) {
                    try {
                        // Score as every other evaluation does — without a
                        // predictor name — so the parent's minibatch score is
                        // comparable with the child's, which `evaluateProgram`
                        // measures that way. Then ask again, naming the
                        // predictor, for feedback aimed at it (as DSPy does: a
                        // metric may grade differently when it sees `predName`).
                        score = metricScore(await o.metric(example, run.prediction, run.trace));
                        feedback = metricFeedback(
                            await o.metric(example, run.prediction, run.trace, path, predTrace)
                        );
                    } catch (error) {
                        feedback = `The metric failed on this output: ${describeError(error)}`;
                    }
                } else {
                    feedback = `The program failed with an error: ${describeError(run.error)}`;
                }
                // The metric ran twice for a prediction: once to score, once
                // for feedback. With a judge model behind it, both are spend.
                s.metricCalls += run.prediction ? 2 : 1;
                oldScores.push(score);
                const step: MetricTraceStep | undefined = predTrace.at(-1);
                if (step) {
                    records.push({
                        inputs: step.inputs,
                        outputs: step.outputs,
                        feedback: feedback ?? `This trajectory got a score of ${score}.`,
                    });
                } else if (!run.prediction) {
                    records.push({
                        inputs,
                        outputs: `Error: ${describeError(run.error)}`,
                        feedback: feedback ?? `This trajectory got a score of ${score}.`,
                    });
                }
            }

            if (o.skipPerfectScore !== false && oldScores.every((score) => score >= perfect)) {
                log({ kind: 'skipped', parent: parent.id, predictor: path, accepted: false });
                progress('iteration', { message: `skipped: ${path} already perfect` });
                await this.saveState(s, rng);
                continue;
            }
            if (records.length === 0) {
                log({ kind: 'skipped', parent: parent.id, predictor: path, accepted: false });
                await this.saveState(s, rng);
                continue;
            }

            const proposal = await this.reflect(parent.instructions[path], records);
            if (!proposal || proposal === parent.instructions[path]) {
                log({ kind: 'skipped', parent: parent.id, predictor: path, accepted: false });
                await this.saveState(s, rng);
                continue;
            }

            const instructions = { ...parent.instructions, [path]: proposal };
            const newScores = (await evaluateProgram(build(instructions), batch, o.metric, o))
                .scores;
            s.metricCalls += batch.length;
            const oldSum = oldScores.reduce((a, b) => a + b, 0);
            const newSum = newScores.reduce((a, b) => a + b, 0);
            const accepted = newSum > oldSum;
            log({
                kind: 'reflection',
                parent: parent.id,
                predictor: path,
                oldScore: oldSum / batch.length,
                newScore: newSum / batch.length,
                accepted,
            });

            if (accepted) {
                const candidate = addCandidate(
                    instructions,
                    [parent.id],
                    'reflection',
                    await evalVal(instructions)
                );
                if (s.mergesDone < maxMerges) s.mergesDue += 1;
                progress('accept', {
                    score: candidate.score,
                    message: `candidate ${candidate.id} from ${parent.id} via ${path}`,
                });
            } else {
                progress('iteration', { message: `rejected proposal for ${path}` });
            }
            await this.saveState(s, rng);
        }

        const bestIndex = s.candidates.reduce(
            (best, candidate, index) =>
                candidate.score > s.candidates[best].score ? index : best,
            0
        );
        const fronts = paretoFronts(s.candidates, valset.length);
        this.detailedResults = {
            candidates: s.candidates,
            scores: s.candidates.map((c) => c.score),
            bestIndex,
            paretoFront: fronts.map((front, instance) => ({
                instance,
                score: front.score,
                candidates: [...front.ids],
            })),
            lineage: Object.fromEntries(s.candidates.map((c) => [c.id, [...c.parents]])),
            totalMetricCalls: s.metricCalls,
            budget,
            iterations: s.iteration,
            ...(o.trackStats ? { iterationLog: s.iterationLog } : {}),
            ...(o.logDir ? { logDir: o.logDir } : {}),
        };
        return attachResults(
            build(s.candidates[bestIndex].instructions),
            'detailedResults',
            this.detailedResults
        ) as M & { detailedResults: GEPADetailedResults };
    }

    /** Ask the reflection model for a better instruction. `undefined` on failure. */
    private async reflect(
        current: string,
        records: ReflectiveRecord[]
    ): Promise<string | undefined> {
        const lm = this.options.reflectionLM;
        const prompt = buildReflectionPrompt(current, renderRecords(records));
        const messages: ChatMessage[] = [{ role: 'user', content: prompt }];
        const options = providerOptions(transportOptions(this.options.callOptions));
        try {
            const reply = await invokeLM(
                { lm, method: 'chat', traceText: prompt, messages, options },
                () => lm.chat(messages, options),
                (text) => text
            );
            const instruction = extractInstruction(reply);
            return instruction || undefined;
        } catch {
            return undefined;
        }
    }

    /** The candidate to improve, by the configured strategy. */
    private selectCandidate(s: GEPAState, rng: Rng): GEPACandidate {
        if (this.options.candidateSelectionStrategy === 'current_best') {
            return s.candidates.reduce((best, c) => (c.score > best.score ? c : best));
        }
        const fronts = paretoFronts(s.candidates, s.valsetSize);
        const survivors = nonDominated(s.candidates, fronts);
        const wins = s.candidates.map((c) =>
            survivors.has(c.id) ? fronts.filter((front) => front.ids.has(c.id)).length : 0
        );
        if (wins.every((w) => w === 0)) return s.candidates[0];
        return s.candidates[sampleCategorical(wins, rng)];
    }

    /**
     * Try to merge two frontier candidates that share an ancestor. Returns
     * `undefined` when no merge is possible, otherwise whether it was accepted.
     */
    private async tryMerge(
        s: GEPAState,
        valsetSize: number,
        rng: Rng,
        evalVal: (instructions: Record<string, string>) => Promise<number[]>,
        addCandidate: (
            instructions: Record<string, string>,
            parents: number[],
            origin: GEPACandidate['origin'],
            valScores: number[]
        ) => GEPACandidate
    ): Promise<boolean | undefined> {
        const fronts = paretoFronts(s.candidates, valsetSize);
        const frontier = [...nonDominated(s.candidates, fronts)].sort((a, b) => a - b);
        const ancestors = (id: number): Set<number> => {
            const seen = new Set<number>();
            const stack = [...s.candidates[id].parents];
            while (stack.length > 0) {
                const next = stack.pop()!;
                if (seen.has(next)) continue;
                seen.add(next);
                stack.push(...s.candidates[next].parents);
            }
            return seen;
        };

        const pairs: Array<[number, number]> = [];
        for (const i of frontier) for (const j of frontier) if (i < j) pairs.push([i, j]);
        const existing = new Set(s.candidates.map((c) => JSON.stringify(c.instructions)));

        for (const [i, j] of shuffled(pairs, rng)) {
            const ai = ancestors(i);
            const aj = ancestors(j);
            if (ai.has(j) || aj.has(i)) continue;
            const common = [...ai].filter((a) => aj.has(a)).sort((a, b) => a - b);
            for (const a of common) {
                const key = `${i}+${j}^${a}`;
                if (s.triedMerges.includes(key)) continue;
                const ci = s.candidates[i];
                const cj = s.candidates[j];
                const ca = s.candidates[a];
                let useful = false;
                const merged: Record<string, string> = {};
                for (const path of s.paths) {
                    const fromI = ci.instructions[path];
                    const fromJ = cj.instructions[path];
                    const fromA = ca.instructions[path];
                    if (fromI === fromA && fromJ !== fromA) {
                        merged[path] = fromJ;
                        useful = true;
                    } else if (fromJ === fromA && fromI !== fromA) {
                        merged[path] = fromI;
                        useful = true;
                    } else if (fromI !== fromA && fromJ !== fromA) {
                        merged[path] = ci.score >= cj.score ? fromI : fromJ;
                    } else {
                        merged[path] = fromA;
                    }
                }
                if (!useful || existing.has(JSON.stringify(merged))) continue;
                s.triedMerges.push(key);

                const full = await evalVal(merged);
                // Accept when the merge does at least as well as both parents.
                const accepted = mean(full) >= Math.max(ci.score, cj.score);
                if (accepted) addCandidate(merged, [i, j], 'merge', full);
                return accepted;
            }
        }
        return undefined;
    }

    private async loadState(
        paths: string[],
        valsetSize: number
    ): Promise<GEPAState | undefined> {
        const dir = this.options.logDir;
        if (!dir) return undefined;
        const { readFile } = await import('node:fs/promises');
        const { join } = await import('node:path');
        let text: string;
        try {
            text = await readFile(join(dir, STATE_FILE), 'utf8');
        } catch {
            return undefined;
        }
        let state: GEPAState;
        try {
            state = JSON.parse(text) as GEPAState;
        } catch (cause) {
            throw new TsDspyError(`GEPA state in ${dir} is not valid JSON.`, { cause });
        }
        if (
            state.version !== 1 ||
            JSON.stringify(state.paths) !== JSON.stringify(paths) ||
            state.valsetSize !== valsetSize
        ) {
            throw new TsDspyError(
                `GEPA state in ${dir} was written for a different program or valset ` +
                    `(predictors ${JSON.stringify(state.paths)}, valset size ${state.valsetSize}). ` +
                    'Use a fresh logDir.'
            );
        }
        return state;
    }

    private async saveState(
        state: GEPAState,
        rng: Rng & { draws: () => number }
    ): Promise<void> {
        const dir = this.options.logDir;
        if (!dir) return;
        state.rngDraws = rng.draws();
        const { mkdir, writeFile, rename } = await import('node:fs/promises');
        const { join } = await import('node:path');
        await mkdir(dir, { recursive: true });
        // Write then rename, so an interrupted write never leaves a torn state file.
        const target = join(dir, STATE_FILE);
        await writeFile(`${target}.tmp`, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
        await rename(`${target}.tmp`, target);
    }
}

/** Per valset example: the best score and the candidates that reach it. */
function paretoFronts(
    candidates: GEPACandidate[],
    valsetSize: number
): Array<{ score: number; ids: Set<number> }> {
    return Array.from({ length: valsetSize }, (_, instance) => {
        let score = -Infinity;
        let ids = new Set<number>();
        for (const candidate of candidates) {
            const value = candidate.valScores[instance] ?? -Infinity;
            if (value > score) {
                score = value;
                ids = new Set([candidate.id]);
            } else if (value === score) {
                ids.add(candidate.id);
            }
        }
        return { score, ids };
    });
}

/**
 * Frontier candidates that are not dominated, as in GEPA: going from the
 * lowest mean score up, a candidate is dropped when every front it belongs to
 * also holds another surviving candidate.
 */
function nonDominated(
    candidates: GEPACandidate[],
    fronts: Array<{ ids: Set<number> }>
): Set<number> {
    const onFront = new Set<number>();
    for (const front of fronts) for (const id of front.ids) onFront.add(id);
    const order = [...onFront].sort(
        (a, b) => candidates[a].score - candidates[b].score || a - b
    );
    const dominated = new Set<number>();
    let changed = true;
    while (changed) {
        changed = false;
        for (const id of order) {
            if (dominated.has(id)) continue;
            const mine = fronts.filter((front) => front.ids.has(id));
            const covered = mine.every((front) =>
                [...front.ids].some((other) => other !== id && !dominated.has(other))
            );
            if (covered) {
                dominated.add(id);
                changed = true;
                break;
            }
        }
    }
    return new Set(order.filter((id) => !dominated.has(id)));
}

function describeError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
