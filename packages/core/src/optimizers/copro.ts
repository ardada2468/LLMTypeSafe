import { type Module } from '../core/module';
import type { ILanguageModel } from '../types/language-model';
import { type Metric } from './types';
import {
    attachResults,
    describeProgram,
    evaluateProgram,
    predictorInstruction,
    proposeInstructions,
    reportProgress,
    type BaseOptimizerOptions,
    type CompileConfig,
} from './utils';

export interface COPROOptions extends BaseOptimizerOptions {
    /** Scores each candidate instruction. */
    metric: Metric;
    /** Writes the candidate instructions. Defaults to the configured model. */
    promptModel?: ILanguageModel;
    /** Candidates per round, per predictor. Default 10. Must be at least 2. */
    breadth?: number;
    /** Rounds per predictor. Default 3. */
    depth?: number;
    /** Proposer temperature, kept high for variety. Default 1.4, as in DSPy. */
    initTemperature?: number;
}

/** One instruction COPRO tried. */
export interface COPROAttempt {
    /** The predictor it was tried on. */
    path: string;
    /** Round, from 0. */
    depth: number;
    instruction: string;
    /** Score of the whole program with this instruction in place. */
    score: number;
}

export interface COPROResults {
    /** Score of the returned program. */
    bestScore: number;
    /** The instruction chosen per predictor path, with its score. */
    best: Record<string, { instruction: string; score: number }>;
    /** Every attempt, in the order evaluated. */
    history: COPROAttempt[];
}

/**
 * DSPy's `COPRO`: coordinate ascent over instructions.
 *
 * Each predictor in turn gets `depth` rounds. The first round scores the
 * current instruction and `breadth - 1` fresh proposals; every later round
 * shows the proposer the attempts so far *with their scores* and asks for
 * `breadth` better ones. The best instruction is locked in before moving to
 * the next predictor, so later predictors are tuned in the context of the
 * earlier ones' improvements.
 *
 * Costs `depth × breadth` evaluations per predictor (fewer when proposals
 * repeat). The student is not modified; {@link results} and the returned
 * program's non-enumerable `coproResults` hold the full history.
 */
export class COPRO {
    private readonly options: COPROOptions;
    private readonly breadth: number;
    private readonly depth: number;

    results?: COPROResults;

    constructor(options: COPROOptions) {
        if (typeof options?.metric !== 'function') {
            throw new Error('COPRO requires a metric function.');
        }
        this.breadth = options.breadth ?? 10;
        this.depth = options.depth ?? 3;
        if (!Number.isInteger(this.breadth) || this.breadth < 2) {
            throw new RangeError(
                `COPRO breadth must be an integer of at least 2, got ${this.breadth}.`
            );
        }
        if (!Number.isInteger(this.depth) || this.depth < 1) {
            throw new RangeError(`COPRO depth must be a positive integer, got ${this.depth}.`);
        }
        this.options = options;
    }

    async compile<M extends Module>(
        student: M,
        config: CompileConfig
    ): Promise<M & { coproResults: COPROResults }> {
        const o = this.options;
        const valset = config.valset ?? config.trainset;
        const working = student.deepCopy();
        const history: COPROAttempt[] = [];
        const best: COPROResults['best'] = {};
        let bestScore = 0;
        const predictors = working.namedPredictors();
        const totalSteps = predictors.length * this.depth;
        let step = 0;

        for (const [path, predictor] of predictors) {
            const original = predictorInstruction(predictor);
            const attempts: COPROAttempt[] = [];
            const tried = new Set<string>();

            for (let depth = 0; depth < this.depth; depth++) {
                const incumbent = bestOf(attempts);
                const proposals = await proposeInstructions({
                    lm: o.promptModel,
                    n: depth === 0 ? this.breadth - 1 : this.breadth,
                    basicInstruction: incumbent?.instruction ?? original,
                    programDescription: describeProgram(working),
                    // Ascending, so the best attempts sit closest to the question.
                    previous:
                        depth === 0
                            ? undefined
                            : [...attempts]
                                  .sort((a, b) => a.score - b.score)
                                  .slice(-this.breadth)
                                  .map(({ instruction, score }) => ({ instruction, score })),
                    temperature: o.initTemperature ?? 1.4,
                    callOptions: o.callOptions,
                });
                const candidates = depth === 0 ? [original, ...proposals] : proposals;

                for (const instruction of candidates) {
                    if (tried.has(instruction)) continue;
                    tried.add(instruction);
                    const candidate = working.deepCopy();
                    const target = candidate.namedPredictors().find(([p]) => p === path)![1];
                    target.setInstructions(instruction);
                    const { score } = await evaluateProgram(candidate, valset, o.metric, o);
                    const attempt = { path, depth, instruction, score };
                    attempts.push(attempt);
                    history.push(attempt);
                }

                step += 1;
                reportProgress(o.onProgress, {
                    optimizer: 'COPRO',
                    phase: 'round',
                    step,
                    total: totalSteps,
                    bestScore: bestOf(attempts)?.score,
                    message: `${path}, round ${depth + 1}/${this.depth}`,
                });
            }

            const winner = bestOf(attempts)!;
            // Leave an untouched predictor untouched: writing the signature's own
            // description back as an override would change its saved state.
            if (winner.instruction !== original || predictor.getInstructions() !== undefined) {
                predictor.setInstructions(winner.instruction);
            }
            best[path] = { instruction: winner.instruction, score: winner.score };
            bestScore = winner.score;
        }

        this.results = { bestScore, best, history };
        return attachResults(working, 'coproResults', this.results) as M & {
            coproResults: COPROResults;
        };
    }
}

/** The highest-scoring attempt; the earliest wins a tie. */
function bestOf(attempts: COPROAttempt[]): COPROAttempt | undefined {
    let best: COPROAttempt | undefined;
    for (const attempt of attempts) if (!best || attempt.score > best.score) best = attempt;
    return best;
}
