/**
 * A Tree-structured Parzen Estimator (TPE) for categorical search spaces, in
 * plain TypeScript.
 *
 * MIPROv2 searches over "which instruction, which demo set" for every
 * predictor — a handful of categorical choices whose combinations are far too
 * many to try exhaustively, and each try costs a batch of model calls. TPE
 * spends those tries well: after some random exploration it splits the trials
 * so far into a good group (the top `gamma` fraction) and a bad group, builds
 * a smoothed histogram — a Parzen estimator, for categories — of each
 * parameter's values in each group, and proposes the value that maximises
 * `l(x) / g(x)`: likely among good trials, unlikely among bad ones. This is
 * the algorithm Optuna's default sampler uses, which is what DSPy's MIPROv2
 * runs on.
 *
 * Parameters are modelled independently (Optuna's default), and every random
 * draw comes from the seeded RNG, so a search is reproducible.
 */
import { createRng, type Rng } from './random';

/** Parameter name → number of categories. A parameter's values are `0 … n-1`. */
export type TPESpace = Record<string, number>;

/** One completed trial. */
export interface TPETrial {
    params: Record<string, number>;
    /** Higher is better. */
    value: number;
}

export interface TPEOptions {
    /** Seed for every draw. Default 0. */
    seed?: number;
    /**
     * Fraction of trials treated as "good". Default 0.25. At least one trial is
     * always good and, once there are two, at least one is bad.
     */
    gamma?: number;
    /**
     * Trials suggested uniformly at random before the model takes over.
     * Default 10 (Optuna's default); fewer when the space is tiny.
     */
    nStartupTrials?: number;
    /** Candidates drawn from `l(x)` per parameter, of which the best is kept. Default 24. */
    nEICandidates?: number;
    /**
     * Weight of the uniform prior mixed into each histogram, in units of
     * trials. Default 1. Keeps unseen values possible and stops a single lucky
     * trial from dominating.
     */
    priorWeight?: number;
}

/**
 * Suggests parameter assignments with TPE. Drive it with {@link suggest} and
 * {@link tell}:
 *
 * ```ts
 * const tpe = new TPESampler({ instruction: 5, demos: 4 }, { seed: 1 });
 * for (let i = 0; i < 30; i++) {
 *     const params = tpe.suggest();
 *     tpe.tell(params, await score(params));
 * }
 * tpe.best(); // { params, value }
 * ```
 */
export class TPESampler {
    private readonly space: Array<[string, number]>;
    private readonly rng: Rng;
    private readonly gamma: number;
    private readonly nStartupTrials: number;
    private readonly nEICandidates: number;
    private readonly priorWeight: number;
    private readonly history: TPETrial[] = [];

    constructor(space: TPESpace, options: TPEOptions = {}) {
        this.space = Object.entries(space);
        for (const [name, size] of this.space) {
            if (!Number.isInteger(size) || size < 1) {
                throw new RangeError(
                    `TPE parameter "${name}" needs a positive integer number of categories, got ${size}.`
                );
            }
        }
        this.rng = createRng(options.seed ?? 0);
        this.gamma = options.gamma ?? 0.25;
        const combinations = this.space.reduce((product, [, size]) => product * size, 1);
        this.nStartupTrials = Math.min(options.nStartupTrials ?? 10, combinations);
        this.nEICandidates = Math.max(1, options.nEICandidates ?? 24);
        this.priorWeight = options.priorWeight ?? 1;
    }

    /** Completed trials, in the order they were told. */
    get trials(): readonly TPETrial[] {
        return this.history;
    }

    /** The best trial so far; the earliest wins a tie. */
    best(): TPETrial | undefined {
        let best: TPETrial | undefined;
        for (const trial of this.history) if (!best || trial.value > best.value) best = trial;
        return best;
    }

    /** Record a trial's outcome. A non-finite value is recorded as the worst possible. */
    tell(params: Record<string, number>, value: number): void {
        this.history.push({
            params: { ...params },
            value: Number.isFinite(value) ? value : -Infinity,
        });
    }

    /** The next assignment to try. */
    suggest(): Record<string, number> {
        if (this.history.length < this.nStartupTrials) {
            return Object.fromEntries(
                this.space.map(([name, size]) => [name, Math.floor(this.rng() * size)])
            );
        }

        const { good, bad } = this.split();
        const params: Record<string, number> = {};
        for (const [name, size] of this.space) {
            const l = this.density(good, name, size);
            const g = this.density(bad, name, size);
            let bestValue = 0;
            let bestRatio = -Infinity;
            for (let i = 0; i < this.nEICandidates; i++) {
                const candidate = sampleCategorical(l, this.rng);
                const ratio = Math.log(l[candidate]) - Math.log(g[candidate]);
                if (ratio > bestRatio) {
                    bestRatio = ratio;
                    bestValue = candidate;
                }
            }
            params[name] = bestValue;
        }
        return params;
    }

    /** Trials split into the top `gamma` fraction and the rest. */
    private split(): { good: TPETrial[]; bad: TPETrial[] } {
        const ranked = this.history
            .map((trial, index) => ({ trial, index }))
            .sort((a, b) => b.trial.value - a.trial.value || a.index - b.index)
            .map(({ trial }) => trial);
        const n = ranked.length;
        let nGood = Math.max(1, Math.ceil(this.gamma * n));
        if (n >= 2) nGood = Math.min(nGood, n - 1);
        return { good: ranked.slice(0, nGood), bad: ranked.slice(nGood) };
    }

    /**
     * A categorical Parzen estimator: each value's share of `trials`, mixed
     * with a uniform prior worth `priorWeight` trials, so every value keeps a
     * non-zero probability.
     */
    private density(trials: TPETrial[], name: string, size: number): number[] {
        const counts = new Array<number>(size).fill(this.priorWeight / size);
        for (const trial of trials) {
            const value = trial.params[name];
            if (Number.isInteger(value) && value >= 0 && value < size) counts[value] += 1;
        }
        const total = trials.length + this.priorWeight;
        return counts.map((count) => count / total);
    }
}

/** Draw an index with probability proportional to `weights`. */
export function sampleCategorical(weights: readonly number[], rng: Rng): number {
    const total = weights.reduce((sum, w) => sum + w, 0);
    let threshold = rng() * total;
    for (let i = 0; i < weights.length; i++) {
        threshold -= weights[i];
        if (threshold < 0) return i;
    }
    return weights.length - 1;
}
