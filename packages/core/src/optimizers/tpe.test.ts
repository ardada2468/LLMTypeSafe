import { describe, expect, it } from 'vitest';
import { TPESampler, sampleCategorical } from './tpe';
import { createRng } from './random';

const target = { a: 2, b: 4, c: 1 };
/** Matches against a hidden optimum: 3 at the optimum, 0 when nothing matches. */
const objective = (params: Record<string, number>) =>
    Number(params.a === target.a) +
    Number(params.b === target.b) +
    Number(params.c === target.c);

function run(seed: number, trials: number) {
    const tpe = new TPESampler({ a: 5, b: 5, c: 5 }, { seed });
    const suggestions: Array<Record<string, number>> = [];
    for (let i = 0; i < trials; i++) {
        const params = tpe.suggest();
        suggestions.push(params);
        tpe.tell(params, objective(params));
    }
    return { tpe, suggestions };
}

describe('TPESampler', () => {
    it('converges on the optimum of a synthetic objective', () => {
        const { tpe, suggestions } = run(1, 60);

        expect(tpe.best()?.value).toBe(3);
        expect(tpe.best()?.params).toEqual(target);

        // Once the model takes over it keeps proposing the optimum. Uniform
        // sampling would hit it with probability 1/125 per trial.
        const late = suggestions.slice(-20).filter((params) => objective(params) === 3);
        expect(late.length).toBeGreaterThanOrEqual(10);
    });

    it('beats random search on the same budget across seeds', () => {
        let tpeWins = 0;
        for (let seed = 0; seed < 10; seed++) {
            const { tpe } = run(seed, 40);
            const rng = createRng(seed + 1000);
            let randomBest = 0;
            for (let i = 0; i < 40; i++) {
                const params = {
                    a: Math.floor(rng() * 5),
                    b: Math.floor(rng() * 5),
                    c: Math.floor(rng() * 5),
                };
                randomBest = Math.max(randomBest, objective(params));
            }
            if ((tpe.best()?.value ?? 0) >= randomBest) tpeWins += 1;
        }
        expect(tpeWins).toBeGreaterThanOrEqual(8);
    });

    it('explores uniformly for the startup trials and stays in range', () => {
        const tpe = new TPESampler({ x: 3, y: 2 }, { seed: 4, nStartupTrials: 5 });
        for (let i = 0; i < 30; i++) {
            const params = tpe.suggest();
            expect(params.x).toBeGreaterThanOrEqual(0);
            expect(params.x).toBeLessThan(3);
            expect(params.y).toBeGreaterThanOrEqual(0);
            expect(params.y).toBeLessThan(2);
            tpe.tell(params, params.x === 1 && params.y === 0 ? 1 : 0);
        }
        expect(tpe.trials).toHaveLength(30);
        expect(tpe.best()?.params).toEqual({ x: 1, y: 0 });
    });

    it('is deterministic for a seed', () => {
        expect(run(9, 25).suggestions).toEqual(run(9, 25).suggestions);
        expect(run(9, 25).suggestions).not.toEqual(run(10, 25).suggestions);
    });

    it('records a non-finite value as the worst', () => {
        const tpe = new TPESampler({ x: 2 });
        tpe.tell({ x: 0 }, NaN);
        tpe.tell({ x: 1 }, 0);
        expect(tpe.best()?.params).toEqual({ x: 1 });
    });

    it('rejects an empty category list', () => {
        expect(() => new TPESampler({ x: 0 })).toThrow(RangeError);
    });

    it('samples categories in proportion to their weights', () => {
        const rng = createRng(3);
        const counts = [0, 0, 0];
        for (let i = 0; i < 3000; i++) counts[sampleCategorical([1, 2, 7], rng)] += 1;
        expect(counts[2]).toBeGreaterThan(counts[1]);
        expect(counts[1]).toBeGreaterThan(counts[0]);
        expect(counts[2] / 3000).toBeCloseTo(0.7, 1);
    });
});
