/**
 * Deterministic randomness for optimizers.
 *
 * `Math.random()` cannot be seeded, so a compile that used it would pick a
 * different set of demos on every run — nobody could reproduce a result or write
 * a stable test for one. These are small, self-contained, and identical across
 * platforms and Node versions.
 */

/** A seeded pseudo-random source producing values in `[0, 1)`. */
export type Rng = () => number;

/**
 * mulberry32: a 32-bit PRNG that is short, fast, and good enough for shuffling.
 * Not cryptographically secure, and not meant to be.
 */
export function createRng(seed: number): Rng {
    // Coerce to a 32-bit integer so any finite seed, including a negative or
    // fractional one, produces a usable state.
    let state = Math.trunc(seed) | 0;
    return function next(): number {
        state = (state + 0x6d2b79f5) | 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Fisher-Yates over a copy: the input array is left alone. */
export function shuffled<T>(items: readonly T[], rng: Rng): T[] {
    const result = [...items];
    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
}
