import { AsyncLocalStorage } from 'node:async_hooks';
import { extendSignature, type SignatureLike } from './signature';

/**
 * Per-predictor hints for one scope — how `Refine` hands each predictor the
 * advice from an earlier, failed attempt.
 *
 * A scope rather than a field on the predictor because the advice belongs to
 * one run, not to the program: concurrent runs of the same program must not
 * see each other's hints, and nothing about a hint should ever be saved by
 * `dumpState()`.
 */
const hints = new AsyncLocalStorage<ReadonlyMap<object, string>>();

/** Name of the input a hinted predictor gains, as in DSPy's `Refine`. */
export const HINT_FIELD = 'hint_';

/** Description of {@link HINT_FIELD}, as in DSPy's `Refine`. */
export const HINT_DESCRIPTION = 'A hint to the module from an earlier run';

/** Run `fn` with `map` (predictor → advice) as the hints in scope. */
export function runWithPredictorHints<T>(map: ReadonlyMap<object, string>, fn: () => T): T {
    return hints.run(map, fn);
}

/** The hint for `predictor` in this scope, if it has one. */
export function predictorHint(predictor: object): string | undefined {
    return hints.getStore()?.get(predictor);
}

const objectCache = new WeakMap<object, SignatureLike>();
const stringCache = new Map<string, SignatureLike>();

/** `signature` with a trailing {@link HINT_FIELD} input. Built once per signature. */
export function withHintField(signature: SignatureLike): SignatureLike {
    const cached =
        typeof signature === 'string' ? stringCache.get(signature) : objectCache.get(signature);
    if (cached) return cached;

    const extended = extendSignature(signature, {
        appendInputs: { [HINT_FIELD]: { description: HINT_DESCRIPTION, type: 'string' } },
    });
    if (typeof signature === 'string') stringCache.set(signature, extended);
    else objectCache.set(signature, extended);
    return extended;
}
