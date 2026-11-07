import { Predict } from './predict';
import {
    extendSignature,
    type Signature,
    type SignatureLike,
    type SignatureSource,
} from '../core/signature';
import type { SignatureOutput } from '../types/signature';

/** The output of a {@link ChainOfThought}: the signature's fields plus `reasoning`. */
export type WithReasoning<TOutput> = TOutput & { reasoning: string };

/** Description of the `reasoning` field ChainOfThought prepends, as in DSPy. */
export const REASONING_DESCRIPTION = 'Think step by step in order to produce the outputs';

/**
 * Prediction with a reasoning step: the signature gains a leading `reasoning`
 * output, so the model writes out its thinking before — and in the same reply
 * as — the answer. One language-model call, exactly as in DSPy.
 *
 * The result carries `reasoning` alongside the signature's own outputs. It is
 * a `Predict` in every other respect: demos (with or without a `reasoning`
 * field), instructions, adapters, images, self-repair and streaming all work
 * the same, and `namedPredictors()` reports it like any other predictor.
 *
 * Like {@link Predict}, `TOutput` can be supplied for precise output types;
 * `reasoning` is added to it.
 */
export class ChainOfThought<
    TSignature extends SignatureSource = typeof Signature,
    TOutput extends Record<string, any> = SignatureOutput<TSignature>,
> extends Predict<TSignature, WithReasoning<TOutput>> {
    protected predictionSignature(): SignatureLike {
        return withReasoning(this.requireSignature());
    }
}

// Extended signatures, built once per base signature. Kept off the instance so
// the composition walk and deep copies never see them.
const objectCache = new WeakMap<object, SignatureLike>();
const stringCache = new Map<string, SignatureLike>();

/** `signature` with a leading `reasoning` output, as {@link ChainOfThought} sends it. */
export function withReasoning(signature: SignatureLike): SignatureLike {
    const cached =
        typeof signature === 'string' ? stringCache.get(signature) : objectCache.get(signature);
    if (cached) return cached;

    const extended = extendSignature(signature, {
        prependOutputs: { reasoning: { description: REASONING_DESCRIPTION, type: 'string' } },
    });
    if (typeof signature === 'string') stringCache.set(signature, extended);
    else objectCache.set(signature, extended);
    return extended;
}
