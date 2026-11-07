import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { context, findDefaultLM } from './config';
import { wrapLM, type LMMiddleware } from './middleware';
import { type Module } from './module';
import { type Prediction } from './prediction';

/** What makes one rollout's language-model calls differ from another's. */
export interface RolloutSettings {
    /**
     * Joins the cache key (see `buildCacheKey`), so a rollout is a fresh
     * sample rather than the previous rollout's cached reply.
     */
    rolloutId: number;
    /** Sampling temperature for every call the rollout makes. */
    temperature: number;
}

/**
 * Middleware stamping a rollout onto every call: its temperature, and its id
 * in `options.metadata.rolloutId`. Both override whatever the call carried —
 * a rollout that inherited `temperature: 0` from a module's defaults would
 * sample the same answer every time, which defeats the point.
 *
 * A call that already carries a rollout id — from a rollout nested inside
 * this one (a `BestOfN` inside a `Refine`), whose middleware runs first — keeps
 * it, prefixed with this rollout's: `"1/0"`, `"1/1"`, …. Overwriting it would
 * give every inner rollout the same id, and with caching on the inner
 * `BestOfN` would be served its first attempt's reply on every attempt.
 */
export function rolloutMiddleware(rollout: RolloutSettings): LMMiddleware {
    return (request, next) => {
        const inner: unknown = request.options?.metadata?.rolloutId;
        const rolloutId =
            inner === undefined || inner === null
                ? rollout.rolloutId
                : `${rollout.rolloutId}/${String(inner)}`;
        return next({
            ...request,
            options: {
                ...request.options,
                temperature: rollout.temperature,
                metadata: { ...request.options?.metadata, rolloutId },
            },
        });
    };
}

/** Rollout copy → what it was copied from. */
const origins = new WeakMap<object, object>();

/**
 * The object `copy` was made from by a rollout, if it is a rollout copy. A
 * rollout of a rollout's copy chains, so callers may need to follow it.
 */
export function rolloutOrigin(copy: object): object | undefined {
    return origins.get(copy);
}

/**
 * Run one rollout of `module`: a {@link Module.deepCopy} of it whose every
 * language model — wherever it comes from — is wrapped with
 * {@link rolloutMiddleware}.
 *
 * DSPy does this with `lm.copy(rollout_id=…, temperature=1.0)` and
 * `set_lm`. Here a model can reach a call three ways, and all three are
 * covered, because a hand-written `forward` cannot be relied on to pass
 * per-call options down to the predictors it calls:
 *
 * - a module's own `lm` field, on the copy and every sub-module of it;
 * - the scope's model, replaced for the rollout with `settings.context`;
 * - `options.lm`, wrapped in place.
 *
 * The copy is what keeps concurrent rollouts — and the caller's original
 * module — from seeing each other's wrapped models.
 */
export function runRollout(
    module: Module,
    inputs: Record<string, any>,
    options: LLMCallOptions | undefined,
    rollout: RolloutSettings,
    prepare?: (copy: Module) => void
): Promise<Prediction> {
    const middleware = rolloutMiddleware(rollout);
    const wrap = (lm: ILanguageModel) => wrapLM(lm, middleware);

    const copy = module.deepCopy();
    // The copy has the original's shape, so its predictors pair up with the
    // original's in order. Recorded so `streamify` can label a rollout's
    // field events with the path of the predictor it is a copy of.
    const originals = module.namedPredictors();
    copy.namedPredictors().forEach(([, predictor], index) => {
        const original = originals[index]?.[1];
        if (original) origins.set(predictor, original);
    });
    for (const target of [copy, ...copy.namedSubModules().map(([, sub]) => sub)]) {
        const holder = target as unknown as { lm?: ILanguageModel };
        if (holder.lm) holder.lm = wrap(holder.lm);
    }
    prepare?.(copy);

    const scoped = findDefaultLM();
    const callOptions = options?.lm ? { ...options, lm: wrap(options.lm) } : options;
    return context(scoped ? { lm: wrap(scoped) } : {}, () => copy.forward(inputs, callOptions));
}
