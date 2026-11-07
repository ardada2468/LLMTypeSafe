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
 */
export function rolloutMiddleware(rollout: RolloutSettings): LMMiddleware {
    return (request, next) =>
        next({
            ...request,
            options: {
                ...request.options,
                temperature: rollout.temperature,
                metadata: { ...request.options?.metadata, rolloutId: rollout.rolloutId },
            },
        });
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
    for (const target of [copy, ...copy.namedSubModules().map(([, sub]) => sub)]) {
        const holder = target as unknown as { lm?: ILanguageModel };
        if (holder.lm) holder.lm = wrap(holder.lm);
    }
    prepare?.(copy);

    const scoped = findDefaultLM();
    const callOptions = options?.lm ? { ...options, lm: wrap(options.lm) } : options;
    return context(scoped ? { lm: wrap(scoped) } : {}, () => copy.forward(inputs, callOptions));
}
