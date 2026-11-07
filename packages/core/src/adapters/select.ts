import type { ILanguageModel } from '../types/language-model';
import { getAdapter } from '../core/config';
import { type Adapter } from './base';
import { ChatAdapter } from './chat';
import { JSONAdapter } from './json';

// Adapters are stateless, so one shared instance of each default serves every call.
let chatAdapter: ChatAdapter | undefined;
let jsonAdapter: JSONAdapter | undefined;

/**
 * The adapter used when nothing chooses one: {@link JSONAdapter} for a model
 * with native structured output, whose decoding can then be constrained to the
 * signature's schema, and {@link ChatAdapter} (with its JSON fallback) for
 * everything else.
 */
export function defaultAdapter(lm: ILanguageModel): Adapter {
    if (lm.getCapabilities().supportsStructuredOutput) {
        return (jsonAdapter ??= new JSONAdapter());
    }
    return (chatAdapter ??= new ChatAdapter());
}

/**
 * Pick the adapter for one call: the first of `preferred` that is set (the
 * call option, then the module's own), then the active `settings.context`
 * scope's or the global `configure({ adapter })`, then {@link defaultAdapter}.
 */
export function resolveAdapter(
    lm: ILanguageModel,
    ...preferred: Array<Adapter | undefined>
): Adapter {
    for (const adapter of preferred) {
        if (adapter) return adapter;
    }
    return getAdapter() ?? defaultAdapter(lm);
}
