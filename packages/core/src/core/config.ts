import { AsyncLocalStorage } from 'node:async_hooks';
import { type ILanguageModel } from '../types/language-model';
import { MemoryCache, type Cache } from './cache';
import { type TraceEntry } from '../types/module';
import type { Callback } from './callbacks';
import type { Adapter } from './adapter';

/**
 * Receives every trace entry as it is recorded. Forward entries to Langfuse,
 * OpenTelemetry, or your own logger from here — the library never prints.
 */
export type TraceHandler = (entry: TraceEntry) => void;

/**
 * Price of one model, in US dollars (or any currency you like — it is only ever
 * multiplied) per **million** tokens.
 */
export interface ModelPricing {
    /** Per million prompt tokens. */
    input: number;
    /** Per million completion tokens (reasoning tokens are billed as these). */
    output: number;
    /**
     * Per million prompt tokens served from the provider's prompt cache. Falls
     * back to `input` when omitted.
     */
    cachedInput?: number;
}

/**
 * Prices keyed by `provider/model` (e.g. `openai/gpt-5.2`), or by the bare
 * model name as a fallback. ts-dspy ships no price table of its own — tables
 * go stale — so cost is reported only for models you price here.
 */
export type PricingTable = Record<string, ModelPricing>;

/**
 * Settings that can be overridden for a single async scope with
 * {@link settings.context}. Everything here can also be set globally with
 * {@link configure}.
 */
export interface SettingsOverrides {
    /** Model used by any module without one of its own. */
    lm?: ILanguageModel;
    /**
     * Response caching, off unless you ask for it. `true` uses a process-wide
     * {@link MemoryCache}, `false` disables caching, and a {@link Cache}
     * implementation routes lookups through your own store — Redis, SQLite,
     * disk, anything.
     */
    cache?: boolean | Cache;
    /**
     * Record a trace for every module invocation. Off by default, and costs
     * nothing while off.
     */
    tracing?: boolean;
    /** Called with each trace entry as it is recorded. Requires `tracing`. */
    onTrace?: TraceHandler;
    /**
     * Observers notified as modules, language models and tools run. In a
     * {@link settings.context} scope this list **replaces** the global one
     * rather than adding to it, so a scope can also silence callbacks with `[]`.
     */
    callbacks?: Callback[];
    /**
     * Prompt formatting/parsing strategy for every module that does not choose
     * its own. A call's `adapter` option and a module's `adapter` option both
     * take precedence; with none set anywhere, modules use `JSONAdapter` on a
     * model with native structured output and `ChatAdapter` otherwise.
     */
    adapter?: Adapter;
    /**
     * Collect exact per-invocation token usage, readable afterwards with
     * `prediction.getLMUsage()`. Tracing turns this on implicitly.
     */
    trackUsage?: boolean;
    /** Prices used to fill `totalCost` in usage reports. */
    pricing?: PricingTable;
}

export interface ConfigureOptions extends SettingsOverrides {
    /**
     * How many entries `inspectHistory()` retains. Defaults to 100; older
     * entries are dropped first.
     */
    traceHistorySize?: number;
}

const DEFAULT_TRACE_HISTORY_SIZE = 100;

/**
 * Process-wide defaults. Kept as a plain object rather than the old singleton
 * class: every reader goes through a getter below that consults the active
 * scope first, so there is nothing for a class to encapsulate.
 */
const globals: {
    lm?: ILanguageModel;
    /**
     * Opt-in: a process-wide cache that replays a previous answer for a
     * repeated prompt changes what a program does, so it is never switched on
     * behind the caller's back.
     */
    cache: boolean | Cache;
    defaultCache?: MemoryCache;
    tracing: boolean;
    onTrace?: TraceHandler;
    traceHistorySize: number;
    callbacks: Callback[];
    adapter?: Adapter;
    trackUsage: boolean;
    pricing?: PricingTable;
} = {
    cache: false,
    tracing: false,
    traceHistorySize: DEFAULT_TRACE_HISTORY_SIZE,
    callbacks: [],
    trackUsage: false,
};

/**
 * The overrides in force for the current async scope.
 *
 * `AsyncLocalStorage` rather than a mutable "current settings" variable: two
 * requests handled concurrently by one server must be able to use different
 * models without either seeing the other's, and only async-context
 * propagation gives that across `await` boundaries.
 */
const scope = new AsyncLocalStorage<Readonly<SettingsOverrides>>();

/**
 * Update the global defaults. Fields left out keep their current value; pass
 * `onTrace: undefined` explicitly to detach a handler.
 */
export function configure(options: ConfigureOptions): void {
    if (options.lm) globals.lm = options.lm;
    if (options.cache !== undefined) globals.cache = options.cache;
    if (options.tracing !== undefined) globals.tracing = options.tracing;
    // Presence, not truthiness: `onTrace: undefined` detaches the handler.
    if ('onTrace' in options) globals.onTrace = options.onTrace;
    if (options.callbacks !== undefined) globals.callbacks = [...options.callbacks];
    if ('adapter' in options) globals.adapter = options.adapter;
    if (options.trackUsage !== undefined) globals.trackUsage = options.trackUsage;
    if ('pricing' in options) globals.pricing = options.pricing;
    if (options.traceHistorySize !== undefined) {
        // A NaN limit would silently unbound the buffer, so fall back instead.
        globals.traceHistorySize = Number.isFinite(options.traceHistorySize)
            ? Math.max(0, Math.floor(options.traceHistorySize))
            : DEFAULT_TRACE_HISTORY_SIZE;
    }
}

/**
 * Run `fn` with `overrides` layered over the current settings.
 *
 * The overrides apply to everything `fn` does, including work it awaits, and
 * to nothing else — concurrent scopes never see each other's values, and the
 * globals are untouched:
 *
 * ```ts
 * const [a, b] = await Promise.all([
 *     settings.context({ lm: gpt }, () => program.forward(input)),
 *     settings.context({ lm: claude }, () => program.forward(input)),
 * ]);
 * ```
 *
 * Scopes nest; an inner scope inherits the outer one's overrides and replaces
 * only the fields it sets. A field set to `undefined` is treated as unset, so
 * spreading a partially-filled options object cannot accidentally erase an
 * outer override.
 *
 * Returns whatever `fn` returns — a promise if `fn` is async.
 */
export function context<T>(overrides: SettingsOverrides, fn: () => T): T {
    const merged: SettingsOverrides = { ...scope.getStore() };
    for (const [key, value] of Object.entries(overrides)) {
        if (value !== undefined) {
            (merged as Record<string, unknown>)[key] = value;
        }
    }
    return scope.run(merged, fn);
}

/** Alias for {@link context}, for code that reads better as a verb. */
export const withSettings = context;

/** The overrides active in this async scope, if any. */
function active(): Readonly<SettingsOverrides> | undefined {
    return scope.getStore();
}

/** The model for the current scope, or `undefined` when none is configured. */
export function findDefaultLM(): ILanguageModel | undefined {
    return active()?.lm ?? globals.lm;
}

/**
 * The model for the current scope.
 *
 * @throws when neither a scope nor `configure()` has supplied one.
 */
export function getDefaultLM(): ILanguageModel {
    const lm = findDefaultLM();
    if (!lm) {
        throw new Error(
            'No language model configured. Pass one to the module, call ' +
                'configure({ lm }), or run inside settings.context({ lm }, ...).'
        );
    }
    return lm;
}

function cacheSetting(): boolean | Cache {
    return active()?.cache ?? globals.cache;
}

export function isCacheEnabled(): boolean {
    return cacheSetting() !== false;
}

/**
 * The active cache, or `undefined` when caching is off. The built-in
 * memory cache is created on first use so that a process which never
 * caches never allocates one. Scopes asking for `cache: true` share it.
 */
export function getCache(): Cache | undefined {
    const setting = cacheSetting();
    if (setting === false) return undefined;
    if (setting !== true) return setting;

    globals.defaultCache ??= new MemoryCache();
    return globals.defaultCache;
}

/**
 * Empty every cache this config knows about, whether or not caching is
 * currently enabled — otherwise `configure({ cache: false })` followed by a
 * reset would leave entries in place for the next time it is switched on.
 * A custom {@link Cache} without a `clear` method is left untouched.
 */
export async function clearCache(): Promise<void> {
    globals.defaultCache?.clear();
    if (typeof globals.cache === 'object') {
        await globals.cache.clear?.();
    }
    const scoped = active()?.cache;
    if (typeof scoped === 'object' && scoped !== globals.cache) {
        await scoped.clear?.();
    }
}

export function isTracingEnabled(): boolean {
    return active()?.tracing ?? globals.tracing;
}

export function getTraceHandler(): TraceHandler | undefined {
    const scoped = active();
    if (scoped && 'onTrace' in scoped) return scoped.onTrace;
    return globals.onTrace;
}

export function getTraceHistorySize(): number {
    return globals.traceHistorySize;
}

/** Callbacks for the current scope. Never `undefined`, often empty. */
export function getCallbacks(): readonly Callback[] {
    return active()?.callbacks ?? globals.callbacks;
}

/** The adapter of the active scope, else the global one. See {@link SettingsOverrides.adapter}. */
export function getAdapter(): Adapter | undefined {
    return active()?.adapter ?? globals.adapter;
}

/** Whether per-invocation usage should be collected: explicitly, or for tracing. */
export function isUsageTrackingEnabled(): boolean {
    return (active()?.trackUsage ?? globals.trackUsage) || isTracingEnabled();
}

export function getPricing(): PricingTable | undefined {
    return active()?.pricing ?? globals.pricing;
}

/**
 * Settings, as one object: `configure` for the globals, `context` for a scope,
 * and the getters every module consults at call time.
 */
export const settings = {
    configure,
    context,
    get lm(): ILanguageModel | undefined {
        return findDefaultLM();
    },
    get cache(): Cache | undefined {
        return getCache();
    },
    get tracing(): boolean {
        return isTracingEnabled();
    },
    get callbacks(): readonly Callback[] {
        return getCallbacks();
    },
    get adapter(): Adapter | undefined {
        return getAdapter();
    },
    get trackUsage(): boolean {
        return isUsageTrackingEnabled();
    },
    get pricing(): PricingTable | undefined {
        return getPricing();
    },
} as const;
