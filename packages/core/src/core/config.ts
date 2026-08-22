import { type ILanguageModel } from '../types/language-model';
import { MemoryCache, type Cache } from './cache';

export interface ConfigureOptions {
    /** Model used by any module constructed without an explicit one. */
    lm?: ILanguageModel;
    /**
     * Response caching, off unless you ask for it. `true` uses a process-wide
     * {@link MemoryCache}, `false` disables caching, and a {@link Cache}
     * implementation routes lookups through your own store — Redis, SQLite,
     * disk, anything.
     */
    cache?: boolean | Cache;
    tracing?: boolean;
}

class DSPyConfig {
    private static instance: DSPyConfig;
    private _defaultLM?: ILanguageModel;
    /**
     * Opt-in: a process-wide cache that replays a previous answer for a
     * repeated prompt changes what a program does, so it is never switched on
     * behind the caller's back.
     */
    private _cache: boolean | Cache = false;
    private _defaultCache?: MemoryCache;
    private _tracing: boolean = false;

    private constructor() {}

    static getInstance(): DSPyConfig {
        if (!DSPyConfig.instance) {
            DSPyConfig.instance = new DSPyConfig();
        }
        return DSPyConfig.instance;
    }

    static configure(options: ConfigureOptions): void {
        const config = DSPyConfig.getInstance();
        if (options.lm) config._defaultLM = options.lm;
        if (options.cache !== undefined) config._cache = options.cache;
        if (options.tracing !== undefined) config._tracing = options.tracing;
    }

    static getDefaultLM(): ILanguageModel {
        const config = DSPyConfig.getInstance();
        if (!config._defaultLM) {
            throw new Error(
                'No default language model configured. Call configure({ lm: ... }) first.'
            );
        }
        return config._defaultLM;
    }

    static isCacheEnabled(): boolean {
        return DSPyConfig.getInstance()._cache !== false;
    }

    /**
     * The active cache, or `undefined` when caching is off. The built-in
     * memory cache is created on first use so that a process which never
     * caches never allocates one.
     */
    static getCache(): Cache | undefined {
        const config = DSPyConfig.getInstance();
        if (config._cache === false) return undefined;
        if (config._cache !== true) return config._cache;

        config._defaultCache ??= new MemoryCache();
        return config._defaultCache;
    }

    /**
     * Empty every cache this config knows about, whether or not caching is
     * currently enabled — otherwise `configure({ cache: false })` followed by a
     * reset would leave entries in place for the next time it is switched on.
     * A custom {@link Cache} without a `clear` method is left untouched.
     */
    static async clearCache(): Promise<void> {
        const config = DSPyConfig.getInstance();
        config._defaultCache?.clear();
        if (typeof config._cache === 'object') {
            await config._cache.clear?.();
        }
    }

    static isTracingEnabled(): boolean {
        return DSPyConfig.getInstance()._tracing;
    }
}

export const configure = DSPyConfig.configure;
export const getDefaultLM = DSPyConfig.getDefaultLM;
export const isCacheEnabled = DSPyConfig.isCacheEnabled;
export const getCache = DSPyConfig.getCache;
export const clearCache = DSPyConfig.clearCache;
export const isTracingEnabled = DSPyConfig.isTracingEnabled;
