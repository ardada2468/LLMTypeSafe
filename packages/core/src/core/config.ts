import { ILanguageModel } from '../types/language-model';

class DSPyConfig {
    private static instance: DSPyConfig;
    private _defaultLM?: ILanguageModel;
    private _cache: boolean = true;
    private _tracing: boolean = false;

    private constructor() { }

    static getInstance(): DSPyConfig {
        if (!DSPyConfig.instance) {
            DSPyConfig.instance = new DSPyConfig();
        }
        return DSPyConfig.instance;
    }

    static configure(options: {
        lm?: ILanguageModel;
        cache?: boolean;
        tracing?: boolean;
    }): void {
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
        return DSPyConfig.getInstance()._cache;
    }

    static isTracingEnabled(): boolean {
        return DSPyConfig.getInstance()._tracing;
    }
}

export const configure = DSPyConfig.configure;
export const getDefaultLM = DSPyConfig.getDefaultLM;
export const isCacheEnabled = DSPyConfig.isCacheEnabled;
export const isTracingEnabled = DSPyConfig.isTracingEnabled; 