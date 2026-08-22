import { type ILanguageModel } from '../types/language-model';
import { type TraceEntry } from '../types/module';

/**
 * Receives every trace entry as it is recorded. Forward entries to Langfuse,
 * OpenTelemetry, or your own logger from here — the library never prints.
 */
export type TraceHandler = (entry: TraceEntry) => void;

export interface ConfigureOptions {
    /** Language model used by modules constructed without an explicit one. */
    lm?: ILanguageModel;
    cache?: boolean;
    /**
     * Record a trace for every module invocation. Off by default, and costs
     * nothing while off.
     */
    tracing?: boolean;
    /** Called with each trace entry as it is recorded. Requires `tracing`. */
    onTrace?: TraceHandler;
    /**
     * How many entries `inspectHistory()` retains. Defaults to 100; older
     * entries are dropped first.
     */
    traceHistorySize?: number;
}

const DEFAULT_TRACE_HISTORY_SIZE = 100;

class DSPyConfig {
    private static instance: DSPyConfig;
    private _defaultLM?: ILanguageModel;
    private _cache: boolean = true;
    private _tracing: boolean = false;
    private _onTrace?: TraceHandler;
    private _traceHistorySize: number = DEFAULT_TRACE_HISTORY_SIZE;

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
        // Presence, not truthiness: `onTrace: undefined` detaches the handler.
        if ('onTrace' in options) config._onTrace = options.onTrace;
        if (options.traceHistorySize !== undefined) {
            // A NaN limit would silently unbound the buffer, so fall back instead.
            config._traceHistorySize = Number.isFinite(options.traceHistorySize)
                ? Math.max(0, Math.floor(options.traceHistorySize))
                : DEFAULT_TRACE_HISTORY_SIZE;
        }
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

    static getTraceHandler(): TraceHandler | undefined {
        return DSPyConfig.getInstance()._onTrace;
    }

    static getTraceHistorySize(): number {
        return DSPyConfig.getInstance()._traceHistorySize;
    }
}

export const configure = DSPyConfig.configure;
export const getDefaultLM = DSPyConfig.getDefaultLM;
export const isCacheEnabled = DSPyConfig.isCacheEnabled;
export const isTracingEnabled = DSPyConfig.isTracingEnabled;
export const getTraceHandler = DSPyConfig.getTraceHandler;
export const getTraceHistorySize = DSPyConfig.getTraceHistorySize;
