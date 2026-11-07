import { type TraceEntry } from '../types/module';
import { withDataProperties } from './data-proxy';
import type { LMUsage } from './usage';

const DATA: unique symbol = Symbol('ts-dspy.prediction.data');
const USAGE: unique symbol = Symbol('ts-dspy.prediction.usage');

/**
 * A module's output. Fields read as properties (`prediction.answer`), except
 * where a field's name collides with a method or with `trace` — those always
 * win. Every field is reachable through {@link Prediction.get} and
 * {@link Prediction.toObject}. Fields are read-only.
 */
export class Prediction<T = Record<string, any>> {
    // Symbol-keyed so no field name can collide with the internals.
    private [DATA]: T;
    private [USAGE]?: LMUsage;
    /**
     * What the module did to produce this, when `configure({ tracing: true })`
     * is set. Always this property, even when the signature declares an output
     * field named `trace` — read that one with `get('trace')`.
     */
    public trace?: TraceEntry;

    constructor(data: T, trace?: TraceEntry) {
        this[DATA] = data;
        this.trace = trace;
        return withDataProperties(this, () => this[DATA] as Record<string, any>, {
            writable: false,
        });
    }

    get<K extends keyof T>(key: K): T[K] {
        return this[DATA][key];
    }

    /**
     * Token usage of the invocation that produced this prediction — every
     * language-model call it made, nested modules included — keyed by
     * `provider/model`.
     *
     * Collected only when `configure({ trackUsage: true })` or tracing is on
     * (or a callback is registered); otherwise an empty object. Exact under
     * concurrency: each invocation collects into its own async-local scope.
     */
    getLMUsage(): LMUsage {
        const usage = this[USAGE];
        if (!usage) return {};
        const copy: LMUsage = {};
        for (const [key, stats] of Object.entries(usage)) copy[key] = { ...stats };
        return copy;
    }

    toObject(): T {
        return { ...this[DATA] };
    }

    /** Serialise as the data alone, including fields a method name shadows. */
    toJSON(): T {
        return this.toObject();
    }

    toString(): string {
        return JSON.stringify(this[DATA], null, 2);
    }
}

/** Attach collected usage to a prediction. Internal: called by `Module`. */
export function attachLMUsage(prediction: Prediction<any>, usage: LMUsage): void {
    (prediction as unknown as Record<symbol, LMUsage>)[USAGE] = usage;
}
