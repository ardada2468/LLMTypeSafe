import { type TraceEntry } from '../types/module';
import { withDataProperties } from './data-proxy';

const DATA: unique symbol = Symbol('ts-dspy.prediction.data');

/**
 * A module's output. Fields read as properties (`prediction.answer`), except
 * where a field's name collides with a method or with `trace` — those always
 * win. Every field is reachable through {@link Prediction.get} and
 * {@link Prediction.toObject}. Fields are read-only.
 */
export class Prediction<T = Record<string, any>> {
    // Symbol-keyed so no field name can collide with the internals.
    private [DATA]: T;
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
