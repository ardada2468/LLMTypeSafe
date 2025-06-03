import { TraceEntry } from '../types/module';

export class Prediction<T = Record<string, any>> {
    private _data: T;
    public trace?: TraceEntry;

    constructor(data: T, trace?: TraceEntry) {
        this._data = data;
        this.trace = trace;

        // Make all data properties accessible directly on the prediction
        Object.keys(data as Record<string, any>).forEach(key => {
            Object.defineProperty(this, key, {
                get: () => (this._data as any)[key],
                enumerable: true,
                configurable: true
            });
        });
    }

    get<K extends keyof T>(key: K): T[K] {
        return this._data[key];
    }

    toObject(): T {
        return { ...this._data };
    }

    toString(): string {
        return JSON.stringify(this._data, null, 2);
    }
} 