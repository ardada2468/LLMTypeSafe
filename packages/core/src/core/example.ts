import { withDataProperties } from './data-proxy';

const DATA: unique symbol = Symbol('ts-dspy.example.data');
const INPUT_KEYS: unique symbol = Symbol('ts-dspy.example.inputKeys');

/**
 * One labelled data point: a bag of named fields, optionally split into inputs
 * and outputs with {@link Example.withInputs}.
 *
 * Fields read as properties (`ex.question`), except where a field's name
 * collides with a method or built-in property — `ex.get` is always the method.
 * Every field is reachable through {@link Example.get} and {@link Example.toObject}.
 */
export class Example {
    // Symbol-keyed so no field name can collide with the internals.
    private [DATA]: Record<string, any>;
    private [INPUT_KEYS]?: string[];

    constructor(data: Record<string, any>) {
        this[DATA] = { ...data };
        const store = this[DATA];
        return withDataProperties(this, () => store, { writable: true });
    }

    get(key: string): any {
        return this[DATA][key];
    }

    set(key: string, value: any): void {
        this[DATA][key] = value;
    }

    withInputs(...inputKeys: string[]): Example {
        const newExample = new Example(this[DATA]);
        newExample[INPUT_KEYS] = inputKeys;
        return newExample;
    }

    /** True when {@link Example.withInputs} has designated the input fields. */
    hasInputKeys(): boolean {
        return this[INPUT_KEYS] !== undefined;
    }

    /** The designated input field names, or `undefined` before {@link Example.withInputs}. */
    getInputKeys(): string[] | undefined {
        const keys = this[INPUT_KEYS];
        return keys ? [...keys] : undefined;
    }

    getInputs(): Record<string, any> {
        const inputKeys = this[INPUT_KEYS];
        if (!inputKeys) {
            throw new Error('Input keys not specified. Use withInputs() first.');
        }

        const inputs: Record<string, any> = {};
        inputKeys.forEach((key) => {
            inputs[key] = this[DATA][key];
        });
        return inputs;
    }

    getOutputs(): Record<string, any> {
        const inputKeys = this[INPUT_KEYS];
        if (!inputKeys) {
            throw new Error('Input keys not specified. Use withInputs() first.');
        }

        const outputs: Record<string, any> = {};
        Object.keys(this[DATA]).forEach((key) => {
            if (!inputKeys.includes(key)) {
                outputs[key] = this[DATA][key];
            }
        });
        return outputs;
    }

    toObject(): Record<string, any> {
        return { ...this[DATA] };
    }

    /** Serialise as the data alone, including fields a method name shadows. */
    toJSON(): Record<string, any> {
        return this.toObject();
    }
}
