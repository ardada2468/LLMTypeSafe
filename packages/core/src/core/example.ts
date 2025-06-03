export class Example {
    private _data: Record<string, any>;
    private _inputKeys?: string[];

    constructor(data: Record<string, any>) {
        this._data = { ...data };

        // Make all data properties accessible directly
        Object.keys(data).forEach(key => {
            Object.defineProperty(this, key, {
                get: () => this._data[key],
                set: (value) => { this._data[key] = value; },
                enumerable: true,
                configurable: true
            });
        });
    }

    get(key: string): any {
        return this._data[key];
    }

    set(key: string, value: any): void {
        this._data[key] = value;
        Object.defineProperty(this, key, {
            get: () => this._data[key],
            set: (val) => { this._data[key] = val; },
            enumerable: true,
            configurable: true
        });
    }

    withInputs(...inputKeys: string[]): Example {
        const newExample = new Example(this._data);
        newExample._inputKeys = inputKeys;
        return newExample;
    }

    getInputs(): Record<string, any> {
        if (!this._inputKeys) {
            throw new Error('Input keys not specified. Use withInputs() first.');
        }

        const inputs: Record<string, any> = {};
        this._inputKeys.forEach(key => {
            inputs[key] = this._data[key];
        });
        return inputs;
    }

    getOutputs(): Record<string, any> {
        if (!this._inputKeys) {
            throw new Error('Input keys not specified. Use withInputs() first.');
        }

        const outputs: Record<string, any> = {};
        Object.keys(this._data).forEach(key => {
            if (!this._inputKeys!.includes(key)) {
                outputs[key] = this._data[key];
            }
        });
        return outputs;
    }

    toObject(): Record<string, any> {
        return { ...this._data };
    }
} 