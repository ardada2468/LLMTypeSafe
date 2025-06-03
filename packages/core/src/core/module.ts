import { Signature } from './signature';
import { Prediction } from './prediction';
import { ILanguageModel } from '../types/language-model';
import { getDefaultLM } from './config';
import * as fs from 'fs';

export abstract class Module {
    protected lm: ILanguageModel;
    protected signature?: typeof Signature | string;
    protected _compiled: boolean = false;

    constructor(signature?: typeof Signature | string, lm?: ILanguageModel) {
        this.signature = signature;
        this.lm = lm || getDefaultLM();
    }

    abstract forward(inputs: Record<string, any>): Promise<Prediction>;

    // Make modules callable like functions
    async __call__(inputs: Record<string, any>): Promise<Prediction> {
        return this.forward(inputs);
    }

    // Enable direct call syntax
    async call(inputs: Record<string, any>): Promise<Prediction> {
        return this.forward(inputs);
    }

    async save(path: string): Promise<void> {
        const serialized = {
            type: this.constructor.name,
            signature: this.signature,
            compiled: this._compiled,
            // Add any module-specific state here
        };
        await fs.promises.writeFile(path, JSON.stringify(serialized, null, 2));
    }

    static async load(path: string): Promise<Module> {
        const data = JSON.parse(await fs.promises.readFile(path, 'utf-8'));
        // Implementation depends on module registry
        throw new Error('Module loading not implemented yet');
    }

    get compiled(): boolean {
        return this._compiled;
    }
} 