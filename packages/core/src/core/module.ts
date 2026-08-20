import { type Signature } from './signature';
import { type Prediction } from './prediction';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { getDefaultLM } from './config';

export abstract class Module {
    protected lm: ILanguageModel;
    protected signature?: typeof Signature | string;

    constructor(signature?: typeof Signature | string, lm?: ILanguageModel) {
        this.signature = signature;
        this.lm = lm || getDefaultLM();
    }

    abstract forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction>;

    /** Alias for {@link forward}, so modules read like function calls. */
    async call(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /** Alias for {@link forward}, mirroring DSPy's Python naming. */
    async __call__(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }
}
