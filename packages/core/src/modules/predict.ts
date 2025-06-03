import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { Signature } from '../core/signature';
import { ILanguageModel } from '../types/language-model';
import { parseOutput, buildPrompt } from '../utils/parsing';
import { SignatureOutput } from '../types/signature';

export class Predict<TSignature extends typeof Signature = typeof Signature> extends Module {
    constructor(signature: TSignature | string, lm?: ILanguageModel) {
        super(signature, lm);
    }

    async forward(
        inputs: Record<string, any>
    ): Promise<Prediction<SignatureOutput<TSignature>> & SignatureOutput<TSignature>> {
        try {
            const prompt = this.buildPrompt(inputs);
            const rawOutput = await this.lm.generate(prompt);

            const parsed = this.parseOutput(rawOutput) as SignatureOutput<TSignature>;

            return new Prediction(parsed) as Prediction<SignatureOutput<TSignature>> & SignatureOutput<TSignature>;
        } catch (error) {
            throw new Error(`Prediction failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    protected buildPrompt(inputs: Record<string, any>): string {
        if (!this.signature) {
            throw new Error('No signature provided');
        }
        return buildPrompt(this.signature, inputs);
    }

    protected parseOutput(rawOutput: string): Record<string, any> {
        if (!this.signature) {
            throw new Error('No signature provided');
        }
        return parseOutput(this.signature, rawOutput);
    }
} 