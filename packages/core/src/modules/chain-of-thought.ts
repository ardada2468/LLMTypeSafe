import { Predict } from './predict';
import { Prediction } from '../core/prediction';
import { Signature } from '../core/signature';
import { SignatureOutput } from '../types/signature';

export class ChainOfThought<TSignature extends typeof Signature = typeof Signature> extends Predict<TSignature> {
    async forward(
        inputs: Record<string, any>
    ): Promise<Prediction<SignatureOutput<TSignature> & { reasoning: string }> & SignatureOutput<TSignature> & { reasoning: string }> {
        try {
            // Step 1: Generate reasoning
            const reasoningPrompt = this.buildReasoningPrompt(inputs);
            const reasoning = await this.lm.generate(reasoningPrompt);

            // Step 2: Generate final answer with reasoning
            const finalPrompt = this.buildFinalPrompt(inputs, reasoning);
            const finalOutput = await this.lm.generate(finalPrompt);

            // parseOutput returns Record<string, any>, cast to base output shape
            const parsed = this.parseOutput(finalOutput) as SignatureOutput<TSignature>;

            // Combine parsed output with the reasoning
            const combinedOutput = { ...parsed, reasoning };

            return new Prediction(combinedOutput) as Prediction<SignatureOutput<TSignature> & { reasoning: string }> & SignatureOutput<TSignature> & { reasoning: string };
        } catch (error) {
            throw new Error(`ChainOfThought failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    private buildReasoningPrompt(inputs: Record<string, any>): string {
        const basePrompt = this.buildPrompt(inputs);
        return `${basePrompt}\n\nLet's think step by step. Please provide your reasoning:`;
    }

    private buildFinalPrompt(inputs: Record<string, any>, reasoning: string): string {
        const basePrompt = this.buildPrompt(inputs);
        return `${basePrompt}\n\nReasoning: ${reasoning}\n\nBased on this reasoning, provide your final answer:`;
    }
} 