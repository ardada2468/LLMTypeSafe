import { Predict } from './predict';
import { type Prediction } from '../core/prediction';
import { type Signature, type SignatureSource } from '../core/signature';
import type { LLMCallOptions } from '../types/language-model';
import type { SignatureInput, SignatureOutput } from '../types/signature';

type WithReasoning<TOutput> = TOutput & { reasoning: string };

/**
 * Two-step prediction: reason in free text, then answer with that reasoning in
 * context. The result carries the signature's output fields plus `reasoning`.
 *
 * Like {@link Predict}, `TOutput` can be supplied for precise output types, and
 * `options.repairAttempts` buys extra round-trips when the answer fails
 * validation. Repair applies to the answering step only — the reasoning is
 * already settled by then, so it is reused rather than regenerated.
 */
export class ChainOfThought<
    TSignature extends SignatureSource = typeof Signature,
    TOutput extends Record<string, any> = SignatureOutput<TSignature>,
> extends Predict<TSignature, TOutput> {
    async forward(
        inputs: SignatureInput<TSignature>,
        options?: LLMCallOptions
    ): Promise<Prediction<WithReasoning<TOutput>> & WithReasoning<TOutput>> {
        const prediction = await this.traced<WithReasoning<TOutput>>(inputs, async (span) => {
            // Step 1: reason in the open, as free text.
            const reasoningPrompt = this.buildReasoningPrompt(inputs);
            span?.startCall(reasoningPrompt);
            const reasoning = await this.lm.generate(reasoningPrompt, options);
            span?.endCall(reasoning);

            // Step 2: answer with that reasoning in context, validated against
            // the signature. `complete` owns the repair loop, so a failed answer
            // is retried against this same prompt without reasoning again.
            const finalPrompt = this.buildFinalPrompt(inputs, reasoning);
            const parsed = (await this.complete(finalPrompt, options, span)) as TOutput;

            return { ...parsed, reasoning } as WithReasoning<TOutput>;
        });

        return prediction as Prediction<WithReasoning<TOutput>> & WithReasoning<TOutput>;
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
