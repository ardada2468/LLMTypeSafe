import { Predict } from './predict';
import { Prediction } from '../core/prediction';
import { type Signature } from '../core/signature';
import type { LLMCallOptions } from '../types/language-model';
import type { SignatureOutput } from '../types/signature';

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
    TSignature extends typeof Signature = typeof Signature,
    TOutput extends Record<string, any> = SignatureOutput<TSignature>,
> extends Predict<TSignature, TOutput> {
    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction<WithReasoning<TOutput>> & WithReasoning<TOutput>> {
        // Step 1: reason in the open, as free text.
        const reasoningPrompt = this.buildReasoningPrompt(inputs);
        const reasoning = await this.lm.generate(reasoningPrompt, options);

        // Step 2: answer with that reasoning in context, validated against the
        // signature. `complete` owns the repair loop, so a failed answer is
        // retried against this same prompt without reasoning again.
        const finalPrompt = this.buildFinalPrompt(inputs, reasoning);
        const parsed = (await this.complete(finalPrompt, options)) as TOutput;

        const combinedOutput = { ...parsed, reasoning } as WithReasoning<TOutput>;

        return new Prediction(combinedOutput) as Prediction<WithReasoning<TOutput>> &
            WithReasoning<TOutput>;
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
