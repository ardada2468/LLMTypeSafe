import {
    Predict,
    throwIfAborted,
    type PartialOutput,
    type PredictionStream,
    type StreamOptions,
} from './predict';
import { Prediction } from '../core/prediction';
import { type Signature } from '../core/signature';
import type { LLMCallOptions } from '../types/language-model';
import type { SignatureOutput } from '../types/signature';

type WithReasoning<TOutput> = TOutput & { reasoning: string };

/**
 * Two-step prediction: reason in free text, then answer with that reasoning in
 * context. The result carries the signature's output fields plus `reasoning`.
 *
 * Like {@link Predict}, `TOutput` can be supplied for precise output types.
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

        // Step 2: answer with that reasoning in context, validated against the signature.
        const finalPrompt = this.buildFinalPrompt(inputs, reasoning);
        const parsed = (await this.complete(finalPrompt, options)) as TOutput;

        const combinedOutput = { ...parsed, reasoning } as WithReasoning<TOutput>;

        return new Prediction(combinedOutput) as Prediction<WithReasoning<TOutput>> &
            WithReasoning<TOutput>;
    }

    /**
     * Stream the answer step.
     *
     * The reasoning step runs first and in full — its text is what the answer
     * step is conditioned on, so there is nothing to show progressively until it
     * has finished. Every snapshot from then on carries the finished `reasoning`
     * alongside the output fields filled in so far.
     */
    async *stream(
        inputs: Record<string, any>,
        options?: StreamOptions
    ): PredictionStream<WithReasoning<TOutput>> {
        throwIfAborted(options?.signal);

        const reasoning = await this.lm.generate(this.buildReasoningPrompt(inputs), options);
        const finalPrompt = this.buildFinalPrompt(inputs, reasoning);

        const inner = this.streamComplete(finalPrompt, options);
        let step = await inner.next();
        try {
            while (!step.done) {
                yield { ...step.value, reasoning } as PartialOutput<WithReasoning<TOutput>>;
                step = await inner.next();
            }
        } finally {
            // Abandoning this generator must close the one underneath it.
            await inner.return({});
        }

        const combined = { ...step.value, reasoning } as WithReasoning<TOutput>;

        return new Prediction(combined) as Prediction<WithReasoning<TOutput>> &
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
