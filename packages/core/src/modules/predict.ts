import { Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { type Signature, type SignatureLike, type SignatureSource } from '../core/signature';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { parseOutput, buildPrompt } from '../utils/parsing';
import {
    buildOutputSchema,
    buildOutputJsonSchema,
    getOutputFieldConfigs,
    stripAbsentNulls,
} from '../utils/schema';
import { ValidationError, type FieldValidationIssue } from '../core/errors';
import { type TraceSpan } from '../core/trace';
import type { SignatureInput, SignatureOutput } from '../types/signature';

/**
 * Single-shot prediction against a signature.
 *
 * `TOutput` defaults to whatever can be inferred from the signature. A zod
 * signature carries its shape in the type system, so inference is exact and no
 * type argument is needed:
 *
 * ```ts
 * const review = await new Predict(AnalyzeReview).forward({ review: text });
 * review.sentiment; // 'positive' | 'negative' | 'neutral'
 * ```
 *
 * Decorated classes carry no per-field literal types at compile time, so that
 * inference yields `Record<string, any>`; supply `TOutput` for precise types:
 *
 * ```ts
 * type QAOutput = { answer: string; confidence: number };
 * const qa = new Predict<typeof AnswerQuestion, QAOutput>(AnswerQuestion);
 * ```
 *
 * Runtime validation always comes from the signature, whatever `TOutput` says.
 */
export class Predict<
    TSignature extends SignatureSource = typeof Signature,
    TOutput extends Record<string, any> = SignatureOutput<TSignature>,
> extends Module {
    constructor(signature: TSignature | string, lm?: ILanguageModel) {
        super(signature, lm);
    }

    async forward(
        inputs: SignatureInput<TSignature>,
        options?: LLMCallOptions
    ): Promise<Prediction<TOutput> & TOutput> {
        const prediction = await this.traced<TOutput>(inputs, async (span) => {
            const prompt = this.buildPrompt(inputs);
            return (await this.complete(prompt, options, span)) as TOutput;
        });

        return prediction as Prediction<TOutput> & TOutput;
    }

    /**
     * Run one completion and validate it against the signature.
     *
     * Uses the provider's native structured-output mode when it has one — that
     * constrains decoding rather than merely asking for JSON — and falls back to
     * parsing labelled text otherwise. Both paths end in the same validation.
     *
     * `span` is supplied when tracing is on, and records the call either way.
     */
    protected async complete(
        prompt: string,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();

        if (this.lm.getCapabilities().supportsStructuredOutput) {
            const schema = buildOutputJsonSchema(signature);
            span?.startCall(prompt);
            const raw = await this.lm.generateStructured<Record<string, any>>(
                prompt,
                schema,
                options
            );
            span?.endCall(JSON.stringify(raw));
            return this.validateStructured(raw);
        }

        span?.startCall(prompt);
        const rawOutput = await this.lm.generate(prompt, options);
        span?.endCall(rawOutput);
        return parseOutput(signature, rawOutput);
    }

    /** Validate a provider's structured response against the signature. */
    protected validateStructured(raw: Record<string, any>): Record<string, any> {
        const signature = this.requireSignature();
        // Optional fields are expressed as nullable in the JSON Schema, so strip
        // the nulls that stand in for absent values before validating.
        const cleaned = stripAbsentNulls(signature, raw);

        const result = buildOutputSchema(signature).safeParse(cleaned);
        if (result.success) {
            return result.data as Record<string, any>;
        }

        const fields = getOutputFieldConfigs(signature);
        const issues: FieldValidationIssue[] = result.error.issues.map((issue) => {
            const field = String(issue.path[0] ?? '(root)');
            return {
                field,
                expected: fields[field]?.type ?? 'string',
                received: cleaned[field],
                message: issue.message,
            };
        });
        throw new ValidationError(issues, JSON.stringify(raw));
    }

    protected requireSignature(): SignatureLike {
        if (!this.signature) {
            throw new Error('No signature provided');
        }
        return this.signature;
    }

    protected buildPrompt(inputs: Record<string, any>): string {
        return buildPrompt(this.requireSignature(), inputs);
    }

    protected parseOutput(rawOutput: string): Record<string, any> {
        return parseOutput(this.requireSignature(), rawOutput);
    }
}
