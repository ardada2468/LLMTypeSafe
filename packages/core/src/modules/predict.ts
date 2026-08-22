import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { type Signature } from '../core/signature';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { parseOutput, buildPrompt } from '../utils/parsing';
import { buildOutputSchema, buildOutputJsonSchema } from '../utils/schema';
import { ValidationError, type FieldValidationIssue } from '../core/errors';
import { buildRepairPrompt, isRepeatedFailure, normaliseRepairAttempts } from '../core/repair';
import type { SignatureOutput } from '../types/signature';

/**
 * Single-shot prediction against a signature.
 *
 * `TOutput` defaults to whatever can be inferred from the signature. Decorated
 * classes carry no per-field literal types at compile time, so that inference
 * yields `Record<string, any>`; supply `TOutput` when you want precise types:
 *
 * ```ts
 * type QAOutput = { answer: string; confidence: number };
 * const qa = new Predict<typeof AnswerQuestion, QAOutput>(AnswerQuestion);
 * ```
 *
 * Runtime validation always comes from the signature, whatever `TOutput` says.
 */
export class Predict<
    TSignature extends typeof Signature = typeof Signature,
    TOutput extends Record<string, any> = SignatureOutput<TSignature>,
> extends Module {
    constructor(signature: TSignature | string, lm?: ILanguageModel) {
        super(signature, lm);
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction<TOutput> & TOutput> {
        const prompt = this.buildPrompt(inputs);
        const parsed = (await this.complete(prompt, options)) as TOutput;

        return new Prediction(parsed) as Prediction<TOutput> & TOutput;
    }

    /**
     * Run a completion, validate it against the signature, and optionally spend
     * `options.repairAttempts` further round-trips fixing a response that fails.
     *
     * Uses the provider's native structured-output mode when it has one — that
     * constrains decoding rather than merely asking for JSON — and falls back to
     * parsing labelled text otherwise. Both paths end in the same validation, so
     * both are repairable.
     */
    protected async complete(
        prompt: string,
        options?: LLMCallOptions
    ): Promise<Record<string, any>> {
        const repairAttempts = normaliseRepairAttempts(options?.repairAttempts);
        const structured = this.lm.getCapabilities().supportsStructuredOutput;
        let attemptPrompt = prompt;
        let previousError: ValidationError | undefined;

        for (let attempt = 0; ; attempt++) {
            try {
                return await this.completeOnce(attemptPrompt, structured, options);
            } catch (error) {
                if (!(error instanceof ValidationError) || attempt >= repairAttempts) {
                    throw error;
                }
                // An identical failure would produce an identical repair prompt, so
                // every remaining attempt is guaranteed to fail the same way against
                // a deterministic model. Stop rather than bill for the repeats.
                if (previousError && isRepeatedFailure(previousError, error)) {
                    throw error;
                }
                previousError = error;

                // Repair from the original prompt, not the previous repair prompt,
                // so successive attempts do not stack up every earlier correction.
                attemptPrompt = buildRepairPrompt(
                    prompt,
                    error,
                    structured ? 'structured' : 'text'
                );
            }
        }
    }

    /** One completion plus validation, with no repair loop around it. */
    protected async completeOnce(
        prompt: string,
        structured: boolean,
        options?: LLMCallOptions
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();

        if (structured) {
            const schema = buildOutputJsonSchema(signature);
            const raw = await this.lm.generateStructured<Record<string, any>>(
                prompt,
                schema,
                options
            );
            return this.validateStructured(raw);
        }

        const rawOutput = await this.lm.generate(prompt, options);
        return parseOutput(signature, rawOutput);
    }

    /** Validate a provider's structured response against the signature. */
    protected validateStructured(raw: Record<string, any>): Record<string, any> {
        const signature = this.requireSignature();
        // Optional fields are expressed as nullable in the JSON Schema, so strip
        // nulls before validating rather than failing on them.
        const cleaned: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(raw ?? {})) {
            if (value !== null) {
                cleaned[key] = value;
            }
        }

        const result = buildOutputSchema(signature).safeParse(cleaned);
        if (result.success) {
            return result.data as Record<string, any>;
        }

        const fields = typeof signature === 'string' ? {} : signature.getOutputFields();
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

    protected requireSignature(): typeof Signature | string {
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
