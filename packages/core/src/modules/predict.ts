import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { type Signature } from '../core/signature';
import { type Example } from '../core/example';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { parseOutput, buildPrompt } from '../utils/parsing';
import { buildOutputSchema, buildOutputJsonSchema } from '../utils/schema';
import { ValidationError, type FieldValidationIssue } from '../core/errors';
import type { SignatureOutput } from '../types/signature';

/** Construction options for {@link Predict} and its subclasses. */
export interface PredictOptions {
    /** Language model for this module. Defaults to the configured one. */
    lm?: ILanguageModel;
    /**
     * Worked examples rendered into the prompt before the real input. Usually
     * produced by an optimizer, but hand-written demos work just as well.
     */
    demos?: Example[];
}

/**
 * Work out whether the second constructor argument is a language model or an
 * options bag.
 *
 * `new Predict(Sig, lm)` predates `new Predict(Sig, { demos })` and both must
 * keep working, so the two are told apart structurally rather than by a marker
 * property a hand-rolled `ILanguageModel` would not have.
 *
 * An object carrying only one of `generate`/`chat` is rejected loudly rather
 * than quietly treated as options: falling through would drop the caller's model
 * and silently run against the globally configured one instead, and the only
 * symptom would be a surprising bill.
 */
function resolveOptions(value?: ILanguageModel | PredictOptions): PredictOptions {
    if (value === undefined || value === null) {
        return {};
    }
    if (typeof value !== 'object') {
        throw new Error(
            'Predict expects a language model or an options object as its second argument.'
        );
    }

    const candidate = value as Partial<ILanguageModel> & PredictOptions;
    const hasGenerate = typeof candidate.generate === 'function';
    const hasChat = typeof candidate.chat === 'function';

    if (hasGenerate && hasChat) {
        return { lm: candidate as ILanguageModel };
    }
    if (hasGenerate || hasChat) {
        throw new Error(
            `Predict was given an object with ${hasChat ? 'chat()' : 'generate()'} but not ` +
                `${hasChat ? 'generate()' : 'chat()'}. Implement ILanguageModel in full, or ` +
                'extend BaseLM, which supplies generate() for you.'
        );
    }

    return candidate;
}

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
    /** Worked examples prepended to every prompt this module builds. */
    protected demos: Example[] = [];

    constructor(signature: TSignature | string, lmOrOptions?: ILanguageModel | PredictOptions) {
        const options = resolveOptions(lmOrOptions);

        super(signature, options.lm);
        this.demos = [...(options.demos ?? [])];
    }

    /** The demos this module renders, as a copy. */
    getDemos(): Example[] {
        return [...this.demos];
    }

    /**
     * A copy of this module that renders `demos`.
     *
     * Returns a new module rather than mutating this one: an optimizer hands
     * back a compiled program while leaving the student it was given untouched,
     * so the same student can be compiled twice and compared.
     */
    withDemos(demos: Example[]): this {
        return this.cloneWith({ demos: [...demos] });
    }

    /**
     * A copy of this module that calls `lm`.
     *
     * This is what makes a teacher model possible: bootstrap the demos with a
     * stronger model, then attach them to the cheaper student.
     */
    withLM(lm: ILanguageModel): this {
        return this.cloneWith({ lm });
    }

    /**
     * Shallow-copy this module, preserving its concrete subclass, with some
     * fields replaced — so `ChainOfThought.withDemos()` returns a
     * `ChainOfThought`.
     *
     * Only own enumerable properties are carried over, which covers ordinary
     * public fields but not `#private` ones; a subclass using those should
     * override `withDemos`/`withLM` with its own copy constructor.
     */
    private cloneWith(patch: { demos?: Example[]; lm?: ILanguageModel }): this {
        const clone = Object.create(Object.getPrototypeOf(this)) as this;
        Object.assign(clone, this, patch);
        // Never share the demo array with the module we copied from, or pushing
        // to one module's demos would silently alter another's.
        (clone as Predict).demos = [...(clone as Predict).demos];
        return clone;
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
     * Run one completion and validate it against the signature.
     *
     * Uses the provider's native structured-output mode when it has one — that
     * constrains decoding rather than merely asking for JSON — and falls back to
     * parsing labelled text otherwise. Both paths end in the same validation.
     */
    protected async complete(
        prompt: string,
        options?: LLMCallOptions
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();

        if (this.lm.getCapabilities().supportsStructuredOutput) {
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
        // Demos must demonstrate the shape the reply will actually take. A
        // provider with native structured output has its decoding constrained to
        // JSON, so labelled `field: value` demos would be modelling a format the
        // model is not permitted to emit.
        const format = this.lm.getCapabilities().supportsStructuredOutput ? 'json' : 'labelled';
        return buildPrompt(this.requireSignature(), inputs, this.demos, { format });
    }

    protected parseOutput(rawOutput: string): Record<string, any> {
        return parseOutput(this.requireSignature(), rawOutput);
    }
}
