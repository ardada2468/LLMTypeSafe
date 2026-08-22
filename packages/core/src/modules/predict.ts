import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { type Signature, type SignatureLike, type SignatureSource } from '../core/signature';
import { type Example } from '../core/example';
import type { ChatMessage, ILanguageModel, LLMCallOptions } from '../types/language-model';
import { parseOutput, buildPrompt } from '../utils/parsing';
import {
    buildOutputSchema,
    buildOutputJsonSchema,
    getOutputFieldConfigs,
    stripAbsentNulls,
} from '../utils/schema';
import { parsePartialJson } from '../utils/partial-json';
import { ValidationError, type FieldValidationIssue } from '../core/errors';
import { buildRepairPrompt, isRepeatedFailure, normaliseRepairAttempts } from '../core/repair';
import { type TraceSpan } from '../core/trace';
import type { FieldConfig, SignatureInput, SignatureOutput } from '../types/signature';

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

/** Options for {@link Predict.stream}, on top of the usual call options. */
export interface StreamOptions extends LLMCallOptions {
    /**
     * Cancels the stream. The generator closes the underlying provider stream
     * and rejects with the signal's reason.
     */
    signal?: AbortSignal;
}

/**
 * A snapshot of the output fields parsed so far.
 *
 * Every field is optional, because it may not have arrived yet, and a field
 * declared as something other than a string may still be the raw text the model
 * is part-way through writing: `confidence` is `'0.'` before it is `0.95`.
 * Coercion belongs to validation, which happens once at the end, so only the
 * final snapshot is guaranteed to match the signature's declared types.
 */
export type PartialOutput<TOutput> = {
    [K in keyof TOutput]?: TOutput[K] | string;
};

/**
 * What {@link Predict.stream} produces.
 *
 * Each `yield` is a {@link PartialOutput} snapshot, so a field fills in as its
 * tokens arrive. The **last yield** is the complete, validated output, and the
 * generator's **return value** is the validated {@link Prediction} — reachable
 * by driving `next()` by hand, since `for await` discards return values.
 */
export type PredictionStream<TOutput extends Record<string, any>> = AsyncGenerator<
    PartialOutput<TOutput>,
    Prediction<TOutput> & TOutput,
    void
>;
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
     * Run the prediction as a stream of progressively-filled output fields.
     *
     * ```ts
     * for await (const partial of qa.stream({ question })) {
     *     render(partial.answer); // grows as tokens arrive
     * }
     * ```
     *
     * The last yielded snapshot is the complete output, validated against the
     * signature exactly as {@link forward} validates it — a stream that ends
     * in something the signature rejects still throws a `ValidationError`, from
     * the final `next()`. The generator's return value is the `Prediction`
     * wrapper, for callers who iterate by hand.
     *
     * Falls back to a single non-streaming call, yielded once, when the model
     * does not support streaming.
     */
    async *stream(
        inputs: Record<string, any>,
        options?: StreamOptions
    ): PredictionStream<TOutput> {
        const prompt = this.buildPrompt(inputs);
        const parsed = (yield* this.streamComplete(prompt, options)) as TOutput;

        return new Prediction(parsed) as Prediction<TOutput> & TOutput;
    }

    /**
     * Stream one completion, yielding each new snapshot of the output fields and
     * returning the validated result.
     *
     * Mirrors {@link complete}'s two paths. With native structured output the
     * buffer is JSON, read by {@link parsePartialJson}; otherwise it is labelled
     * text, read by the same heuristics {@link parseOutput} uses, relaxed so a
     * half-written buffer yields what it has instead of throwing.
     */
    protected async *streamComplete(
        prompt: string,
        options?: StreamOptions
    ): AsyncGenerator<PartialOutput<TOutput>, Record<string, any>, void> {
        throwIfAborted(options?.signal);

        const signature = this.requireSignature();
        const structured = this.lm.getCapabilities().supportsStructuredOutput;

        if (!this.canStream()) {
            const parsed = await this.complete(prompt, options);
            yield { ...parsed } as PartialOutput<TOutput>;
            return parsed;
        }

        // A provider's native structured-output mode is not wired into its chat
        // stream, so the schema has to travel in the prompt on this path.
        const streamPrompt = structured ? this.buildStructuredPrompt(prompt) : prompt;
        const labels = structured ? [] : Object.keys(getOutputFieldConfigs(signature));
        const lenient = structured ? undefined : lenientOutputView(signature);

        let buffer = '';
        let lastSnapshot = '';

        for await (const delta of this.streamChunks(streamPrompt, options)) {
            buffer += delta;

            if (!structured && isPartialLabel(buffer, labels)) {
                // The buffer so far is the beginning of a field's own label, and
                // showing "answ" as the answer would be worse than showing nothing.
                continue;
            }

            const snapshot = structured
                ? readPartialJsonFields(buffer)
                : parseOutput(lenient!, buffer);

            const key = snapshotKey(snapshot);
            if (key !== lastSnapshot && Object.keys(snapshot).length > 0) {
                lastSnapshot = key;
                yield snapshot as PartialOutput<TOutput>;
            }
        }

        throwIfAborted(options?.signal);

        const parsed = structured
            ? this.validateStructured(readPartialJsonFields(buffer))
            : parseOutput(signature, buffer);

        // Validation coerces, so the final snapshot usually differs from the last
        // partial one; when it does not, one yield stands for both.
        const final = { ...parsed };
        if (snapshotKey(final) !== lastSnapshot) {
            yield final as PartialOutput<TOutput>;
        }
        return parsed;
    }

    /**
     * Yield the text of each streamed chunk, closing the provider's stream
     * however the loop ends — consumer `break`, abort, or error.
     */
    private async *streamChunks(
        prompt: string,
        options?: StreamOptions
    ): AsyncGenerator<string, void, void> {
        const messages: ChatMessage[] = [{ role: 'user', content: prompt }];
        const signal = options?.signal;
        const stream = this.lm.chatStream!(messages, options);

        let reading = false;
        try {
            for (;;) {
                throwIfAborted(signal);

                reading = true;
                // Raced rather than awaited outright: a signal must be able to cut
                // short a read that is waiting on a token which may never come.
                const chunk = await raceAbort(stream.next(), signal);
                reading = false;

                if (chunk.done) {
                    break;
                }
                if (chunk.value.content) {
                    yield chunk.value.content;
                }
            }
        } finally {
            // `return()` queues behind a read still in flight, so it is only
            // awaited when nothing is pending — otherwise closing an abandoned
            // stream would block the caller on that same stalled read.
            const closing = Promise.resolve(stream.return?.(undefined)).catch(() => undefined);
            if (!reading) {
                await closing;
            }
        }
    }

    /** Whether this model can actually stream: `chatStream` is optional. */
    protected canStream(): boolean {
        return (
            this.lm.getCapabilities().supportsStreaming &&
            typeof this.lm.chatStream === 'function'
        );
    }

    /** Ask for the signature's JSON schema in the prompt, as `BaseLM` does. */
    protected buildStructuredPrompt(prompt: string): string {
        const schema = buildOutputJsonSchema(this.requireSignature());
        return (
            `${prompt}\n\nRespond with JSON matching this schema. ` +
            `Output only the JSON object, with no surrounding prose or code fences.\n` +
            `${JSON.stringify(schema, null, 2)}`
        );
    }

    /**
     * Run a completion, validate it against the signature, and optionally spend
     * `options.repairAttempts` further round-trips fixing a response that fails.
     *
     * Uses the provider's native structured-output mode when it has one — that
     * constrains decoding rather than merely asking for JSON — and falls back to
     * parsing labelled text otherwise. Both paths end in the same validation, so
     * both are repairable.
     *
     * `span` is supplied when tracing is on, and records the call either way.
     */
    protected async complete(
        prompt: string,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const repairAttempts = normaliseRepairAttempts(options?.repairAttempts);
        const structured = this.lm.getCapabilities().supportsStructuredOutput;
        let attemptPrompt = prompt;
        let previousError: ValidationError | undefined;

        for (let attempt = 0; ; attempt++) {
            try {
                return await this.completeOnce(attemptPrompt, structured, options, span);
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

    /**
     * One completion plus validation, with no repair loop around it.
     *
     * Each repair attempt reports its own call to `span`, so a trace shows every
     * round trip that was paid for rather than only the one that succeeded.
     */
    protected async completeOnce(
        prompt: string,
        structured: boolean,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();

        if (structured) {
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

/** Throw the signal's reason if the caller has cancelled. */
export function throwIfAborted(signal?: AbortSignal): void {
    if (!signal?.aborted) {
        return;
    }
    throw abortReason(signal);
}

function abortReason(signal: AbortSignal): unknown {
    return signal.reason ?? Object.assign(new Error('Stream aborted'), { name: 'AbortError' });
}

/** Settle with `promise`, or reject as soon as `signal` aborts — whichever is first. */
function raceAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
    if (!signal) {
        return promise;
    }

    return new Promise<T>((resolve, reject) => {
        const onAbort = () => reject(abortReason(signal));
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(resolve, reject).finally(() => {
            signal.removeEventListener('abort', onAbort);
        });
    });
}

/**
 * Whether the whole buffer is still the beginning of a field's own label.
 *
 * `parseOutput` treats an unlabelled reply to a single-output signature as the
 * value itself, which is right for a model that answers bare and wrong for the
 * first few tokens of `answer: …`. Suppressing the snapshot until the label is
 * settled costs one chunk and avoids showing text that then jumps backwards.
 */
function isPartialLabel(buffer: string, fieldNames: string[]): boolean {
    const seen = buffer.trimStart().toLowerCase();
    if (seen === '') {
        return true;
    }
    return fieldNames.some((name) => `${name.toLowerCase()}:`.startsWith(seen));
}

/** A key for comparing two snapshots, insensitive to the order fields arrived in. */
function snapshotKey(snapshot: Record<string, any>): string {
    const entries = Object.entries(snapshot).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return JSON.stringify(entries);
}

/**
 * A view of a signature's outputs with every field optional and untyped.
 *
 * `parseOutput` throws on a missing or uncoercible field, which is right at the
 * end of a stream and wrong in the middle of one. Relaxing the fields lets the
 * same extraction heuristics run over a half-written buffer and return whatever
 * is legible so far; the real signature validates the final text.
 */
function lenientOutputView(signature: SignatureLike): typeof Signature {
    const fields: Record<string, FieldConfig> = {};
    for (const [name, config] of Object.entries(getOutputFieldConfigs(signature))) {
        fields[name] = { ...config, type: 'string', required: false };
    }

    // `getOutputFields` is the only member the parsing path consults.
    return { getOutputFields: () => fields } as unknown as typeof Signature;
}

/** Read the top-level object of a truncated JSON buffer, minus null fields. */
function readPartialJsonFields(buffer: string): Record<string, any> {
    const value = parsePartialJson<unknown>(buffer);
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
        return {};
    }

    const fields: Record<string, any> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
        // Optional fields are nullable in the JSON Schema; a null is not a value
        // to show, it is the model declining to fill the field in.
        if (entry !== null) {
            fields[key] = entry;
        }
    }
    return fields;
}
