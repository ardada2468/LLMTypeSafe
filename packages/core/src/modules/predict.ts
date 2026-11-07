import {
    Module,
    PREDICTOR,
    cloneModule,
    type PredictorState,
    type SerializedDemo,
} from '../core/module';
import { parseJsonResponse } from '../core/base-lm';
import { Prediction } from '../core/prediction';
import { type Signature, type SignatureLike, type SignatureSource } from '../core/signature';
import { Example } from '../core/example';
import { contentToText, textPart } from '../utils/content';
import type {
    ChatMessage,
    ContentPart,
    ILanguageModel,
    LLMCallOptions,
    MessageContent,
} from '../types/language-model';
import {
    parseOutput,
    buildPrompt,
    buildPromptContent,
    type PromptOptions,
} from '../utils/parsing';
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
import { invokeLM, providerOptions, startLMCall } from '../core/lm-call';
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
    /**
     * Task instructions, replacing the signature's own description in the
     * prompt. Optimizers that rewrite instructions set this; the signature is
     * left alone.
     */
    instructions?: string;
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

    /**
     * Instructions overriding the signature's description in the prompt, or
     * `undefined` to use the description. Optimizable state, like `demos`:
     * saved by `dumpState()` and cleared by `resetCopy()`.
     */
    protected instructions?: string;

    constructor(signature: TSignature | string, lmOrOptions?: ILanguageModel | PredictOptions) {
        const options = resolveOptions(lmOrOptions);

        super(signature, options.lm);
        this.demos = [...(options.demos ?? [])];
        this.instructions = options.instructions;
    }

    /** The demos this module renders, as a copy. */
    getDemos(): Example[] {
        return [...this.demos];
    }

    /**
     * Replace this module's demos **in place**. Prefer {@link withDemos}, which
     * leaves this module untouched; this exists for loading saved state and
     * for optimizers that tune a copy they already own.
     */
    setDemos(demos: Example[]): void {
        this.demos = [...demos];
    }

    /** The instruction override, or `undefined` when the signature's description is used. */
    getInstructions(): string | undefined {
        return this.instructions;
    }

    /** Set (or with `undefined`, clear) the instruction override **in place**. */
    setInstructions(instructions: string | undefined): void {
        this.instructions = instructions;
    }

    /**
     * A copy of this module whose prompt opens with `instructions` instead of
     * the signature's description. Like {@link withDemos}, it leaves this
     * module untouched.
     */
    withInstructions(instructions: string): this {
        return this.cloneWith({ instructions });
    }

    /** This predictor's optimizable state, as plain JSON. Used by `Module.dumpState`. */
    dumpPredictorState(): PredictorState {
        const demos: SerializedDemo[] = this.demos.map((demo) => {
            const inputKeys = demo.getInputKeys();
            // Round-tripped through JSON so the saved state is exactly what a
            // file would hold, with no live references into the demo.
            const data = JSON.parse(JSON.stringify(demo.toObject())) as Record<string, unknown>;
            return inputKeys ? { data, inputKeys } : { data };
        });
        return this.instructions !== undefined
            ? { demos, instructions: this.instructions }
            : { demos };
    }

    /** Restore state from {@link dumpPredictorState}, in place. Used by `Module.loadState`. */
    loadPredictorState(state: PredictorState): void {
        this.demos = state.demos.map((demo) => {
            const example = new Example(demo.data);
            return demo.inputKeys ? example.withInputs(...demo.inputKeys) : example;
        });
        this.instructions = state.instructions;
    }

    /** Clear demos and the instruction override, in place. Used by `Module.resetCopy`. */
    resetPredictorState(): void {
        this.demos = [];
        this.instructions = undefined;
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
     * Own properties are carried over, which covers ordinary public fields but
     * not `#private` ones; a subclass using those should override
     * `withDemos`/`withLM` with its own copy constructor.
     */
    private cloneWith(patch: {
        demos?: Example[];
        lm?: ILanguageModel;
        instructions?: string;
    }): this {
        const clone = cloneModule(this, patch);
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
            const prompt = this.buildPromptContent(inputs, options);
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
        const prompt = this.buildPrompt(inputs, options);
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
        const structured = this.resolveLM(options).getCapabilities().supportsStructuredOutput;

        if (!this.canStream(options)) {
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
        const lm = this.resolveLM(options);
        const callOptions = providerOptions(options);
        const call = startLMCall({
            lm,
            method: 'chatStream',
            traceText: prompt,
            messages,
            options: callOptions,
            moduleId: this.moduleId,
        });
        const stream = call
            ? call.run(() => lm.chatStream!(messages, callOptions))
            : lm.chatStream!(messages, callOptions);

        let reading = false;
        let received = '';
        let settled = false;
        try {
            for (;;) {
                throwIfAborted(signal);

                reading = true;
                // Raced rather than awaited outright: a signal must be able to cut
                // short a read that is waiting on a token which may never come.
                // Each read runs in the call's scope so usage the provider
                // records mid-stream is attributed to this call.
                const next = call ? call.run(() => stream.next()) : stream.next();
                const chunk = await raceAbort(next, signal);
                reading = false;

                if (chunk.done) {
                    break;
                }
                if (chunk.value.content) {
                    received += chunk.value.content;
                    yield chunk.value.content;
                }
            }
            settled = true;
            call?.end(received, received);
        } catch (error) {
            settled = true;
            call?.fail(error);
            throw error;
        } finally {
            // A consumer that stopped early still ends the call, with what it got.
            if (!settled) call?.end(received, received);
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
    protected canStream(options?: LLMCallOptions): boolean {
        const lm = this.resolveLM(options);
        return lm.getCapabilities().supportsStreaming && typeof lm.chatStream === 'function';
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
        prompt: MessageContent,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const repairAttempts = normaliseRepairAttempts(options?.repairAttempts);
        const structured = this.resolveLM(options).getCapabilities().supportsStructuredOutput;
        let attemptPrompt: MessageContent = prompt;
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
                const format = structured ? 'structured' : 'text';
                attemptPrompt =
                    typeof prompt === 'string'
                        ? buildRepairPrompt(prompt, error, format)
                        : // Rebuilding a content prompt would re-send the image on
                          // every attempt. Appending the correction as a trailing
                          // text part keeps the parts, and their order, intact.
                          [...prompt, textPart(`\n${buildRepairPrompt('', error, format)}`)];
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
        prompt: MessageContent,
        structured: boolean,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();

        // A prompt carrying images has to travel as chat content: neither
        // `generate` nor `generateStructured` takes anything but a string, and
        // flattening would reduce the image to its placeholder.
        if (typeof prompt !== 'string') {
            return this.completeFromContent(prompt, structured, options, span);
        }

        const lm = this.resolveLM(options);
        const callOptions = providerOptions(options);

        if (structured) {
            const schema = buildOutputJsonSchema(signature);
            const raw = await invokeLM(
                {
                    lm,
                    method: 'generateStructured',
                    traceText: prompt,
                    prompt,
                    options: callOptions,
                    span,
                    moduleId: this.moduleId,
                },
                () => lm.generateStructured<Record<string, any>>(prompt, schema, callOptions),
                (result) => JSON.stringify(result)
            );
            return this.validateStructured(raw);
        }

        const rawOutput = await this.generateText(prompt, options, span);
        return parseOutput(signature, rawOutput);
    }

    /**
     * One instrumented plain-text `generate` call against the resolved model.
     * Shared by every subclass that needs free text, such as the reasoning
     * step of `ChainOfThought`.
     */
    protected generateText(
        prompt: string,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<string> {
        const lm = this.resolveLM(options);
        const callOptions = providerOptions(options);
        return invokeLM(
            {
                lm,
                method: 'generate',
                traceText: prompt,
                prompt,
                options: callOptions,
                span,
                moduleId: this.moduleId,
            },
            () => lm.generate(prompt, callOptions),
            (text) => text
        );
    }

    /**
     * The multimodal path: one `chat` turn carrying text and image parts.
     *
     * Native structured output is unavailable here because the provider methods
     * that constrain decoding take a string prompt, so the schema is requested
     * in the prompt instead — the same fallback `BaseLM.generateStructured` uses
     * for providers without a JSON-schema mode. Validation is identical either
     * way.
     */
    protected async completeFromContent(
        content: ContentPart[],
        structured: boolean,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();
        const parts = structured ? withSchemaInstruction(content, signature) : content;

        const lm = this.resolveLM(options);
        const callOptions = providerOptions(options);
        const messages: ChatMessage[] = [{ role: 'user', content: parts }];
        const rawOutput = await invokeLM(
            {
                lm,
                method: 'chat',
                traceText: contentToText(parts),
                messages,
                options: callOptions,
                span,
                moduleId: this.moduleId,
            },
            () => lm.chat(messages, callOptions),
            (text) => text
        );

        if (!structured) {
            return parseOutput(signature, rawOutput);
        }
        return this.validateStructured(extractJsonObject(rawOutput));
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

    /**
     * The prompt as chat content: a plain string when every input is text, and
     * content parts when a field is declared `image`, so the image reaches the
     * provider instead of the `[image: …]` placeholder a string is limited to.
     */
    protected buildPromptContent(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): MessageContent {
        return buildPromptContent(
            this.requireSignature(),
            inputs,
            this.demos,
            this.promptOptions(options)
        );
    }

    protected buildPrompt(inputs: Record<string, any>, options?: LLMCallOptions): string {
        return buildPrompt(
            this.requireSignature(),
            inputs,
            this.demos,
            this.promptOptions(options)
        );
    }

    private promptOptions(options?: LLMCallOptions): PromptOptions {
        // Demos must demonstrate the shape the reply will actually take. A
        // provider with native structured output has its decoding constrained to
        // JSON, so labelled `field: value` demos would be modelling a format the
        // model is not permitted to emit.
        const structured = this.resolveLM(options).getCapabilities().supportsStructuredOutput;
        return { format: structured ? 'json' : 'labelled', instructions: this.instructions };
    }

    protected parseOutput(rawOutput: string): Record<string, any> {
        return parseOutput(this.requireSignature(), rawOutput);
    }
}

// Marks every Predict (and subclass) as a predictor for `namedPredictors()`.
Object.defineProperty(Predict.prototype, PREDICTOR, { value: true });

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

/** Append the JSON-schema request as a trailing text part. */
function withSchemaInstruction(
    content: ContentPart[],
    signature: SignatureLike
): ContentPart[] {
    const schema = buildOutputJsonSchema(signature);
    return [
        ...content,
        textPart(
            `\n\nRespond with JSON matching this schema. ` +
                `Output only the JSON object, with no surrounding prose or code fences.\n` +
                `${JSON.stringify(schema, null, 2)}`
        ),
    ];
}

/** Read a JSON object out of a reply, tolerating code fences and stray prose. */
function extractJsonObject(raw: string): Record<string, any> {
    return parseJsonResponse<Record<string, any>>(
        raw,
        () =>
            new ValidationError(
                [
                    {
                        field: '(root)',
                        expected: 'object',
                        received: raw,
                        message: 'model did not return a JSON object',
                    },
                ],
                raw
            )
    );
}
