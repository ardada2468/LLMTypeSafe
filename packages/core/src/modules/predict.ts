import {
    Module,
    PREDICTOR,
    cloneModule,
    type PredictorState,
    type SerializedDemo,
} from '../core/module';
import type { Prediction } from '../core/prediction';
import { type Signature, type SignatureLike, type SignatureSource } from '../core/signature';
import { Example } from '../core/example';
import type { ChatMessage, ILanguageModel, LLMCallOptions } from '../types/language-model';
import { type TraceSpan } from '../core/trace';
import { providerOptions, startLMCall } from '../core/lm-call';
import { type Adapter, type AdapterCallInput, renderTranscript } from '../adapters/base';
import { HINT_FIELD, predictorHint, withHintField } from '../core/hints';
import {
    AsyncChannel,
    activeStreamListeners,
    consumerGoneReason,
    runWithStreamListener,
    type StreamListener,
} from '../core/stream-context';
import { resolveAdapter } from '../adapters/select';
import type { SignatureInput, SignatureOutput } from '../types/signature';

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
    /**
     * Prompt format for this module, overriding the configured adapter. A
     * call's own `adapter` option still wins.
     */
    adapter?: Adapter;
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

    /** This module's adapter, or `undefined` to use the configured or default one. */
    protected adapter?: Adapter;

    constructor(signature: TSignature | string, lmOrOptions?: ILanguageModel | PredictOptions) {
        const options = resolveOptions(lmOrOptions);

        super(signature, options.lm);
        this.demos = [...(options.demos ?? [])];
        this.instructions = options.instructions;
        this.adapter = options.adapter;
    }

    /**
     * The signature this module actually sends — for a `ChainOfThought`, the
     * signature with its leading `reasoning` output. Optimizers read it to
     * describe a program's steps and to resolve a predictor's current
     * instructions.
     */
    getSignature(): SignatureLike {
        return this.predictionSignature();
    }

    /** The adapter this module was given, or `undefined` when it uses the configured one. */
    getAdapter(): Adapter | undefined {
        return this.adapter;
    }

    /** A copy of this module that formats its prompts with `adapter`. */
    withAdapter(adapter: Adapter): this {
        return this.cloneWith({ adapter });
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
        adapter?: Adapter;
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
        const prediction = await this.traced<TOutput>(
            inputs,
            async (span) => (await this.predict(inputs, options, span)) as TOutput
        );

        return prediction as Prediction<TOutput> & TOutput;
    }

    /**
     * The signature this module actually sends. `Predict` sends its own;
     * `ChainOfThought` adds a leading `reasoning` output.
     */
    protected predictionSignature(): SignatureLike {
        return this.requireSignature();
    }

    /**
     * The adapter for one call: the call's `adapter` option, then this
     * module's, then the configured one, then the default for `lm`.
     */
    protected resolveAdapter(lm: ILanguageModel, options?: LLMCallOptions): Adapter {
        return resolveAdapter(lm, options?.adapter, this.adapter);
    }

    /**
     * One prediction through the adapter: format, call the model, parse and
     * validate, repairing up to `options.repairAttempts` times.
     *
     * Two things in the surrounding scope change how the call is made, without
     * the caller having to thread anything through a hand-written `forward`:
     *
     * - a hint from `Refine` for this predictor adds a `hint_` input carrying
     *   the advice from an earlier attempt;
     * - a stream listener (`Predict.stream`, `streamify`) makes the call
     *   stream, reporting partial outputs as they arrive, when the model and
     *   adapter can.
     */
    protected predict(
        inputs: Record<string, any>,
        options?: LLMCallOptions,
        span?: TraceSpan
    ): Promise<Record<string, any>> {
        const lm = this.resolveLM(options);
        const adapter = this.resolveAdapter(lm, options);

        let signature = this.predictionSignature();
        let callInputs = inputs;
        const hint = predictorHint(this);
        if (hint !== undefined) {
            signature = withHintField(signature);
            callInputs = { ...inputs, [HINT_FIELD]: hint };
        }

        const input: AdapterCallInput = {
            signature,
            demos: this.demos,
            inputs: callInputs,
            instructions: this.instructions,
            options,
            span,
            moduleId: this.moduleId,
        };

        const listeners = activeStreamListeners();
        if (listeners.length === 0) return adapter.call(lm, input);
        if (this.canStream(options) && adapter.parsePartial) {
            return this.predictStreaming(lm, adapter, input, listeners);
        }
        // Listening, but this model or adapter cannot stream: one ordinary
        // call, reported as complete so listeners still see its fields.
        return adapter.call(lm, input).then((outputs) => {
            notifyComplete(listeners, this, {}, outputs);
            return outputs;
        });
    }

    /**
     * The streaming form of {@link predict}: stream the reply, reporting each
     * new partial snapshot to `listeners` (and waiting for them, which is the
     * back-pressure that lets a consumer that stops reading stop the provider
     * stream too), then parse, validate and — on request — repair the full
     * reply exactly as a non-streaming call would.
     */
    private async predictStreaming(
        lm: ILanguageModel,
        adapter: Adapter,
        input: AdapterCallInput,
        listeners: readonly StreamListener[]
    ): Promise<Record<string, any>> {
        // The innermost listener's signal, for a call whose options carried none.
        const signal =
            input.options?.signal ??
            [...listeners].reverse().find((listener) => listener.signal)?.signal;
        throwIfAborted(signal);

        const messages = adapter.formatMessages(input);
        const invocation = {};

        let buffer = '';
        let lastSnapshot = '';
        for await (const delta of this.streamChunks(lm, messages, input, signal)) {
            buffer += delta;
            const snapshot = adapter.parsePartial!(input.signature, buffer);
            const key = snapshotKey(snapshot);
            if (key !== lastSnapshot && Object.keys(snapshot).length > 0) {
                lastSnapshot = key;
                await Promise.all(
                    listeners.map((listener) =>
                        listener.onPartial({ predictor: this, invocation, snapshot })
                    )
                );
            }
        }

        throwIfAborted(signal);

        const outputs = await adapter.completeFrom(lm, messages, buffer, input);
        notifyComplete(listeners, this, invocation, outputs);
        return outputs;
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
     * The adapter formats the prompt and reads each partial reply with its
     * `parsePartial` — `[[ ## field ## ]]` sections for `ChatAdapter`, a
     * truncated JSON object for `JSONAdapter`. The last yielded snapshot is
     * the complete output, validated against the signature exactly as
     * {@link forward} validates it — a stream that ends in something the
     * signature rejects still throws a `ValidationError`, from the final
     * `next()`. The generator's return value is the `Prediction` wrapper, for
     * callers who iterate by hand.
     *
     * Under the hood this *is* {@link forward}, run with a stream listener in
     * scope: the invocation is traced, reported to module callbacks, has its
     * usage collected, and is repaired on request (`repairAttempts`) exactly
     * like a non-streaming call. Snapshots are handed over with back-pressure,
     * so breaking out of the loop stops reading from the provider and closes
     * its stream.
     *
     * Falls back to a single non-streaming call, yielded once, when the model
     * does not support streaming or the adapter cannot parse partial replies.
     * A provider's native structured-output mode is not wired into its chat
     * stream, so on the streaming path the JSON schema travels in the prompt.
     */
    async *stream(
        inputs: Record<string, any>,
        options?: StreamOptions
    ): PredictionStream<TOutput> {
        const userSignal = options?.signal;
        throwIfAborted(userSignal);

        // One signal for both cancellations: the caller's, and this generator
        // being closed early by a consumer that stopped reading.
        const controller = new AbortController();
        const onAbort = () => controller.abort(abortReason(userSignal!));
        userSignal?.addEventListener('abort', onAbort, { once: true });

        const channel = new AsyncChannel<PartialOutput<TOutput>>();
        const listener: StreamListener = {
            signal: controller.signal,
            onPartial: (event) =>
                event.predictor === this
                    ? channel.push({ ...event.snapshot } as PartialOutput<TOutput>)
                    : undefined,
        };

        let prediction: (Prediction<TOutput> & TOutput) | undefined;
        let settled = false;
        const task = runWithStreamListener(listener, () =>
            this.forward(inputs as SignatureInput<TSignature>, {
                ...options,
                signal: controller.signal,
            })
        ).then(
            (result) => {
                settled = true;
                prediction = result;
                channel.end();
            },
            (error: unknown) => {
                settled = true;
                channel.end({ error });
            }
        );

        try {
            let lastSnapshot = '';
            for (;;) {
                const next = await channel.next();
                if (next.done) break;
                lastSnapshot = snapshotKey(next.value);
                yield next.value;
            }

            // Validation coerces, so the final output usually differs from the
            // last partial one; when it does not, one yield stands for both.
            const final = prediction!.toObject() as PartialOutput<TOutput>;
            if (snapshotKey(final) !== lastSnapshot) {
                yield final;
            }
            return prediction!;
        } finally {
            // Abort before releasing the producer, so the read it resumes into
            // sees the cancellation instead of pulling another chunk.
            if (!settled) controller.abort(consumerGoneReason());
            channel.cancel();
            await task;
            userSignal?.removeEventListener('abort', onAbort);
        }
    }

    /**
     * Yield the text of each streamed chunk, closing the provider's stream
     * however the loop ends — consumer `break`, abort, or error.
     */
    private async *streamChunks(
        lm: ILanguageModel,
        messages: ChatMessage[],
        input: AdapterCallInput,
        signal?: AbortSignal
    ): AsyncGenerator<string, void, void> {
        const callOptions = providerOptions(input.options);
        const call = startLMCall({
            lm,
            method: 'chatStream',
            traceText: renderTranscript(messages),
            messages,
            options: callOptions,
            span: input.span,
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

    protected requireSignature(): SignatureLike {
        if (!this.signature) {
            throw new Error('No signature provided');
        }
        return this.signature;
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

/** Report a finished prediction to every listener. Listeners observe; they cannot fail it. */
function notifyComplete(
    listeners: readonly StreamListener[],
    predictor: object,
    invocation: object,
    outputs: Record<string, any>
): void {
    for (const listener of listeners) {
        try {
            listener.onComplete?.({ predictor, invocation, outputs: { ...outputs } });
        } catch {
            // Deliberately swallowed.
        }
    }
}

/** A key for comparing two snapshots, insensitive to the order fields arrived in. */
function snapshotKey(snapshot: Record<string, any>): string {
    const entries = Object.entries(snapshot).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return JSON.stringify(entries);
}
