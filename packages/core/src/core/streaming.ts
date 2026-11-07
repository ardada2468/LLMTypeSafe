import { context, getCallbacks } from './config';
import {
    type Callback,
    type LMEndEvent,
    type LMStartEvent,
    type ModuleEndEvent,
    type ModuleStartEvent,
    type ToolEndEvent,
    type ToolStartEvent,
} from './callbacks';
import { type Module } from './module';
import { type Prediction } from './prediction';
import {
    AsyncChannel,
    consumerGoneReason,
    runWithStreamListener,
    type StreamListener,
} from './stream-context';
import type { LLMCallOptions } from '../types/language-model';

/** A progress message, from the {@link StatusMessageProvider}. */
export interface StatusStreamEvent {
    type: 'status';
    message: string;
}

/**
 * Part of one output field of one predictor, as it streams in.
 *
 * `value` is the field's whole value so far — uncoerced text until the
 * predictor finishes. `delta` is what was appended since the previous event
 * for the same field, present when the value grew by appending (as text
 * does); a value that changed some other way carries only `value`. The final
 * event for each field has `done: true` and the validated value.
 */
export interface FieldStreamEvent {
    type: 'field';
    /**
     * The predictor's path in the program, as `namedPredictors()` reports it
     * (`self` for a bare predictor), or its `moduleId` when it is not part of
     * the program's tree — a copy made by `BestOfN`, say.
     */
    path: string;
    field: string;
    value: unknown;
    delta?: string;
    done?: true;
}

/** The program's final output — the last event, unless `includeFinal` is off. */
export interface PredictionStreamEvent {
    type: 'prediction';
    prediction: Prediction;
}

export type StreamEvent = StatusStreamEvent | FieldStreamEvent | PredictionStreamEvent;

/**
 * Turns callback events into status messages for {@link streamify}. Return
 * `undefined` (or leave a hook out) for no message — DSPy's
 * `StatusMessageProvider`.
 */
export interface StatusMessageProvider {
    moduleStart?(event: ModuleStartEvent): string | undefined;
    moduleEnd?(event: ModuleEndEvent): string | undefined;
    lmStart?(event: LMStartEvent): string | undefined;
    lmEnd?(event: LMEndEvent): string | undefined;
    toolStart?(event: ToolStartEvent): string | undefined;
    toolEnd?(event: ToolEndEvent): string | undefined;
}

/**
 * DSPy's default messages: one when a tool is called and one when it
 * returns. Module and language-model events say nothing by default — there
 * are too many of them to be useful to an end user.
 */
export const defaultStatusMessageProvider: StatusMessageProvider = {
    toolStart: (event) => `Calling tool ${event.toolName}...`,
    toolEnd: () => 'Tool calling finished! Querying the LLM with tool calling results...',
};

export interface StreamifyOptions {
    /** Status messages to emit. Defaults to {@link defaultStatusMessageProvider}. */
    statusMessageProvider?: StatusMessageProvider;
    /** End the stream with a `prediction` event. Defaults to `true`. */
    includeFinal?: boolean;
}

/** What {@link streamify} returns. */
export type StreamifiedProgram = (
    inputs: Record<string, any>,
    options?: LLMCallOptions
) => AsyncGenerator<StreamEvent, Prediction, void>;

/**
 * Run a whole program as a stream of events — DSPy's `streamify`.
 *
 * ```ts
 * const stream = streamify(rag);
 * for await (const event of stream({ question })) {
 *     if (event.type === 'status') showSpinner(event.message);
 *     if (event.type === 'field' && event.field === 'answer') append(event.delta ?? '');
 *     if (event.type === 'prediction') done(event.prediction);
 * }
 * ```
 *
 * Three kinds of event arrive, in the order they happen:
 *
 * - `status` — from module, language-model and tool callbacks, through the
 *   {@link StatusMessageProvider};
 * - `field` — partial output of every `Predict` (and `ChainOfThought`, and the
 *   predictors inside `ReAct`, …) the program runs, as its reply streams in.
 *   Any predictor runs as a stream while a streamified program is listening,
 *   provided its model supports streaming and its adapter can parse partial
 *   replies; otherwise it makes its usual single call, and its fields arrive
 *   once, complete;
 * - `prediction` — the program's result, last.
 *
 * Nothing about the program has to change: the request to stream reaches
 * predictors through an async-local scope, so it works through hand-written
 * `forward` methods that pass no options down.
 *
 * Field events are handed over with back-pressure — a predictor does not read
 * further from its provider until the consumer has taken the previous event —
 * and breaking out of the loop aborts the run, closes any open provider
 * stream, and waits for the program to settle before the loop exits. A
 * program that ignores the abort (a non-streaming call already in flight, a
 * tool doing its own I/O) is waited for; its result is discarded.
 *
 * The generator's return value is the program's `Prediction` too, for callers
 * who drive `next()` themselves. An error from the program is rethrown from
 * the generator after every event before it has been delivered.
 */
export function streamify(program: Module, options: StreamifyOptions = {}): StreamifiedProgram {
    const provider = options.statusMessageProvider ?? defaultStatusMessageProvider;
    const includeFinal = options.includeFinal ?? true;

    return async function* streamProgram(
        inputs: Record<string, any>,
        callOptions?: LLMCallOptions
    ): AsyncGenerator<StreamEvent, Prediction, void> {
        const userSignal = callOptions?.signal;
        if (userSignal?.aborted) throw userSignal.reason;

        const controller = new AbortController();
        const onAbort = () => controller.abort(userSignal!.reason);
        userSignal?.addEventListener('abort', onAbort, { once: true });

        const channel = new AsyncChannel<StreamEvent>();
        const paths = new Map<object, string>(
            program.namedPredictors().map(([path, predictor]) => [predictor, path])
        );
        const pathOf = (predictor: object) =>
            paths.get(predictor) ?? (predictor as { moduleId?: string }).moduleId ?? 'unknown';

        // The last value seen per invocation and field, for computing deltas.
        const lastValues = new WeakMap<object, Record<string, unknown>>();
        const fieldEvents = (
            predictor: object,
            invocation: object,
            fields: Record<string, unknown>,
            done: boolean
        ): FieldStreamEvent[] => {
            const previous = lastValues.get(invocation) ?? {};
            const events: FieldStreamEvent[] = [];
            for (const [field, value] of Object.entries(fields)) {
                const before = previous[field];
                if (!done && sameValue(before, value)) continue;
                const event: FieldStreamEvent = {
                    type: 'field',
                    path: pathOf(predictor),
                    field,
                    value,
                };
                const delta = textDelta(before, value);
                if (delta !== undefined) event.delta = delta;
                if (done) event.done = true;
                events.push(event);
            }
            lastValues.set(invocation, { ...previous, ...fields });
            return events;
        };

        const listener: StreamListener = {
            signal: controller.signal,
            onPartial: async ({ predictor, invocation, snapshot }) => {
                for (const event of fieldEvents(predictor, invocation, snapshot, false)) {
                    await channel.push(event);
                }
            },
            onComplete: ({ predictor, invocation, outputs }) => {
                for (const event of fieldEvents(predictor, invocation, outputs, true)) {
                    void channel.push(event);
                }
            },
        };

        const status = (message: string | undefined) => {
            if (typeof message === 'string' && message !== '') {
                void channel.push({ type: 'status', message });
            }
        };
        const statusCallback: Callback = {
            onModuleStart: (event) => status(provider.moduleStart?.(event)),
            onModuleEnd: (event) => status(provider.moduleEnd?.(event)),
            onLMStart: (event) => status(provider.lmStart?.(event)),
            onLMEnd: (event) => status(provider.lmEnd?.(event)),
            onToolStart: (event) => status(provider.toolStart?.(event)),
            onToolEnd: (event) => status(provider.toolEnd?.(event)),
        };

        let prediction: Prediction | undefined;
        let settled = false;
        const task = context({ callbacks: [...getCallbacks(), statusCallback] }, () =>
            runWithStreamListener(listener, () =>
                program.forward(inputs, { ...callOptions, signal: controller.signal })
            )
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
            for (;;) {
                const next = await channel.next();
                if (next.done) break;
                yield next.value;
            }
            if (includeFinal) yield { type: 'prediction', prediction: prediction! };
            return prediction!;
        } finally {
            // Abort before releasing producers, so a predictor resuming from
            // back-pressure sees the cancellation rather than reading on.
            if (!settled) controller.abort(consumerGoneReason());
            channel.cancel();
            await task;
            userSignal?.removeEventListener('abort', onAbort);
        }
    };
}

/** What was appended, when `after` is text extending `before`. */
function textDelta(before: unknown, after: unknown): string | undefined {
    if (typeof after !== 'string') return undefined;
    if (before === undefined) return after;
    if (typeof before === 'string' && after.startsWith(before))
        return after.slice(before.length);
    return undefined;
}

function sameValue(a: unknown, b: unknown): boolean {
    if (a === b) return true;
    if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null)
        return false;
    try {
        return JSON.stringify(a) === JSON.stringify(b);
    } catch {
        return false;
    }
}
