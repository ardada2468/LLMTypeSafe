import { AsyncLocalStorage } from 'node:async_hooks';

/**
 * Internal plumbing shared by `Predict.stream` and `streamify`: a way for code
 * far above a predictor to ask it to stream, and a back-pressured channel to
 * carry what it streams.
 *
 * Asking through `AsyncLocalStorage` rather than through call options matters
 * because a hand-written `forward` is free to drop the options it was given.
 * The listener still reaches every `Predict` the program runs, however deeply
 * nested, as long as it runs inside the listening scope.
 */

/** One invocation's partial output, as a predictor reports it. */
export interface PartialOutputEvent {
    /** The predictor streaming. Compared by identity. */
    predictor: object;
    /** Fresh per invocation, so two calls of one predictor can be told apart. */
    invocation: object;
    /** The output fields parsed so far, uncoerced. */
    snapshot: Record<string, unknown>;
}

/** One invocation's validated output, reported once it has been parsed. */
export interface CompleteOutputEvent {
    predictor: object;
    invocation: object;
    outputs: Record<string, unknown>;
}

/**
 * Receives partial outputs from every streaming predictor in scope.
 *
 * Returning a promise applies back-pressure: the predictor does not read the
 * next chunk from the provider until it settles. That is what lets a consumer
 * that stops reading also stop the provider stream, rather than the program
 * racing on to the end in the background.
 */
export interface StreamListener {
    onPartial(event: PartialOutputEvent): void | Promise<void>;
    onComplete?(event: CompleteOutputEvent): void;
    /**
     * Cancels the streaming calls made on this listener's behalf, for a
     * predictor whose own call options carried no signal.
     */
    signal?: AbortSignal;
}

const listeners = new AsyncLocalStorage<readonly StreamListener[]>();

/**
 * Run `fn` with `listener` added to the listeners in scope. Scopes nest: an
 * inner `Predict.stream` inside a `streamify`'d program reports to both.
 */
export function runWithStreamListener<T>(listener: StreamListener, fn: () => T): T {
    return listeners.run([...(listeners.getStore() ?? []), listener], fn);
}

/** The listeners in scope, outermost first. Empty — never `undefined` — when none. */
export function activeStreamListeners(): readonly StreamListener[] {
    return listeners.getStore() ?? [];
}

/**
 * A single-consumer queue whose `push` resolves only once the consumer has
 * moved past the pushed item, so a producer that awaits it can never get more
 * than one item ahead of the reader.
 *
 * Producers that cannot wait — callback hooks, which are synchronous — simply
 * ignore the promise; their items are buffered.
 */
export class AsyncChannel<T> {
    private readonly items: Array<{ value: T; ack: () => void }> = [];
    private inFlight?: () => void;
    private wake?: () => void;
    private ended = false;
    private cancelled = false;
    private failure?: { error: unknown };

    /**
     * Queue `value`. Resolves when the consumer asks for the item after it, or
     * at once if the consumer has gone away — a producer must never be left
     * waiting on a reader that no longer exists.
     */
    push(value: T): Promise<void> {
        if (this.cancelled || this.ended) return Promise.resolve();
        return new Promise<void>((resolve) => {
            this.items.push({ value, ack: resolve });
            this.notify();
        });
    }

    /** No more items. With `error`, the consumer's next read past the queue throws it. */
    end(error?: { error: unknown }): void {
        this.ended = true;
        this.failure = error;
        this.notify();
    }

    /**
     * The next item, or `done` once the channel has ended and drained. Asking
     * acknowledges the previous item, releasing its producer.
     */
    async next(): Promise<IteratorResult<T, undefined>> {
        this.release();
        for (;;) {
            const item = this.items.shift();
            if (item) {
                this.inFlight = item.ack;
                return { done: false, value: item.value };
            }
            if (this.ended || this.cancelled) {
                if (this.failure && !this.cancelled) throw this.failure.error;
                return { done: true, value: undefined };
            }
            await new Promise<void>((resolve) => {
                this.wake = resolve;
            });
        }
    }

    /** The consumer is leaving: release every producer and drop what is queued. */
    cancel(): void {
        this.cancelled = true;
        this.release();
        for (const item of this.items.splice(0)) item.ack();
        this.notify();
    }

    private release(): void {
        const ack = this.inFlight;
        this.inFlight = undefined;
        ack?.();
    }

    private notify(): void {
        const wake = this.wake;
        this.wake = undefined;
        wake?.();
    }
}

/** Abort with a reason that reads as an abort, for a consumer that stopped reading. */
export function consumerGoneReason(): Error {
    return Object.assign(new Error('Stream consumer stopped reading'), { name: 'AbortError' });
}
