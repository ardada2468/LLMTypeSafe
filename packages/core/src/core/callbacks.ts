import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import type { ChatMessage, ILanguageModel, LLMCallOptions } from '../types/language-model';
import type { LMUsage } from './usage';
import { getCallbacks } from './config';

/** Fields every callback event carries. */
export interface CallbackEventBase {
    /**
     * Unique per invocation. The matching `…End` event carries the same id, so
     * start and end can be paired even when invocations interleave.
     */
    callId: string;
    /**
     * The `callId` of the invocation this one ran inside — the enclosing module
     * call, usually — or `undefined` at the top level. Tracked with
     * `AsyncLocalStorage`, so it is correct under concurrency.
     */
    parentCallId?: string;
    /** `Date.now()` when the invocation started. */
    startTime: number;
}

/** Fields every `…End` event adds. */
export interface CallbackEndBase extends CallbackEventBase {
    /** `Date.now()` when the invocation settled. */
    endTime: number;
    /** `endTime - startTime`, in milliseconds. */
    durationMs: number;
    /** Set when the invocation threw; `outputs` is then absent. */
    error?: unknown;
}

/**
 * Minimal view of a module, kept structural so this file does not import
 * `Module` (which imports this file).
 */
export interface ModuleLike {
    readonly moduleId: string;
}

export interface ModuleStartEvent extends CallbackEventBase {
    /** The module instance being invoked. */
    instance: ModuleLike;
    /** Its stable id, e.g. `Predict#3`. */
    moduleId: string;
    inputs: Record<string, any>;
}

export interface ModuleEndEvent extends CallbackEndBase {
    instance: ModuleLike;
    moduleId: string;
    /** The prediction's fields. */
    outputs?: Record<string, any>;
    /**
     * Every language-model call made during the invocation, including nested
     * modules', keyed by `provider/model`.
     */
    usage: LMUsage;
}

/** Which `ILanguageModel` method was called. */
export type LMMethod =
    'generate' | 'generateStructured' | 'chat' | 'chatWithTools' | 'chatStream';

export interface LMStartEvent extends CallbackEventBase {
    lm: ILanguageModel;
    /** `getModelName()` at call time. */
    model: string;
    method: LMMethod;
    /** The prompt, for `generate`/`generateStructured`. */
    prompt?: string;
    /** The messages, for the chat methods. */
    messages?: ChatMessage[];
    /** Call options as sent to the provider. */
    options?: LLMCallOptions;
    /** The module that made the call, when one did. */
    moduleId?: string;
}

export interface LMEndEvent extends CallbackEndBase {
    lm: ILanguageModel;
    model: string;
    method: LMMethod;
    moduleId?: string;
    /** The provider's reply: text, a structured object, or a `ChatResult`. */
    outputs?: unknown;
    /** Usage for this one call, keyed by `provider/model`. */
    usage: LMUsage;
}

export interface ToolStartEvent extends CallbackEventBase {
    toolName: string;
    args: unknown;
}

export interface ToolEndEvent extends CallbackEndBase {
    toolName: string;
    outputs?: unknown;
}

/** Emitted when an adapter renders a call as chat messages. */
export interface AdapterFormatEvent extends CallbackEndBase {
    /** The adapter's `name`, e.g. `ChatAdapter`. */
    adapter: string;
    /** The call's input field values. */
    inputs: Record<string, any>;
    /** The messages produced. */
    outputs?: ChatMessage[];
}

/**
 * Emitted when an adapter parses and validates a reply. Carries `error` — a
 * `ValidationError`, usually — when the reply did not satisfy the signature.
 */
export interface AdapterParseEvent extends CallbackEndBase {
    adapter: string;
    /** The model's reply text. */
    inputs: string;
    /** The validated output fields. */
    outputs?: Record<string, any>;
}

/**
 * Observes a program as it runs. Implement any subset of the hooks and
 * register the object with `configure({ callbacks: [...] })`, or for one scope
 * with `settings.context({ callbacks: [...] }, fn)`.
 *
 * Hooks are called synchronously and their return values ignored; anything a
 * hook throws is swallowed, because instrumentation must never be able to fail
 * the run it is instrumenting. Do slow work (exporting spans, say) off the
 * hot path.
 */
export interface Callback {
    onModuleStart?(event: ModuleStartEvent): void;
    onModuleEnd?(event: ModuleEndEvent): void;
    onLMStart?(event: LMStartEvent): void;
    onLMEnd?(event: LMEndEvent): void;
    onToolStart?(event: ToolStartEvent): void;
    onToolEnd?(event: ToolEndEvent): void;
    onAdapterFormat?(event: AdapterFormatEvent): void;
    onAdapterParse?(event: AdapterParseEvent): void;
}

type HookName = keyof Callback;
type EventOf<H extends HookName> = Parameters<NonNullable<Callback[H]>>[0];

/**
 * Notify every registered callback that implements `hook`. Exceptions are
 * dropped one callback at a time, so one broken observer does not starve the
 * rest.
 */
export function emit<H extends HookName>(
    hook: H,
    event: EventOf<H>,
    callbacks: readonly Callback[] = getCallbacks()
): void {
    for (const callback of callbacks) {
        const fn = callback[hook] as ((event: EventOf<H>) => void) | undefined;
        if (typeof fn !== 'function') continue;
        try {
            // A hook written `async` rejects rather than throws; left alone
            // that is an unhandled rejection, which ends the process.
            const returned: unknown = fn.call(callback, event);
            if (isThenable(returned)) returned.then(undefined, () => {});
        } catch {
            // Deliberately swallowed — see the doc comment on Callback.
        }
    }
}

/** Whether a hook's return value is a promise whose rejection needs handling. */
export function isThenable(value: unknown): value is PromiseLike<unknown> {
    return (
        value !== null &&
        (typeof value === 'object' || typeof value === 'function') &&
        typeof (value as { then?: unknown }).then === 'function'
    );
}

/** One open invocation on the logical call stack. */
export interface CallFrame {
    callId: string;
    parentCallId?: string;
    /** The module this frame belongs to, for module invocations. */
    instance?: ModuleLike;
}

const stack = new AsyncLocalStorage<CallFrame>();

/** The innermost open invocation in this async scope. */
export function currentCallFrame(): CallFrame | undefined {
    return stack.getStore();
}

/** Allocate a frame nested inside the current one, without entering it. */
export function newCallFrame(instance?: ModuleLike): CallFrame {
    return { callId: randomUUID(), parentCallId: stack.getStore()?.callId, instance };
}

/** Run `fn` with `frame` as the innermost open invocation. */
export function runInFrame<T>(frame: CallFrame, fn: () => T): T {
    return stack.run(frame, fn);
}

/** Whether anything is listening, so callers can skip building events. */
export function hasCallbacks(): boolean {
    return getCallbacks().length > 0;
}
