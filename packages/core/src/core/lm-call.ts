import type { ChatMessage, ILanguageModel, LLMCallOptions } from '../types/language-model';
import {
    emit,
    hasCallbacks,
    newCallFrame,
    runInFrame,
    type CallFrame,
    type LMMethod,
} from './callbacks';
import { type TraceSpan } from './trace';
import { currentUsageCollector, runWithCollector, UsageCollector } from './usage';

/**
 * Remove the options ts-dspy consumes itself before they reach a provider.
 *
 * `lm` selects the model and `adapter` the prompt format; neither means
 * anything to a provider SDK. The original object is returned untouched when
 * there is nothing to strip, so a caller comparing options by identity or deep
 * equality sees exactly what it passed.
 */
export function providerOptions(options?: LLMCallOptions): LLMCallOptions | undefined {
    if (!options || (!('lm' in options) && !('adapter' in options))) return options;
    const { lm: _lm, adapter: _adapter, ...rest } = options;
    return rest;
}

/** What a module is about to send, for tracing and callbacks. */
export interface LMCallInfo {
    lm: ILanguageModel;
    method: LMMethod;
    /** Text recorded as the call's prompt in a trace. */
    traceText: string;
    prompt?: string;
    messages?: ChatMessage[];
    /** Options as sent to the provider (already stripped). */
    options?: LLMCallOptions;
    /** Open when tracing is on. */
    span?: TraceSpan;
    moduleId?: string;
}

/**
 * One instrumented language-model call in flight: reported to the trace span
 * and to callbacks, with its usage collected in a scope of its own so the
 * figure is exactly this call's.
 */
export class LMCallHandle {
    private readonly frame: CallFrame;
    private readonly usage: UsageCollector;
    private readonly startTime = Date.now();
    private readonly notify: boolean;

    constructor(private readonly info: LMCallInfo) {
        this.frame = newCallFrame();
        // Nested in the enclosing collector so the module's own figure, and
        // any evaluation around it, still sees this call.
        this.usage = new UsageCollector(currentUsageCollector());
        this.notify = hasCallbacks();

        info.span?.startCall(info.traceText);
        if (this.notify) {
            emit('onLMStart', {
                callId: this.frame.callId,
                parentCallId: this.frame.parentCallId,
                startTime: this.startTime,
                lm: info.lm,
                model: safeModelName(info.lm),
                method: info.method,
                prompt: info.prompt,
                messages: info.messages,
                options: info.options,
                moduleId: info.moduleId,
            });
        }
    }

    /** Run part of the call inside its frame and usage scope. */
    run<T>(fn: () => T): T {
        return runInFrame(this.frame, () => runWithCollector(this.usage, fn));
    }

    /** The call succeeded with `outputs`; `text` is what the trace records. */
    end(outputs: unknown, text: string): void {
        this.info.span?.endCall(text, this.usage.total());
        this.finish({ outputs });
    }

    /** The call threw. The trace keeps its prompt with an empty reply. */
    fail(error: unknown): void {
        this.info.span?.endCall('', this.usage.total());
        this.finish({ error });
    }

    private finish(result: { outputs?: unknown; error?: unknown }): void {
        if (!this.notify) return;
        const endTime = Date.now();
        emit('onLMEnd', {
            callId: this.frame.callId,
            parentCallId: this.frame.parentCallId,
            startTime: this.startTime,
            endTime,
            durationMs: endTime - this.startTime,
            lm: this.info.lm,
            model: safeModelName(this.info.lm),
            method: this.info.method,
            moduleId: this.info.moduleId,
            usage: this.usage.snapshot(),
            ...result,
        });
    }
}

/**
 * Open an instrumented call, or return `undefined` when neither tracing nor
 * any callback is listening — the default path then costs one check.
 */
export function startLMCall(info: LMCallInfo): LMCallHandle | undefined {
    if (!info.span && !hasCallbacks()) return undefined;
    return new LMCallHandle(info);
}

/**
 * Make one language-model call on behalf of a module. Every LM call the
 * built-in modules make goes through here, so tracing, callbacks and per-call
 * usage are handled in exactly one place.
 *
 * `describe` renders the result as the text a trace records.
 */
export async function invokeLM<T>(
    info: LMCallInfo,
    run: () => Promise<T>,
    describe: (result: T) => string
): Promise<T> {
    const handle = startLMCall(info);
    if (!handle) return run();

    let result: T;
    try {
        result = await handle.run(run);
    } catch (error) {
        handle.fail(error);
        throw error;
    }
    handle.end(result, describe(result));
    return result;
}

function safeModelName(lm: ILanguageModel): string {
    try {
        return lm.getModelName();
    } catch {
        return 'unknown';
    }
}
