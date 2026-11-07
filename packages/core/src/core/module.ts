import { type SignatureLike } from './signature';
import { Prediction, attachLMUsage } from './prediction';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import type { Predict } from '../modules/predict';
import { getCallbacks, getDefaultLM, isUsageTrackingEnabled } from './config';
import { mapWithConcurrency, DEFAULT_CONCURRENCY, type SettledResult } from '../utils/pool';
import { beginTrace, nextModuleId, type TraceSpan } from './trace';
import { currentCallFrame, emit, newCallFrame, runInFrame } from './callbacks';
import { currentUsageCollector, runWithCollector, UsageCollector } from './usage';
import { Example } from './example';
import { TsDspyError } from './errors';
import { VERSION } from './version';

/** Per-input outcome of {@link Module.batch}, in the shape of `Promise.allSettled`. */
export type BatchResult<T = Prediction> = SettledResult<T>;

/** Options for {@link Module.batch}: pool controls plus the usual per-call options. */
export interface BatchOptions extends LLMCallOptions {
    /** Maximum number of inputs in flight at once. Defaults to 8. */
    concurrency?: number;
    /** Called as each input settles, whether it succeeded or failed. */
    onProgress?: (done: number, total: number) => void;
    /** Reject the whole batch on the first failure instead of capturing it. */
    stopOnError?: boolean;
    /** Stop starting new inputs once aborted, then reject with the abort reason. */
    signal?: AbortSignal;
}

/** One demo as saved by {@link Module.dumpState}: its fields and which are inputs. */
export interface SerializedDemo {
    data: Record<string, unknown>;
    /** Present when the demo designated its input fields with `withInputs`. */
    inputKeys?: string[];
}

/** The optimizable state of one predictor. */
export interface PredictorState {
    demos: SerializedDemo[];
    /** Present only when the instructions were overridden. */
    instructions?: string;
}

export interface ProgramStateMetadata {
    library: 'ts-dspy';
    /** Version of `@ts-dspy/core` that wrote the state. */
    version: string;
    /** ISO-8601 timestamp. */
    savedAt: string;
}

/**
 * A program's learned state, keyed by predictor path (see
 * {@link Module.namedPredictors}), plus a `metadata` entry. Plain JSON, so it
 * can be written to disk, a database, or version control.
 */
export type ProgramState = { metadata: ProgramStateMetadata } & {
    [predictorPath: string]: PredictorState | ProgramStateMetadata;
};

/**
 * Marks a module as a predictor — a unit with its own demos and instructions
 * that optimizers tune. Set on `Predict.prototype`; a symbol rather than an
 * `instanceof Predict` check because `Predict` imports this file.
 */
export const PREDICTOR: unique symbol = Symbol.for('ts-dspy.predictor');

/** The slice of `Predict` that state management needs. */
interface PredictorAccess {
    dumpPredictorState(): PredictorState;
    loadPredictorState(state: PredictorState): void;
    resetPredictorState(): void;
}

function isPredictor(value: unknown): value is Predict<any, any> & PredictorAccess {
    return (
        typeof value === 'object' &&
        value !== null &&
        (value as Record<PropertyKey, unknown>)[PREDICTOR] === true
    );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
    if (value === null || typeof value !== 'object') return false;
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}

/**
 * Fields never walked for sub-modules: they hold a model, a signature and
 * demos, none of which can contain a module, and some (a zod signature) are
 * large object graphs.
 */
const OPAQUE_FIELDS = new Set(['lm', 'signature', 'demos']);

export abstract class Module {
    /**
     * The model this module was given explicitly — by its constructor or
     * `withLM` — or `undefined` to use whatever is configured when it runs.
     *
     * Deliberately **not** resolved at construction: a module built at import
     * time, before `configure()` runs, must still work, and the same module
     * must be able to run against different models in different
     * `settings.context` scopes. Read the model to call through
     * {@link resolveLM}.
     */
    protected lm?: ILanguageModel;
    protected signature?: SignatureLike;

    /** Identifies this instance in trace entries, e.g. `Predict#1`. */
    readonly moduleId: string;

    constructor(signature?: SignatureLike, lm?: ILanguageModel) {
        this.signature = signature;
        this.lm = lm;
        this.moduleId = nextModuleId(this.constructor.name);
        instrument(this);
    }

    abstract forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction>;

    /**
     * The model a call should use, resolved at call time: `options.lm`, then
     * this module's own model, then the active `settings.context` scope's, then
     * the global default from `configure()`.
     *
     * @throws when none of those supplies one.
     */
    protected resolveLM(options?: LLMCallOptions): ILanguageModel {
        return options?.lm ?? this.lm ?? getDefaultLM();
    }

    /** Alias for {@link forward}, so modules read like function calls. */
    async call(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /** Alias for {@link forward}, mirroring DSPy's Python naming. */
    async __call__(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /**
     * Run {@link forward} over many inputs with a bounded number of calls in flight.
     *
     * Results come back **in input order**, whatever order the calls finished in.
     * By default a failing input is captured rather than thrown, so one bad row
     * does not destroy a ten-thousand-row job:
     *
     * ```ts
     * const results = await predict.batch(rows, { concurrency: 16 });
     * const answers = results.filter((r) => r.status === 'fulfilled').map((r) => r.value);
     * ```
     *
     * Every other option — `temperature`, `timeout`, `retries` and the rest — is
     * passed through to each underlying call unchanged.
     *
     * An aborted batch rejects rather than returning the inputs that already
     * finished, so wrap it in a `try` if you cancel it.
     *
     * @throws RangeError if `concurrency` is not a positive integer.
     * @throws the lowest-indexed failure when `stopOnError` is set, or the abort
     *   reason when `signal` aborts.
     */
    async batch(
        inputs: Record<string, any>[],
        options: BatchOptions = {}
    ): Promise<BatchResult<Awaited<ReturnType<this['forward']>>>[]> {
        const {
            concurrency = DEFAULT_CONCURRENCY,
            onProgress,
            stopOnError = false,
            signal,
            ...callOptions
        } = options;

        return mapWithConcurrency(inputs, (input) => this.forward(input, callOptions), {
            concurrency,
            onSettled: onProgress,
            stopOnError,
            signal,
        }) as Promise<BatchResult<Awaited<ReturnType<this['forward']>>>[]>;
    }

    /**
     * Run one module invocation and wrap its output in a `Prediction`.
     *
     * With `configure({ tracing: true })` the callback receives a span to report
     * each language-model call to, and the resulting entry is attached to the
     * prediction as `trace` and appended to the history behind
     * `inspectHistory()`. Failures are recorded too — the prompt behind a
     * `ValidationError` is exactly what you want to read — and then rethrown.
     *
     * With tracing off the span is `undefined` and nothing is timed, copied, or
     * stored.
     */
    protected async traced<TOutput extends Record<string, any>>(
        inputs: Record<string, any>,
        run: (span: TraceSpan | undefined) => Promise<TOutput>
    ): Promise<Prediction<TOutput>> {
        const frame = currentCallFrame();
        if (frame?.instance === this) {
            return this.tracedInFrame(inputs, run);
        }
        // Reached without going through the instrumented `forward` — a clone
        // made by hand without copying it, say — so open this invocation's
        // scope here instead, to keep callbacks and trace nesting correct.
        return runModuleScope(this, inputs, () => this.tracedInFrame(inputs, run)) as Promise<
            Prediction<TOutput>
        >;
    }

    private async tracedInFrame<TOutput extends Record<string, any>>(
        inputs: Record<string, any>,
        run: (span: TraceSpan | undefined) => Promise<TOutput>
    ): Promise<Prediction<TOutput>> {
        const frame = currentCallFrame();
        const span = beginTrace(
            this.moduleId,
            inputs,
            frame?.instance === this ? frame : undefined
        );
        if (!span) {
            return new Prediction(await run(undefined));
        }

        let output: TOutput;
        try {
            output = await run(span);
        } catch (error) {
            span.fail(error);
            throw error;
        }

        return new Prediction(output, span.finish(output));
    }

    // ------------------------------------------------------------ composition

    /**
     * Every module reachable from this one's own enumerable fields, depth
     * first, with the path that reaches it: `retrieve`, `steps[0]`,
     * `experts.math`, `pipeline.generate`.
     *
     * Modules held directly, in arrays, and in plain-object records are found,
     * at any depth. Unlike DSPy's `named_sub_modules`, this module itself is
     * **not** included — only its descendants. A module reachable by two paths
     * is reported once, under the first, and cycles are not followed.
     */
    namedSubModules(): Array<[string, Module]> {
        const found: Array<[string, Module]> = [];
        const seen = new Set<object>([this]);

        const visit = (value: unknown, path: string): void => {
            if (value === null || typeof value !== 'object' || seen.has(value)) return;

            if (value instanceof Module) {
                seen.add(value);
                found.push([path, value]);
                walkFields(value, path);
            } else if (Array.isArray(value)) {
                seen.add(value);
                value.forEach((item, index) => visit(item, `${path}[${index}]`));
            } else if (isPlainObject(value)) {
                seen.add(value);
                for (const [key, item] of Object.entries(value)) {
                    visit(item, `${path}.${key}`);
                }
            }
        };

        const walkFields = (module: Module, prefix: string): void => {
            for (const [key, value] of Object.entries(module)) {
                if (OPAQUE_FIELDS.has(key)) continue;
                visit(value, prefix ? `${prefix}.${key}` : key);
            }
        };

        walkFields(this, '');
        return found;
    }

    /**
     * Every predictor in this program — the `Predict` instances (including
     * `ChainOfThought`) whose demos and instructions an optimizer tunes — with
     * its path.
     *
     * A predictor reports itself as `self`, so a bare `Predict` returns
     * `[['self', predict]]`; a composite returns its predictors' dotted paths.
     * `RespAct` holds no inner `Predict`, so it contributes none.
     */
    namedPredictors(): Array<[string, Predict<any, any>]> {
        const result: Array<[string, Predict<any, any>]> = [];
        if (isPredictor(this)) result.push(['self', this]);
        for (const [path, module] of this.namedSubModules()) {
            if (isPredictor(module)) result.push([path, module]);
        }
        return result;
    }

    /** The predictors of {@link namedPredictors}, without their paths. */
    predictors(): Array<Predict<any, any>> {
        return this.namedPredictors().map(([, predictor]) => predictor);
    }

    /**
     * An independent copy of this whole module tree, preserving every
     * module's class.
     *
     * Sub-modules, the arrays and plain records holding them, and demo lists
     * are copied, so changing the copy's demos or instructions never touches
     * the original. Language models, signatures, tools and other class
     * instances are shared by reference — they are configuration, not learned
     * state, and an LM client is not something to duplicate. Each copied
     * module gets a fresh `moduleId`.
     */
    deepCopy(): this {
        return copyValue(this, new Map()) as this;
    }

    /** A {@link deepCopy} with every predictor's demos cleared and instructions reset. */
    resetCopy(): this {
        const copy = this.deepCopy();
        for (const predictor of copy.predictors()) {
            (predictor as unknown as PredictorAccess).resetPredictorState();
        }
        return copy;
    }

    // ------------------------------------------------------------ persistence

    /**
     * This program's learned state — every predictor's demos and instructions —
     * as plain JSON, keyed by predictor path. Restore it into a freshly built
     * program of the same shape with {@link loadState}.
     */
    dumpState(): ProgramState {
        const state: Record<string, PredictorState | ProgramStateMetadata> = {};
        for (const [path, predictor] of this.namedPredictors()) {
            if (path === 'metadata') {
                throw new TsDspyError(
                    'A predictor at the path "metadata" collides with the state\'s ' +
                        'metadata entry. Rename that field to save this program.'
                );
            }
            state[path] = (predictor as unknown as PredictorAccess).dumpPredictorState();
        }
        state.metadata = {
            library: 'ts-dspy',
            version: VERSION,
            savedAt: new Date().toISOString(),
        };
        return state as ProgramState;
    }

    /**
     * Restore state produced by {@link dumpState}, in place, and return this
     * module.
     *
     * The state must describe exactly this program's predictors: a path the
     * program lacks, or a predictor the state lacks, is an error rather than a
     * silent partial load — loading a prompt into the wrong predictor is far
     * worse than failing. Nothing is changed unless the whole state is valid.
     *
     * @throws {TsDspyError} on unknown or missing predictor paths, or a
     *   malformed entry.
     */
    loadState(state: ProgramState | Record<string, unknown>): this {
        if (!isPlainObject(state)) {
            throw new TsDspyError('Program state must be a plain object.');
        }

        const predictors = this.namedPredictors();
        const known = new Set(predictors.map(([path]) => path));
        const provided = Object.keys(state).filter((key) => key !== 'metadata');
        const unknown = provided.filter((path) => !known.has(path));
        const missing = [...known].filter((path) => !(path in state));

        if (unknown.length > 0 || missing.length > 0) {
            const parts: string[] = [];
            if (unknown.length > 0) {
                parts.push(`unknown predictor path(s): ${unknown.join(', ')}`);
            }
            if (missing.length > 0) {
                parts.push(`missing predictor path(s): ${missing.join(', ')}`);
            }
            throw new TsDspyError(
                `Program state does not match this program (${parts.join('; ')}). ` +
                    `This program's predictors are: ${[...known].join(', ') || '(none)'}.`
            );
        }

        const entries = predictors.map(
            ([path, predictor]) =>
                [predictor, validatePredictorState(path, state[path])] as const
        );
        for (const [predictor, entry] of entries) {
            (predictor as unknown as PredictorAccess).loadPredictorState(entry);
        }
        return this;
    }

    /** Write {@link dumpState} to `path` as JSON. Node only. */
    async save(path: string): Promise<void> {
        // Imported lazily so the main entry point stays loadable where there is
        // no filesystem — a browser bundle, an edge runtime.
        const { writeFile } = await import('node:fs/promises');
        await writeFile(path, `${JSON.stringify(this.dumpState(), null, 2)}\n`, 'utf8');
    }

    /** Read state written by {@link save} and {@link loadState} it. Node only. */
    async load(path: string): Promise<this> {
        const { readFile } = await import('node:fs/promises');
        const text = await readFile(path, 'utf8');
        let state: unknown;
        try {
            state = JSON.parse(text);
        } catch (cause) {
            throw new TsDspyError(`Program state at ${path} is not valid JSON.`, { cause });
        }
        return this.loadState(state as Record<string, unknown>);
    }
}

function validatePredictorState(path: string, value: unknown): PredictorState {
    const fail = (why: string): never => {
        throw new TsDspyError(`Program state for predictor "${path}" is malformed: ${why}.`);
    };

    if (!isPlainObject(value)) fail('expected an object');
    const entry = value as Record<string, unknown>;
    if (!Array.isArray(entry.demos)) fail('`demos` must be an array');
    if (entry.instructions !== undefined && typeof entry.instructions !== 'string') {
        fail('`instructions` must be a string when present');
    }

    const demos = (entry.demos as unknown[]).map((demo, index) => {
        if (!isPlainObject(demo) || !isPlainObject(demo.data)) {
            return fail(`demo ${index} must be an object with a \`data\` object`);
        }
        if (
            demo.inputKeys !== undefined &&
            (!Array.isArray(demo.inputKeys) ||
                !demo.inputKeys.every((key) => typeof key === 'string'))
        ) {
            return fail(`demo ${index} has a non-string-array \`inputKeys\``);
        }
        return demo as unknown as SerializedDemo;
    });

    return {
        demos,
        ...(entry.instructions !== undefined
            ? { instructions: entry.instructions as string }
            : {}),
    };
}

// ------------------------------------------------------------- instrumentation

/**
 * Install the instrumented `forward` on a module instance.
 *
 * Every subclass defines `forward` on its prototype, and the base constructor
 * runs with that prototype already in place — so the base constructor can
 * shadow it with an own property that opens the invocation's scope (call
 * frame, usage collector, callback events) and then runs the real
 * implementation. Doing it here rather than in `call()` means a composite's
 * `forward` that calls `this.retrieve.forward(...)` directly still produces
 * correctly nested events; nobody has to remember to go through `.call()`.
 *
 * The implementation is looked up on the prototype at call time, so a test
 * spy installed on a prototype after construction is still honoured, and a
 * copy made with `Object.create(proto)` behaves identically. Non-enumerable,
 * so it stays out of `Object.keys`, spreads and the composition walk.
 *
 * A subclass that declares `forward` as an arrow-function class field
 * replaces this wrapper; the built-in modules' `traced()` re-opens the scope
 * in that case, but a hand-written module's own events are then lost.
 */
function instrument(module: Module): void {
    Object.defineProperty(module, 'forward', {
        value: instrumentedForward,
        writable: true,
        configurable: true,
        enumerable: false,
    });
}

function instrumentedForward(
    this: Module,
    inputs: Record<string, any>,
    options?: LLMCallOptions
): Promise<Prediction> {
    const impl = (Object.getPrototypeOf(this) as Module).forward;
    return runModuleScope(this, inputs, () => impl.call(this, inputs, options));
}

/**
 * Run one module invocation inside its own call frame and usage collector,
 * notifying callbacks either side.
 *
 * Skipped entirely — `run` is called bare — when no callback is registered and
 * neither tracing nor usage tracking is on, so the default path allocates
 * nothing.
 */
async function runModuleScope(
    module: Module,
    inputs: Record<string, any>,
    run: () => Promise<Prediction>
): Promise<Prediction> {
    const callbacks = getCallbacks();
    const notify = callbacks.length > 0;
    if (!notify && !isUsageTrackingEnabled()) {
        return run();
    }

    const frame = newCallFrame(module);
    const usage = new UsageCollector(currentUsageCollector());
    const startTime = Date.now();
    const base = {
        callId: frame.callId,
        parentCallId: frame.parentCallId,
        startTime,
        instance: module,
        moduleId: module.moduleId,
    };

    if (notify) emit('onModuleStart', { ...base, inputs }, callbacks);

    let prediction: Prediction;
    try {
        prediction = await runInFrame(frame, () => runWithCollector(usage, run));
    } catch (error) {
        if (notify) {
            const endTime = Date.now();
            emit(
                'onModuleEnd',
                {
                    ...base,
                    endTime,
                    durationMs: endTime - startTime,
                    error,
                    usage: usage.snapshot(),
                },
                callbacks
            );
        }
        throw error;
    }

    const snapshot = usage.snapshot();
    if (prediction instanceof Prediction) attachLMUsage(prediction, snapshot);

    if (notify) {
        const endTime = Date.now();
        emit(
            'onModuleEnd',
            {
                ...base,
                endTime,
                durationMs: endTime - startTime,
                outputs:
                    prediction instanceof Prediction
                        ? prediction.toObject()
                        : (prediction as Record<string, any>),
                usage: snapshot,
            },
            callbacks
        );
    }
    return prediction;
}

// ----------------------------------------------------------------- deep copy

/** Copy a demo: its own data, and the same input split. */
function copyExample(example: Example): Example {
    const copy = new Example(example.toObject());
    const keys = example.getInputKeys();
    return keys ? copy.withInputs(...keys) : copy;
}

function copyValue(value: unknown, seen: Map<object, unknown>): unknown {
    if (value === null || typeof value !== 'object') return value;
    if (seen.has(value)) return seen.get(value);

    if (value instanceof Module) return copyModule(value, seen);

    if (value instanceof Example) {
        const copy = copyExample(value);
        seen.set(value, copy);
        return copy;
    }

    if (Array.isArray(value)) {
        const copy: unknown[] = [];
        seen.set(value, copy);
        for (const item of value) copy.push(copyValue(item, seen));
        return copy;
    }

    if (isPlainObject(value)) {
        const copy: Record<string, unknown> = Object.create(Object.getPrototypeOf(value));
        seen.set(value, copy);
        for (const [key, item] of Object.entries(value)) copy[key] = copyValue(item, seen);
        return copy;
    }

    // Models, signatures, tools, schemas and every other class instance are
    // configuration, shared rather than duplicated.
    return value;
}

/** Module fields a deep copy shares by reference: configuration, not state. */
const SHARED_FIELDS = new Set<PropertyKey>(['lm', 'signature', 'forward']);

function copyModule(module: Module, seen: Map<object, unknown>): Module {
    const copy = Object.create(Object.getPrototypeOf(module)) as Module;
    seen.set(module, copy);

    // Every own property, enumerable or not, so the instrumented `forward`
    // and any non-enumerable state a subclass keeps come along.
    for (const key of Reflect.ownKeys(module)) {
        const descriptor = Object.getOwnPropertyDescriptor(module, key)!;
        if ('value' in descriptor && !SHARED_FIELDS.has(key)) {
            descriptor.value = copyValue(descriptor.value, seen);
        }
        Object.defineProperty(copy, key, descriptor);
    }

    Object.defineProperty(copy, 'moduleId', {
        value: nextModuleId(module.constructor.name),
        writable: true,
        configurable: true,
        enumerable: true,
    });
    return copy;
}

/**
 * Shallow-copy a module, preserving its class and every own property —
 * including the instrumented `forward` — with some fields replaced. Used by
 * `Predict.withDemos`/`withLM`.
 */
export function cloneModule<M extends Module>(module: M, patch: Record<string, unknown>): M {
    const copy = Object.create(Object.getPrototypeOf(module)) as M;
    Object.defineProperties(copy, Object.getOwnPropertyDescriptors(module));
    Object.assign(copy, patch);
    return copy;
}
