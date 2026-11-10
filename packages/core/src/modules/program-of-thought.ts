import type * as NodeVM from 'node:vm';
import { Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import { extendSignature, type SignatureLike } from '../core/signature';
import { getInputFieldConfigs, getOutputFieldConfigs } from '../utils/schema';
import { ChainOfThought } from './chain-of-thought';
import { type PredictOptions } from './predict';
import type { LLMCallOptions } from '../types/language-model';

export interface ProgramOfThoughtOptions extends PredictOptions {
    /**
     * Code generations allowed in all: the first, plus regenerations after a
     * failed run. Defaults to 3, as in DSPy.
     */
    maxIters?: number;
    /** Wall-clock budget for one run of the generated code. Defaults to 5000ms. */
    timeoutMs?: number;
}

/** Outcome of running generated code in {@link runSandboxed}. */
export interface SandboxResult {
    /** Everything the code logged, then its completion value. */
    output: string;
    /** Set when the code threw, timed out, or could not be compiled. */
    error?: string;
}

/**
 * Prediction by writing and running code — DSPy's `ProgramOfThought`, with
 * JavaScript in place of Python.
 *
 * Three `ChainOfThought` predictors, as in DSPy:
 *
 * 1. `codeGenerate` writes JavaScript computing the outputs from the inputs;
 * 2. the code runs in a fresh `node:vm` context; if it throws (or produces
 *    nothing), `codeRegenerate` is shown the code and the error and writes a
 *    fix — up to `maxIters` generations in all;
 * 3. `generateOutput` reads the code and what it printed and produces the
 *    signature's outputs.
 *
 * If no generation runs cleanly within `maxIters`, a `TsDspyError` is thrown
 * carrying the last error, as DSPy raises.
 *
 * @experimental **The sandbox is not a security boundary.** `node:vm` gives
 * the code an empty global scope — no `require`, `process`, `import()`,
 * `fetch` or timers — and a synchronous timeout, but Node documents that it
 * "is not a security mechanism": determined code can escape it. Only run
 * this against models and inputs you would trust to run code on the host. Do
 * real isolation (a separate process, container or worker with no
 * privileges) when the inputs come from users.
 */
export class ProgramOfThought extends Module {
    readonly codeGenerate: ChainOfThought;
    readonly codeRegenerate: ChainOfThought;
    readonly generateOutput: ChainOfThought;
    readonly maxIters: number;
    readonly timeoutMs: number;

    constructor(signature: SignatureLike, options: ProgramOfThoughtOptions = {}) {
        super(signature, options.lm);
        const { maxIters = 3, timeoutMs = 5000, ...predictOptions } = options;
        if (!Number.isInteger(maxIters) || maxIters < 1) {
            throw new RangeError(
                `maxIters must be a positive integer, received ${String(maxIters)}`
            );
        }
        this.maxIters = maxIters;
        this.timeoutMs = timeoutMs;

        const inputs = Object.keys(getInputFieldConfigs(signature));
        const outputs = Object.keys(getOutputFieldConfigs(signature));
        const generatedCode = {
            generated_code: {
                description: 'JavaScript code that computes the answer; plain code, no fences',
            },
        };

        this.codeGenerate = new ChainOfThought(
            extendSignature(signature, {
                removeOutputs: outputs,
                appendOutputs: generatedCode,
                instructions: instructions('generate', inputs, outputs),
            }),
            predictOptions
        );
        this.codeRegenerate = new ChainOfThought(
            extendSignature(signature, {
                appendInputs: {
                    previous_code: {
                        description: 'Previously generated JavaScript code that failed',
                    },
                    error: { description: 'Error message from running the previous code' },
                },
                removeOutputs: outputs,
                appendOutputs: generatedCode,
                instructions: instructions('regenerate', inputs, outputs),
            }),
            predictOptions
        );
        this.generateOutput = new ChainOfThought(
            extendSignature(signature, {
                appendInputs: {
                    final_generated_code: {
                        description: 'JavaScript code that computes the answer',
                    },
                    code_output: { description: 'Output of running the code' },
                },
                instructions: instructions('answer', inputs, outputs),
            }),
            predictOptions
        );
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction & Record<string, any>> {
        let generated = await this.codeGenerate.forward(inputs, options);
        let code = extractCode(String(generated.generated_code ?? ''));
        let result = await this.execute(code);

        for (
            let iteration = 1;
            result.error !== undefined && iteration < this.maxIters;
            iteration++
        ) {
            generated = await this.codeRegenerate.forward(
                { ...inputs, previous_code: code, error: result.error },
                options
            );
            code = extractCode(String(generated.generated_code ?? ''));
            result = await this.execute(code);
        }

        if (result.error !== undefined) {
            throw new TsDspyError(
                `ProgramOfThought: no generated code ran successfully in ${this.maxIters} ` +
                    `attempt(s). Last error: ${result.error}`
            );
        }

        return this.generateOutput.forward(
            { ...inputs, final_generated_code: code, code_output: result.output },
            options
        );
    }

    /** Run `code` in the sandbox. Overridable, e.g. to route it to a real isolate. */
    protected execute(code: string): Promise<SandboxResult> {
        return runSandboxed(code, { timeoutMs: this.timeoutMs });
    }
}

/** DSPy's `_generate_instruction`, for JavaScript in a sandbox. */
function instructions(
    mode: 'generate' | 'regenerate' | 'answer',
    inputs: string[],
    outputs: string[]
): string {
    const given = inputs.map((name) => `\`${name}\``).join(', ');
    const wanted = outputs.map((name) => `\`${name}\``).join(', ');
    const sandbox =
        'The code runs as a plain script in an empty JavaScript sandbox: there is no ' +
        '`require`, `import`, `process`, network, filesystem or timers. Print results with ' +
        '`console.log`, or end with an expression whose value is the result.';

    switch (mode) {
        case 'generate':
            return (
                `You will be given ${given} and you will respond with generated_code.\n` +
                `Generate executable JavaScript code that programmatically computes the correct ${wanted}.\n` +
                `${sandbox}`
            );
        case 'regenerate':
            return (
                `You are given ${given}, previous_code and error, which is the error the previous ` +
                `code raised. Fix the code so that it programmatically computes the correct ${wanted}, ` +
                `and respond with generated_code.\n${sandbox}`
            );
        case 'answer':
            return (
                `Given ${given}, the final_generated_code that computes the answer, and ` +
                `code_output — what that code printed — provide the final ${wanted}.`
            );
    }
}

/**
 * The code in a reply: the body of the first fenced block when there is one
 * (```js, ```javascript, or bare), else the whole reply.
 */
export function extractCode(text: string): string {
    const fenced = /```[\w-]*\s*\n([\s\S]*?)```/.exec(text);
    return (fenced ? fenced[1] : text).trim();
}

/** Global through which the host asks the context to finish a run. Non-writable. */
const FINISH = '__tsdspyFinish';

/** Budget for reading back logs and describing an error, beyond `timeoutMs`. */
const HOUSEKEEPING_MS = 100;

/** Cap on captured `console` output, in characters. */
const MAX_OUTPUT_CHARS = 64_000;

/**
 * Sets up a fresh context from the inside: a capturing `console`, and the
 * finishing step that reads the code's completion value (or thrown error),
 * follows it if it is a promise, and formats everything as plain strings.
 *
 * It is built by code running *in* the context so that the generated code
 * never holds a function from the host realm: a host function's `constructor`
 * is the host's `Function`, which would hand the code `process` in one line
 * (`console.log.constructor('return process')()`). And every step that
 * touches a value the code produced — a `then` or `toJSON` it defined, a
 * getter on an error — runs inside a timed evaluation, so none of it can hang
 * the host. The host reads back only `state`'s own string and boolean fields.
 */
const SANDBOX_PRELUDE = `(() => {
    const logs = [];
    const state = Object.create(null);
    state.pending = false;
    state.output = undefined;
    state.error = undefined;
    state.logs = '';
    let stored;
    const format = (value) => {
        if (typeof value === 'string') return value;
        try {
            const json = JSON.stringify(value);
            return json === undefined ? String(value) : json;
        } catch {
            return String(value);
        }
    };
    const describe = (error) => {
        try {
            if (error && typeof error === 'object' && 'message' in error) {
                const name = error.name;
                const message = String(error.message);
                return typeof name === 'string' && name ? name + ': ' + message : message;
            }
            return String(error);
        } catch {
            return 'The code threw a value that could not be described.';
        }
    };
    let logged = 0;
    const log = (...args) => {
        // A loop that prints forever would otherwise grow without bound and
        // land, whole, in the next prompt.
        if (logged > ${MAX_OUTPUT_CHARS}) return;
        const line = args.map(format).join(' ');
        logged += line.length + 1;
        logs.push(
            logged > ${MAX_OUTPUT_CHARS}
                ? line.slice(0, Math.max(0, line.length - (logged - ${MAX_OUTPUT_CHARS}))) +
                      '\\n[output truncated]'
                : line
        );
    };
    Object.defineProperty(globalThis, 'console', {
        value: { log, info: log, warn: log, error: log, debug: log },
        writable: true,
        configurable: true,
    });
    const finish = (kind) => {
        if (kind === 'collect') {
            state.logs = logs.join('\\n');
            return;
        }
        const value = stored;
        stored = undefined;
        if (kind === 'error') {
            state.error = describe(value);
            return;
        }
        try {
            const thenable =
                (typeof value === 'object' || typeof value === 'function') &&
                value !== null &&
                typeof value.then === 'function';
            if (!thenable) {
                if (value !== undefined) state.output = format(value);
                return;
            }
            state.pending = true;
            Promise.resolve(value).then(
                (result) => {
                    state.pending = false;
                    if (result !== undefined) state.output = format(result);
                },
                (error) => {
                    state.pending = false;
                    state.error = describe(error);
                }
            );
        } catch (error) {
            state.error = describe(error);
        }
    };
    Object.defineProperty(globalThis, '${FINISH}', { value: finish });
    return { state, stash: (value) => { stored = value; } };
})()`;

/** What {@link SANDBOX_PRELUDE} hands back. Both belong to the context's realm. */
interface SandboxHooks {
    state: { pending: boolean; output?: string; error?: string; logs: string };
    stash(value: unknown): void;
}

/**
 * Run `code` as a script in a fresh `node:vm` context and report what it
 * logged and its completion value. Never throws: failures come back in
 * `error`, worded for a model to read.
 *
 * The context has nothing in it but a capturing `console` (defined inside
 * the context, so no host function is reachable through it), string-to-code
 * evaluation (`eval`, `new Function`) and WebAssembly compilation are
 * disabled, and execution is cut off after `timeoutMs`.
 *
 * The context runs its own microtask queue (`microtaskMode: 'afterEvaluate'`),
 * drained inside the timeout — so `Promise.resolve().then(() => { while
 * (true) {} })` is cut off too, instead of hanging the host's event loop once
 * the script returns. A completion value that is a promise is reported once it
 * settles; with no timers or I/O in the context, one still pending when the
 * microtasks have drained can never settle, and is reported as timed out.
 * **Not a security boundary** — see {@link ProgramOfThought}.
 *
 * `node:vm` is imported on first use, so loading this library never requires
 * it — on an edge runtime without it, only running code fails.
 */
export async function runSandboxed(
    code: string,
    options: { timeoutMs?: number } = {}
): Promise<SandboxResult> {
    const requested = options.timeoutMs ?? 5000;
    // NaN would reach node:vm as its timeout and be rejected there.
    const timeoutMs = Number.isFinite(requested) && requested > 0 ? requested : 5000;
    if (code.trim() === '') return { output: '', error: 'No code was generated.' };

    let vm: typeof NodeVM;
    try {
        vm = await import('node:vm');
    } catch {
        return { output: '', error: 'node:vm is not available in this runtime.' };
    }

    const sandbox = vm.createContext(
        {},
        { codeGeneration: { strings: false, wasm: false }, microtaskMode: 'afterEvaluate' }
    );
    const { state, stash } = vm.runInContext(SANDBOX_PRELUDE, sandbox) as SandboxHooks;

    const startedAt = Date.now();
    const evaluate = (script: string | NodeVM.Script, timeout: number) => {
        const options = { timeout: Math.max(1, timeout), breakOnSigint: false };
        return typeof script === 'string'
            ? vm.runInContext(script, sandbox, options)
            : script.runInContext(sandbox, options);
    };
    const remaining = () => timeoutMs - (Date.now() - startedAt);
    // Reading back logs and describing an error are bookkeeping, not the
    // code's own work: they get a small budget of their own, so a run that
    // used up its whole budget can still report what happened.
    const housekeeping = (kind: 'collect' | 'error') => {
        try {
            evaluate(`${FINISH}('${kind}')`, HOUSEKEEPING_MS);
        } catch {
            // Out of time; report what was collected before.
        }
    };
    const logs = () => {
        housekeeping('collect');
        return typeof state.logs === 'string' ? state.logs : '';
    };
    // Every error is described inside the context — even vm's own timeout,
    // which with a context-local microtask queue is created in the context's
    // realm — since reading a value of the code's making outside a timed
    // evaluation could run its getters unbounded.
    const failure = (error: unknown): SandboxResult => {
        stash(error);
        housekeeping('error');
        return {
            output: logs(),
            error:
                typeof state.error === 'string'
                    ? state.error
                    : 'The code failed in a way that could not be described.',
        };
    };

    try {
        stash(evaluate(new vm.Script(code, { filename: 'generated.js' }), remaining()));
    } catch (error) {
        return failure(error);
    }

    try {
        // Evaluating also drains the context's microtask queue, which is what
        // settles a promise completion value — within the remaining budget.
        evaluate(`${FINISH}('value')`, remaining());
    } catch (error) {
        return failure(error);
    }

    const output = logs();
    if (typeof state.error === 'string') return { output, error: state.error };
    if (state.pending === true) {
        return {
            output,
            error:
                'Script execution timed out: the returned promise never settled ' +
                '(the sandbox has no timers or I/O).',
        };
    }

    const result = typeof state.output === 'string' ? state.output : undefined;
    const full = result === undefined ? output : output ? `${output}\n${result}` : result;
    if (full.trim() === '') {
        return {
            output: full,
            error: 'The code produced no output. Print the result with console.log, or end with an expression.',
        };
    }
    return { output: full };
}
