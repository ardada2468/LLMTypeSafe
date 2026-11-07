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

/**
 * Run `code` as a script in a fresh `node:vm` context and report what it
 * logged and its completion value. Never throws: failures come back in
 * `error`, worded for a model to read.
 *
 * The context has nothing in it but a capturing `console`, string-to-code
 * evaluation (`eval`, `new Function`) and WebAssembly compilation are
 * disabled, and synchronous execution is cut off after `timeoutMs`. A
 * completion value that is a promise is awaited for whatever is left of the
 * budget. **Not a security boundary** — see {@link ProgramOfThought}.
 *
 * `node:vm` is imported on first use, so loading this library never requires
 * it — on an edge runtime without it, only running code fails.
 */
export async function runSandboxed(
    code: string,
    options: { timeoutMs?: number } = {}
): Promise<SandboxResult> {
    const timeoutMs = options.timeoutMs ?? 5000;
    if (code.trim() === '') return { output: '', error: 'No code was generated.' };

    let vm: typeof NodeVM;
    try {
        vm = await import('node:vm');
    } catch {
        return { output: '', error: 'node:vm is not available in this runtime.' };
    }

    const logs: string[] = [];
    const log = (...args: unknown[]) => {
        logs.push(args.map(formatValue).join(' '));
    };
    const sandbox = vm.createContext(
        { console: { log, info: log, warn: log, error: log, debug: log } },
        { codeGeneration: { strings: false, wasm: false } }
    );

    const startedAt = Date.now();
    let value: unknown;
    try {
        value = new vm.Script(code, { filename: 'generated.js' }).runInContext(sandbox, {
            timeout: timeoutMs,
            breakOnSigint: false,
        });
        if (isThenable(value)) {
            value = await withTimeout(value, Math.max(0, timeoutMs - (Date.now() - startedAt)));
        }
    } catch (error) {
        return { output: logs.join('\n'), error: describeError(error) };
    }

    const lines = [...logs];
    if (value !== undefined) lines.push(formatValue(value));
    const output = lines.join('\n');
    if (output.trim() === '') {
        return {
            output,
            error: 'The code produced no output. Print the result with console.log, or end with an expression.',
        };
    }
    return { output };
}

function isThenable(value: unknown): value is PromiseLike<unknown> {
    return (
        (typeof value === 'object' || typeof value === 'function') &&
        value !== null &&
        typeof (value as { then?: unknown }).then === 'function'
    );
}

function withTimeout<T>(promise: PromiseLike<T>, ms: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`Script execution timed out after ${ms}ms`)),
            ms
        );
        Promise.resolve(promise).then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error: unknown) => {
                clearTimeout(timer);
                reject(error);
            }
        );
    });
}

/**
 * An error as the model should read it. Errors thrown inside the context are
 * instances of *its* `Error`, not ours, so `instanceof` is no use; read the
 * fields structurally.
 */
function describeError(error: unknown): string {
    if (error && typeof error === 'object' && 'message' in error) {
        const name = (error as { name?: unknown }).name;
        const message = String((error as { message: unknown }).message);
        return typeof name === 'string' && name ? `${name}: ${message}` : message;
    }
    return String(error);
}

function formatValue(value: unknown): string {
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value) ?? String(value);
    } catch {
        return String(value);
    }
}
