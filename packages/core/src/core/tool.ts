import { z } from 'zod';
import type { ToolSpec } from '../types/language-model';
import { emit, newCallFrame, runInFrame } from './callbacks';
import { getCallbacks } from './config';
import { TsDspyError } from './errors';

/**
 * A tool's argument schema: a JSON Schema object, or a Zod schema. A Zod
 * schema buys validation on top of the declaration — arguments are parsed
 * through it before the function runs.
 */
export type ToolParameters = Record<string, unknown> | z.ZodType;

export interface ToolOptions {
    /** Defaults to the function's own name. */
    name?: string;
    /** What the tool does, as the model will read it. Defaults to `Tool: <name>`. */
    description?: string;
    /** Argument schema. Defaults to an object accepting no declared arguments. */
    parameters?: ToolParameters;
}

export interface ToolInit<TArgs, TResult> extends ToolOptions {
    name: string;
    fn: (args: TArgs) => TResult | Promise<TResult>;
}

function isZodSchema(value: unknown): value is z.ZodType {
    return (
        typeof value === 'object' &&
        value !== null &&
        typeof (value as { safeParse?: unknown }).safeParse === 'function'
    );
}

const NO_ARGUMENTS: Record<string, unknown> = { type: 'object', properties: {} };

/**
 * A function a model can call: a name, a description, a JSON Schema for its
 * arguments, and the implementation.
 *
 * ```ts
 * const weather = Tool.fromFunction(async ({ city }) => lookup(city), {
 *     name: 'weather',
 *     description: 'Current weather for a city',
 *     parameters: z.object({ city: z.string() }),
 * });
 * ```
 *
 * The provider-neutral currency for tools: `RespAct` accepts these directly,
 * {@link Tool.toSpec} produces what `LLMCallOptions.tools` takes, and
 * {@link Tool.call} validates arguments and reports `onToolStart`/`onToolEnd`
 * to callbacks however the tool is invoked.
 */
export class Tool<TArgs = any, TResult = unknown> {
    readonly name: string;
    readonly description: string;
    /** JSON Schema for the arguments, as advertised to the provider. */
    readonly parameters: Record<string, unknown>;
    readonly fn: (args: TArgs) => TResult | Promise<TResult>;
    /** Present when the parameters were declared with Zod. */
    readonly validator?: z.ZodType;

    constructor(init: ToolInit<TArgs, TResult>) {
        if (!init.name) {
            throw new TsDspyError('A Tool needs a name.');
        }
        this.name = init.name;
        this.description = init.description ?? `Tool: ${init.name}`;
        this.fn = init.fn;

        const declared = init.parameters;
        if (isZodSchema(declared)) {
            this.validator = declared;
            // `io: 'input'` describes what the model must send, before any
            // transforms or defaults the schema applies.
            this.parameters = z.toJSONSchema(declared, { io: 'input' }) as Record<
                string,
                unknown
            >;
        } else {
            this.parameters = declared ?? NO_ARGUMENTS;
        }
    }

    /**
     * Wrap a function as a tool. The name defaults to the function's own,
     * so a named function needs no options at all.
     *
     * @throws {TsDspyError} when no name is given and the function is anonymous.
     */
    static fromFunction<TArgs = any, TResult = unknown>(
        fn: (args: TArgs) => TResult | Promise<TResult>,
        options: ToolOptions = {}
    ): Tool<TArgs, TResult> {
        const name = options.name ?? fn.name;
        if (!name) {
            throw new TsDspyError(
                'Tool.fromFunction needs a name: pass { name } or use a named function.'
            );
        }
        return new Tool({ ...options, name, fn });
    }

    /**
     * Validate `args` (when the schema is Zod) and run the tool.
     *
     * @throws {TsDspyError} when the arguments fail validation; otherwise
     *   whatever the function throws.
     */
    async call(args: unknown): Promise<TResult> {
        const callbacks = getCallbacks();
        if (callbacks.length === 0) {
            return this.fn(this.validate(args));
        }

        const frame = newCallFrame();
        const startTime = Date.now();
        const base = {
            callId: frame.callId,
            parentCallId: frame.parentCallId,
            startTime,
            toolName: this.name,
        };
        emit('onToolStart', { ...base, args }, callbacks);

        try {
            const outputs = await runInFrame(frame, async () => this.fn(this.validate(args)));
            const endTime = Date.now();
            emit(
                'onToolEnd',
                { ...base, endTime, durationMs: endTime - startTime, outputs },
                callbacks
            );
            return outputs;
        } catch (error) {
            const endTime = Date.now();
            emit(
                'onToolEnd',
                { ...base, endTime, durationMs: endTime - startTime, error },
                callbacks
            );
            throw error;
        }
    }

    /** The declaration sent to a provider in `LLMCallOptions.tools`. */
    toSpec(): ToolSpec {
        return { name: this.name, description: this.description, parameters: this.parameters };
    }

    /** The same tool under another name. */
    withName(name: string): Tool<TArgs, TResult> {
        const copy = Object.create(Object.getPrototypeOf(this)) as Tool<TArgs, TResult>;
        Object.assign(copy, this, { name });
        return copy;
    }

    private validate(args: unknown): TArgs {
        if (!this.validator) return args as TArgs;
        const result = this.validator.safeParse(args);
        if (result.success) return result.data as TArgs;
        throw new TsDspyError(
            `Invalid arguments: ${result.error.issues
                .map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`)
                .join('; ')}`
        );
    }
}
