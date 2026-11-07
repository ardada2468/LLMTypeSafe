import type { z } from 'zod';
import { BaseLM } from '../core/base-lm';
import { Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import {
    extendSignature,
    getSignatureInstructions,
    type Signature,
    type SignatureLike,
    type SignatureSource,
} from '../core/signature';
import type {
    ILanguageModel,
    LLMCallOptions,
    ToolCall,
    ToolSpec,
} from '../types/language-model';
import type { SignatureOutput } from '../types/signature';
import { getInputFieldConfigs, getOutputFieldConfigs } from '../utils/schema';
import { contentToText } from '../utils/content';
import { TsDspyError, ValidationError } from '../core/errors';
import { type TraceSpan } from '../core/trace';
import { Tool } from '../core/tool';
import { invokeLM, providerOptions } from '../core/lm-call';
import { type Adapter } from '../adapters/base';
import { resolveAdapter } from '../adapters/select';
import { Predict } from './predict';
import { ChainOfThought } from './chain-of-thought';

export interface ToolFunction {
    (...args: any[]): Promise<any> | any;
}

/**
 * A tool's argument schema: either a JSON Schema object or a Zod schema.
 *
 * A Zod schema buys argument validation on top of the declaration &mdash; the
 * model's arguments are parsed through it before the tool runs, and a failure
 * comes back as an observation the model can correct.
 */
export type ToolParameterSchema = Record<string, unknown> | z.ZodType;

export interface ToolWithDescription {
    description: string;
    function: ToolFunction;
    /**
     * Argument schema. When present the tool is called with a single object of
     * named, validated arguments; when absent it keeps the historical
     * single-string signature.
     */
    parameters?: ToolParameterSchema;
}

export type ToolDefinition = ToolFunction | ToolWithDescription | Tool;

/** A tool after normalisation, as stored on the agent. */
interface NormalizedTool {
    tool: Tool;
    /** False for tools kept on the historical single-string signature. */
    typed: boolean;
}

/** The tool the agent calls to end the loop, as in DSPy. Reserved. */
const FINISH = 'finish';

/**
 * Convert any accepted tool form into a {@link Tool}.
 *
 * A bare function, or a description without `parameters`, keeps the
 * historical calling convention — one string argument — so it becomes a Tool
 * whose schema is a single `input` string and whose function unwraps it.
 */
function normalizeTool(name: string, definition: ToolDefinition): NormalizedTool {
    if (definition instanceof Tool) {
        return {
            tool: definition.name === name ? definition : definition.withName(name),
            typed: true,
        };
    }

    const described: ToolWithDescription =
        typeof definition === 'function'
            ? { description: `Tool: ${name}`, function: definition }
            : definition;
    const declared = described.parameters;

    if (declared === undefined) {
        const fn = described.function;
        return {
            tool: new Tool({
                name,
                description: described.description,
                parameters: untypedToolSchema(name),
                fn: (args: { input: string }) => fn(args.input),
            }),
            typed: false,
        };
    }

    return {
        tool: new Tool({
            name,
            description: described.description,
            parameters: declared,
            fn: described.function,
        }),
        typed: true,
    };
}

/** Schema advertised for a tool that did not declare one. */
function untypedToolSchema(name: string): Record<string, unknown> {
    return {
        type: 'object',
        properties: {
            input: { type: 'string', description: `Input to the ${name} tool.` },
        },
        required: ['input'],
        additionalProperties: false,
    };
}

/** Events emitted as the reasoning loop runs, for logging or debugging. */
export type ReActEvent =
    | { type: 'thought'; step: number; text: string }
    | { type: 'tool_call'; step: number; tool: string; input: string }
    | { type: 'tool_result'; step: number; tool: string; output: string }
    | { type: 'tool_error'; step: number; tool: string; error: unknown }
    | { type: 'repeated_tool_call'; step: number; tool: string; input: string }
    | { type: 'parse_failed'; step: number; error: unknown };

export interface ReActOptions {
    /**
     * The tools on offer: a record keyed by tool name (functions, described
     * functions, or {@link Tool}s — the key wins over a Tool's own name), or
     * an array of {@link Tool}s named by themselves. `finish` is reserved.
     */
    tools: Record<string, ToolDefinition> | Tool[];
    /**
     * Most tool-using steps before the agent moves on to extracting its
     * answer from what it has gathered. Defaults to 6.
     */
    maxSteps?: number;
    /** Language model to use. Defaults to the configured one. */
    lm?: ILanguageModel;
    /** Prompt format for every step and the extract step. Defaults to the configured one. */
    adapter?: Adapter;
    /** Observe the reasoning loop. */
    onEvent?: (event: ReActEvent) => void;
    /**
     * Never use the model's native tool calling, even when it has it: run
     * every step through the adapter as a `next_thought` / `next_tool_name` /
     * `next_tool_args` prediction instead. Useful for comparing the two paths,
     * or when a provider's tool mode misbehaves on a particular model.
     */
    forceTextMode?: boolean;
}

/** What a ReAct run returns on top of the signature's own outputs. */
export interface ReActExtras {
    /** The extract step's reasoning. */
    reasoning: string;
    /**
     * Every step, DSPy-style: `thought_0`, `tool_name_0`, `tool_args_0`,
     * `observation_0`, `thought_1`, …
     */
    trajectory: Record<string, unknown>;
    /** Steps taken in the loop. */
    steps: number;
}

export type ReActOutput<TSignature extends SignatureSource> = SignatureOutput<TSignature> &
    ReActExtras;

/**
 * Reason and act: a tool-using agent over any signature, after DSPy's
 * `dspy.ReAct`.
 *
 * Each step is a {@link Predict} (`react`) over the signature's inputs plus the
 * `trajectory` so far, producing `next_thought`, `next_tool_name` — one of the
 * tools or `finish` — and `next_tool_args`. The tool runs, its observation is
 * appended to the trajectory, and the loop continues until the agent calls
 * `finish` or `maxSteps` is spent. Then a {@link ChainOfThought} (`extract`)
 * reads the inputs and the whole trajectory and produces the signature's
 * outputs. Nothing assumes an input called `question`: the prompt is built
 * from the signature through the adapter.
 *
 * **Running out of steps is not an error.** As in DSPy, the agent proceeds to
 * the extract step with the trajectory it has, and the result's `steps` tells
 * you the budget was spent. A step whose reply cannot be parsed also ends the
 * loop early, the same way.
 *
 * On a model with native tool calling (`supportsFunctionCalling` and a real
 * `chatWithTools`), the loop instead offers the tools natively in a
 * conversation formatted by the adapter, and the model's final, tool-free
 * reply is parsed directly — the extract step only runs when that reply does
 * not satisfy the signature or the steps run out. `forceTextMode` disables
 * the native path.
 *
 * `react` and `extract` are ordinary predictors, so `namedPredictors()`,
 * optimizers and `dumpState()` see both.
 */
export class ReAct<TSignature extends SignatureSource = typeof Signature> extends Module {
    /** The step predictor: inputs + trajectory → next thought, tool and arguments. */
    readonly react: Predict<any, any>;
    /** The final predictor: inputs + trajectory → the signature's outputs. */
    readonly extract: ChainOfThought<any, any>;

    private tools: Record<string, NormalizedTool>;
    private maxSteps: number;
    private onEvent?: (event: ReActEvent) => void;
    private forceTextMode: boolean;
    private adapter?: Adapter;

    constructor(signature: string | TSignature, options: ReActOptions) {
        super(signature, options.lm);

        this.tools = {};
        const entries: Array<[string, ToolDefinition]> = Array.isArray(options.tools)
            ? options.tools.map((tool) => [tool.name, tool])
            : Object.entries(options.tools);
        for (const [name, tool] of entries) {
            if (name === FINISH) {
                throw new TsDspyError(
                    `"${FINISH}" is reserved: ReAct uses it to end the loop. Rename the tool.`
                );
            }
            this.tools[name] = normalizeTool(name, tool);
        }
        this.maxSteps = options.maxSteps ?? 6;
        this.onEvent = options.onEvent;
        this.forceTextMode = options.forceTextMode ?? false;
        this.adapter = options.adapter;

        const trajectory = {
            trajectory: {
                description: 'The thoughts, tool calls and observations so far',
                type: 'string',
            },
        };
        this.react = new Predict(
            extendSignature(signature, {
                appendInputs: trajectory,
                removeOutputs: Object.keys(getOutputFieldConfigs(signature)),
                appendOutputs: {
                    next_thought: { description: 'Reasoning about what to do next' },
                    next_tool_name: {
                        description: 'The tool to call next',
                        type: 'enum',
                        values: [...Object.keys(this.tools), FINISH],
                    },
                    // Declared as text holding JSON, not as an object: OpenAI's
                    // strict JSON Schema cannot express a free-form object, so an
                    // `object` field would be pinned to `{}` on every structured
                    // provider. The arguments are parsed — and for a typed tool,
                    // validated — when the tool runs.
                    next_tool_args: {
                        description: 'Arguments for the tool, as a JSON object; {} for finish',
                    },
                },
                instructions: this.stepInstructions(signature),
            }),
            { adapter: this.adapter }
        );
        this.extract = new ChainOfThought(
            extendSignature(signature, { appendInputs: trajectory }),
            { adapter: this.adapter }
        );
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction<ReActOutput<TSignature>> & ReActOutput<TSignature>> {
        const prediction = await this.traced(inputs, (span) => {
            const lm = this.resolveLM(options);
            return this.usesNativeTools(lm)
                ? this.nativeLoop(lm, inputs, options, span)
                : this.textLoop(lm, inputs, options);
        });

        return prediction as Prediction<ReActOutput<TSignature>> & ReActOutput<TSignature>;
    }

    /**
     * Native tool calling needs a model that advertises it, a real
     * `chatWithTools` implementation, and at least one tool to offer. Anything
     * else runs the adapter loop, which works on any completion model.
     *
     * The identity check against `BaseLM.prototype` matters: the base class
     * supplies a text-only `chatWithTools`, so a plain `typeof` test passes for
     * every subclass. A model that inherited the default but reported
     * `supportsFunctionCalling: true` would otherwise take the native path,
     * never see its tools, and answer without calling one.
     */
    private usesNativeTools(lm: ILanguageModel): boolean {
        if (this.forceTextMode) return false;
        if (Object.keys(this.tools).length === 0) return false;
        if (typeof lm.chatWithTools !== 'function') return false;
        if (lm.chatWithTools === BaseLM.prototype.chatWithTools) return false;
        return lm.getCapabilities().supportsFunctionCalling === true;
    }

    /** Options for the inner predictors: the resolved model travels with them. */
    private innerOptions(lm: ILanguageModel, options?: LLMCallOptions): LLMCallOptions {
        return { ...options, lm };
    }

    // ------------------------------------------------------------------ text

    /** The adapter loop: one `react` prediction per step, then `extract`. */
    private async textLoop(
        lm: ILanguageModel,
        inputs: Record<string, any>,
        options: LLMCallOptions | undefined
    ): Promise<Record<string, any>> {
        const trajectory: Record<string, unknown> = {};
        const previousToolCalls = new Set<string>();
        const stepOptions = this.innerOptions(lm, options);
        let steps = 0;

        for (let step = 0; step < this.maxSteps; step++) {
            let decision: Record<string, any>;
            try {
                decision = await this.react.forward(
                    { ...inputs, trajectory: formatTrajectory(trajectory) },
                    stepOptions
                );
            } catch (error) {
                // As in DSPy: a step the model could not get right ends the
                // loop, and the extract step works with what was gathered.
                if (!(error instanceof ValidationError)) throw error;
                this.emit({ type: 'parse_failed', step, error });
                break;
            }
            steps = step + 1;

            const thought = String(decision.next_thought ?? '');
            const toolName = String(decision.next_tool_name);
            this.emit({ type: 'thought', step, text: thought });

            trajectory[`thought_${step}`] = thought;
            trajectory[`tool_name_${step}`] = toolName;

            if (toolName === FINISH) {
                trajectory[`tool_args_${step}`] = {};
                trajectory[`observation_${step}`] = 'Completed.';
                break;
            }

            const rawArgs = decision.next_tool_args;
            const parsedArgs = this.parseArguments(toolName, rawArgs);
            trajectory[`tool_args_${step}`] = parsedArgs.ok ? parsedArgs.args : rawArgs;

            const description = parsedArgs.ok
                ? describeArguments(parsedArgs.args)
                : String(rawArgs ?? '');
            const key = `${toolName}:${description}`;
            if (previousToolCalls.has(key)) {
                this.emit({
                    type: 'repeated_tool_call',
                    step,
                    tool: toolName,
                    input: description,
                });
                trajectory[`observation_${step}`] =
                    'You have already made this tool call. Please move to the next step.';
                continue;
            }
            previousToolCalls.add(key);
            this.emit({ type: 'tool_call', step, tool: toolName, input: description });

            trajectory[`observation_${step}`] = parsedArgs.ok
                ? await this.runTool(toolName, parsedArgs.args, step)
                : this.reportToolError(toolName, parsedArgs.error, step);
        }

        return this.finish(inputs, trajectory, steps, stepOptions);
    }

    /** Run the extract step over the trajectory and assemble the result. */
    private async finish(
        inputs: Record<string, any>,
        trajectory: Record<string, unknown>,
        steps: number,
        options: LLMCallOptions
    ): Promise<Record<string, any>> {
        const extracted = await this.extract.forward(
            { ...inputs, trajectory: formatTrajectory(trajectory) },
            options
        );
        return { ...extracted.toObject(), trajectory, steps };
    }

    /**
     * The step predictor's instructions: the signature's own, then how the
     * loop works and which tools — with their argument schemas — are on offer.
     */
    private stepInstructions(signature: SignatureLike): string {
        const inputs = Object.keys(getInputFieldConfigs(signature))
            .map((name) => `\`${name}\``)
            .join(', ');
        const outputs = Object.keys(getOutputFieldConfigs(signature))
            .map((name) => `\`${name}\``)
            .join(', ');
        const own = getSignatureInstructions(signature);

        const tools = [
            ...Object.values(this.tools).map(({ tool }) => tool),
            new Tool({
                name: FINISH,
                description: `Marks the task as complete. That is, signals that all information for producing the outputs, i.e. ${outputs}, are now available to be extracted.`,
                fn: () => 'Completed.',
            }),
        ];

        return [
            ...(own ? [own, ''] : []),
            `You are an Agent. In each episode, you will be given the fields ${inputs} as input. And you can see your past trajectory so far.`,
            `Your goal is to use one or more of the supplied tools to collect any necessary information for producing ${outputs}.`,
            '',
            'To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.',
            'After each tool call, you receive a resulting observation, which gets appended to your trajectory.',
            '',
            'When writing next_thought, you may reason about the current situation and plan for future steps.',
            'When selecting the next_tool_name and its next_tool_args, the tool must be one of:',
            '',
            ...tools.map(
                (tool, index) =>
                    `(${index + 1}) ${tool.name}, whose description is <desc>${tool.description}</desc>. It takes arguments ${JSON.stringify(
                        (tool.parameters as { properties?: unknown }).properties ?? {}
                    )}.`
            ),
            '',
            'When providing `next_tool_args`, the value inside the field must be in JSON format.',
        ].join('\n');
    }

    // ---------------------------------------------------------------- native

    /**
     * The native tool-calling loop. The conversation opens with the adapter's
     * system and user messages for the signature itself; tools travel as
     * provider tool declarations. Every model call is reported to `span`.
     */
    private async nativeLoop(
        lm: ILanguageModel,
        inputs: Record<string, any>,
        options: LLMCallOptions | undefined,
        span: TraceSpan | undefined
    ): Promise<Record<string, any>> {
        const signature = this.requireSignature();
        const adapter = resolveAdapter(lm, options?.adapter, this.adapter);
        const own = getSignatureInstructions(signature);
        const messages = adapter.formatMessages({
            signature,
            inputs,
            instructions: [
                ...(own ? [own, ''] : []),
                'You have tools available; call them as needed to gather what you need.',
                'When you have everything you need, reply with the final answer and no tool call.',
            ].join('\n'),
        });

        const trajectory: Record<string, unknown> = {};
        let entry = 0;
        const previousToolCalls = new Set<string>();
        const callOptions: LLMCallOptions = {
            ...providerOptions(options),
            tools: this.toolSpecs(),
        };

        for (let step = 0; step < this.maxSteps; step++) {
            const result = await invokeLM(
                {
                    lm,
                    method: 'chatWithTools',
                    traceText: messages
                        .map((m) => `${m.role}: ${contentToText(m.content)}`)
                        .join('\n\n'),
                    // A snapshot for callbacks: the loop keeps appending to
                    // `messages`, which would rewrite the event after the fact.
                    messages: [...messages],
                    options: callOptions,
                    span,
                    moduleId: this.moduleId,
                },
                () => lm.chatWithTools!(messages, callOptions),
                (reply) => reply.content ?? ''
            );
            const text = result.content ?? '';
            if (text.trim().length > 0) {
                this.emit({ type: 'thought', step, text });
            }

            const toolCalls = result.toolCalls ?? [];
            if (toolCalls.length > 0) {
                messages.push({ role: 'assistant', content: text, toolCalls });
                for (const call of toolCalls) {
                    const observation = await this.runNativeCall(call, step, previousToolCalls);
                    trajectory[`thought_${entry}`] = text;
                    trajectory[`tool_name_${entry}`] = call.name;
                    trajectory[`tool_args_${entry}`] = call.arguments;
                    trajectory[`observation_${entry}`] = observation;
                    entry++;
                    messages.push({
                        role: 'tool',
                        name: call.name,
                        toolCallId: call.id,
                        content: observation,
                    });
                }
                continue;
            }

            if (text.trim().length === 0) {
                // No assistant turn is recorded here on purpose: an empty
                // assistant message is rejected outright by some providers,
                // which would turn this recovery branch into a hard failure.
                messages.push({
                    role: 'user',
                    content:
                        'You returned neither a tool call nor an answer. Call a tool or give the final answer.',
                });
                continue;
            }

            try {
                const parsed = adapter.parseOutputs(signature, text);
                return { ...parsed, trajectory, steps: step + 1 };
            } catch (error) {
                if (!(error instanceof ValidationError)) throw error;
                this.emit({ type: 'parse_failed', step, error });
                // The answer is in there but not in the signature's shape:
                // hand the whole exchange to the extract step.
                trajectory[`thought_${entry}`] = text;
                return this.finish(
                    inputs,
                    trajectory,
                    step + 1,
                    this.innerOptions(lm, options)
                );
            }
        }

        return this.finish(inputs, trajectory, this.maxSteps, this.innerOptions(lm, options));
    }

    /** Execute one model-requested call, honouring the repeat guard. */
    private async runNativeCall(
        call: ToolCall,
        step: number,
        previousToolCalls: Set<string>
    ): Promise<string> {
        const description = describeArguments(call.arguments);
        const key = `${call.name}:${description}`;

        if (previousToolCalls.has(key)) {
            this.emit({
                type: 'repeated_tool_call',
                step,
                tool: call.name,
                input: description,
            });
            return 'You have already made this tool call. Please move to the next step.';
        }

        previousToolCalls.add(key);
        this.emit({ type: 'tool_call', step, tool: call.name, input: description });

        // A provider that ships arguments as a JSON string (OpenAI) reports a
        // payload it could not parse as empty arguments with the original text
        // kept. Running the tool on those empty arguments would feed the model a
        // plausible-looking observation derived from nothing, so say what went
        // wrong instead and let it retry.
        if (unparsedArguments(call)) {
            const error = new Error(`Tool arguments were not valid JSON: ${call.rawArguments}`);
            return this.reportToolError(call.name, error, step);
        }

        return this.runTool(call.name, call.arguments, step);
    }

    /** The tool declarations sent to the provider on every native turn. */
    private toolSpecs(): ToolSpec[] {
        return Object.values(this.tools).map(({ tool }) => tool.toSpec());
    }

    // --------------------------------------------------------------- shared

    /**
     * Turn the step's `next_tool_args` into named arguments. JSON is the
     * documented form; a tool with a single argument also accepts the bare
     * value, which is what models tend to write for one.
     */
    private parseArguments(
        toolName: string,
        raw: unknown
    ): { ok: true; args: Record<string, unknown> } | { ok: false; error: unknown } {
        const tool = this.tools[toolName];
        if (!tool) return { ok: false, error: new Error(this.unknownToolMessage(toolName)) };
        if (raw !== null && typeof raw === 'object' && !Array.isArray(raw)) {
            return { ok: true, args: raw as Record<string, unknown> };
        }
        try {
            return { ok: true, args: parseTextArguments(tool, String(raw ?? '')) };
        } catch (error) {
            return { ok: false, error };
        }
    }

    /**
     * Run a tool from a string of arguments, as the step predictor writes
     * them. Used by tests and kept for callers driving tools by hand.
     */
    protected async executeTool(
        toolName: string,
        input: string,
        step: number
    ): Promise<string> {
        const tool = this.tools[toolName];
        if (!tool) {
            return this.unknownToolMessage(toolName);
        }

        let args: Record<string, unknown>;
        try {
            args = tool.typed ? parseTextArguments(tool, input) : { input };
        } catch (error) {
            return this.reportToolError(toolName, error, step);
        }

        return this.runTool(toolName, args, step);
    }

    /** Validate arguments, call the tool, and turn whatever happens into an observation. */
    private async runTool(
        toolName: string,
        args: Record<string, unknown>,
        step: number
    ): Promise<string> {
        const tool = this.tools[toolName];
        if (!tool) {
            return this.unknownToolMessage(toolName);
        }

        try {
            if (!tool.typed && typeof args.input !== 'string') {
                // An untyped tool is declared as taking one string called
                // `input`. A model that sends something else would otherwise
                // reach the tool as `undefined`.
                throw new Error(
                    `Expected a single string argument named "input". Received: ${JSON.stringify(args)}`
                );
            }

            // `Tool.call` validates Zod-declared arguments and reports the
            // call to any registered callbacks.
            const result = await tool.tool.call(args);
            const output = typeof result === 'string' ? result : stringifyResult(result);
            this.emit({ type: 'tool_result', step, tool: toolName, output });
            return output;
        } catch (error) {
            return this.reportToolError(toolName, error, step);
        }
    }

    private reportToolError(toolName: string, error: unknown, step: number): string {
        this.emit({ type: 'tool_error', step, tool: toolName, error });
        return `Error executing ${toolName}: ${error instanceof Error ? error.message : String(error)}`;
    }

    private unknownToolMessage(toolName: string): string {
        return `Error: Tool '${toolName}' not found. Available tools: ${Object.keys(this.tools).join(', ')}`;
    }

    private requireSignature(): SignatureLike {
        if (!this.signature) throw new Error('No signature provided');
        return this.signature;
    }

    private emit(event: ReActEvent): void {
        this.onEvent?.(event);
    }
}

/**
 * The trajectory as the text the `trajectory` input carries: one
 * `key: value` line per entry, arguments as JSON.
 */
export function formatTrajectory(trajectory: Record<string, unknown>): string {
    return Object.entries(trajectory)
        .map(
            ([key, value]) =>
                `${key}: ${typeof value === 'string' ? value : stringifyResult(value)}`
        )
        .join('\n');
}

/** Objects and arrays are worth sending back as JSON; everything else stringifies. */
function stringifyResult(result: unknown): string {
    if (result === null || typeof result !== 'object') return String(result);
    try {
        return JSON.stringify(result);
    } catch {
        return String(result);
    }
}

/**
 * Did the provider fail to parse this call's arguments?
 *
 * Only providers that transmit arguments as a JSON string can hit this, and they
 * signal it by reporting empty `arguments` while keeping the text they could not
 * parse in `rawArguments`.
 */
function unparsedArguments(call: ToolCall): boolean {
    const raw = call.rawArguments?.trim();
    if (!raw || raw === '{}') return false;
    return Object.keys(call.arguments ?? {}).length === 0;
}

/**
 * Turn a line of argument text into named arguments.
 *
 * JSON is the documented form; a single-property schema also accepts the bare
 * value, coerced to the declared scalar type, so a `number` argument does not
 * fail validation purely because the model wrote `21` rather than `{"n": 21}`.
 */
function parseTextArguments(tool: NormalizedTool, input: string): Record<string, unknown> {
    const trimmed = input.trim();
    if (trimmed.startsWith('{')) {
        try {
            const parsed = JSON.parse(trimmed);
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                return parsed as Record<string, unknown>;
            }
        } catch {
            // Fall through to the single-property case.
        }
    }

    const names = propertyNames(tool);
    if (names.length === 0 && (trimmed === '' || trimmed === '{}')) return {};
    if (names.length === 1) {
        return { [names[0]]: coerceScalar(trimmed, propertyType(tool, names[0])) };
    }

    throw new Error(
        `Arguments must be a JSON object with these keys: ${names.join(', ')}. Received: ${trimmed}`
    );
}

function coerceScalar(value: string, type: string | undefined): unknown {
    switch (type) {
        case 'number':
        case 'integer': {
            const parsed = Number(value);
            return Number.isNaN(parsed) ? value : parsed;
        }
        case 'boolean':
            if (value === 'true') return true;
            if (value === 'false') return false;
            return value;
        default:
            return value;
    }
}

/** Argument names declared by a tool's JSON Schema, in declaration order. */
function propertyNames(tool: NormalizedTool): string[] {
    return Object.keys(schemaProperties(tool));
}

/** The declared JSON Schema `type` of one argument, when it is a simple scalar. */
function propertyType(tool: NormalizedTool, name: string): string | undefined {
    const property = schemaProperties(tool)[name];
    if (!property || typeof property !== 'object') return undefined;
    const type = (property as { type?: unknown }).type;
    return typeof type === 'string' ? type : undefined;
}

function schemaProperties(tool: NormalizedTool): Record<string, unknown> {
    const properties = tool.tool.parameters.properties;
    if (!properties || typeof properties !== 'object') return {};
    return properties as Record<string, unknown>;
}

/** Stable, human-readable rendering of a tool call's arguments. */
function describeArguments(args: Record<string, unknown>): string {
    const keys = Object.keys(args ?? {});
    if (keys.length === 1 && typeof args[keys[0]] === 'string') {
        return args[keys[0]] as string;
    }
    const sorted: Record<string, unknown> = {};
    for (const key of keys.sort()) {
        sorted[key] = args[key];
    }
    try {
        return JSON.stringify(sorted);
    } catch {
        return String(args);
    }
}
