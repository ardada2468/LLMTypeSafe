import { z } from 'zod';
import { BaseLM } from '../core/base-lm';
import { Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { type Signature, type SignatureSource } from '../core/signature';
import type {
    ChatMessage,
    ILanguageModel,
    LLMCallOptions,
    ToolCall,
    ToolSpec,
} from '../types/language-model';
import type { SignatureOutput } from '../types/signature';
import { parseOutput as utilParseOutput } from '../utils/parsing';
import { getOutputFieldConfigs } from '../utils/schema';
import { ValidationError } from '../core/errors';
import { buildRepairObservation } from '../core/repair';
import { type TraceSpan } from '../core/trace';

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

export type ToolDefinition = ToolFunction | ToolWithDescription;

/** A tool after normalisation, as stored on the agent. */
interface NormalizedTool {
    description: string;
    function: ToolFunction;
    /** JSON Schema advertised to the provider. Synthesised for untyped tools. */
    parameters: Record<string, unknown>;
    /** Present only when the caller declared a Zod schema. */
    validator?: z.ZodType;
    /** False for tools kept on the historical single-string signature. */
    typed: boolean;
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

function isZodSchema(value: unknown): value is z.ZodType {
    return (
        typeof value === 'object' &&
        value !== null &&
        typeof (value as { safeParse?: unknown }).safeParse === 'function'
    );
}

/** Events emitted as the reasoning loop runs, for logging or debugging. */
export type RespActEvent =
    | { type: 'thought'; step: number; text: string }
    | { type: 'tool_call'; step: number; tool: string; input: string }
    | { type: 'tool_result'; step: number; tool: string; output: string }
    | { type: 'tool_error'; step: number; tool: string; error: unknown }
    | { type: 'repeated_tool_call'; step: number; tool: string; input: string }
    | { type: 'parse_failed'; step: number; error: unknown };

export interface RespActOptions {
    tools: Record<string, ToolDefinition>;
    maxSteps?: number;
    /** Language model to use. Defaults to the globally configured one. */
    lm?: ILanguageModel;
    /** Observe the reasoning loop. Replaces the previous console logging. */
    onEvent?: (event: RespActEvent) => void;
    /**
     * Force the text-prompting loop even on a model that supports native tool
     * calling. Useful for comparing the two paths, or when a provider's tool
     * mode misbehaves on a particular model.
     */
    forceTextMode?: boolean;
}

export class RespAct<TSignature extends SignatureSource = typeof Signature> extends Module {
    private tools: Record<string, NormalizedTool>;
    private maxSteps: number;
    private onEvent?: (event: RespActEvent) => void;
    private forceTextMode: boolean;

    constructor(signature: string | TSignature, options: RespActOptions) {
        super(signature, options.lm);

        this.tools = {};
        for (const [name, tool] of Object.entries(options.tools)) {
            const described =
                typeof tool === 'function'
                    ? { description: `Tool: ${name}`, function: tool }
                    : tool;
            const declared = described.parameters;

            this.tools[name] = {
                description: described.description,
                function: described.function,
                typed: declared !== undefined,
                validator: isZodSchema(declared) ? declared : undefined,
                parameters: !declared
                    ? untypedToolSchema(name)
                    : isZodSchema(declared)
                      ? (z.toJSONSchema(declared, { io: 'input' }) as Record<string, unknown>)
                      : declared,
            };
        }
        this.maxSteps = options.maxSteps ?? 6;
        this.onEvent = options.onEvent;
        this.forceTextMode = options.forceTextMode ?? false;
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<
        Prediction<SignatureOutput<TSignature> & { steps: number }> &
            SignatureOutput<TSignature> & { steps: number }
    > {
        const prediction = await this.traced(inputs, (span) =>
            this.usesNativeTools()
                ? this.nativeLoop(inputs, options, span)
                : this.textLoop(inputs, options, span)
        );

        return prediction as Prediction<SignatureOutput<TSignature> & { steps: number }> &
            SignatureOutput<TSignature> & { steps: number };
    }

    /**
     * Native tool calling needs a model that advertises it, a real
     * `chatWithTools` implementation, and at least one tool to offer. Anything
     * else falls back to the text-prompting loop, which works on any completion
     * model.
     *
     * The identity check against `BaseLM.prototype` matters: the base class
     * supplies a text-only `chatWithTools`, so a plain `typeof` test passes for
     * every subclass. A model that inherited the default but reported
     * `supportsFunctionCalling: true` — free to do before anything read the flag
     * — would otherwise take the native path, never see its tools, and answer
     * from the first reply without calling one.
     */
    private usesNativeTools(): boolean {
        if (this.forceTextMode) return false;
        if (Object.keys(this.tools).length === 0) return false;
        const lm = this.lm as ILanguageModel | undefined;
        if (!lm || typeof lm.chatWithTools !== 'function') return false;
        if (lm.chatWithTools === BaseLM.prototype.chatWithTools) return false;
        return lm.getCapabilities().supportsFunctionCalling === true;
    }

    // ---------------------------------------------------------------- native

    /**
     * The native tool-calling loop. Every model call is reported to `span` when
     * tracing is on, so a trace holds each turn's prompt and reply.
     */
    private async nativeLoop(
        inputs: Record<string, any>,
        options: LLMCallOptions | undefined,
        span: TraceSpan | undefined
    ): Promise<Record<string, any>> {
        const messages: ChatMessage[] = [
            { role: 'system', content: this.buildNativePrompt() },
            { role: 'user', content: this.questionOf(inputs) },
        ];
        const previousToolCalls = new Set<string>();
        const callOptions: LLMCallOptions = { ...options, tools: this.toolSpecs() };

        for (let step = 0; step < this.maxSteps; step++) {
            const prompt = messages.map((m) => `${m.role}: ${m.content}`).join('\n\n');
            span?.startCall(prompt);
            const result = await this.lm.chatWithTools!(messages, callOptions);
            span?.endCall(result.content ?? '');
            const text = result.content ?? '';
            if (text.trim().length > 0) {
                this.emit({ type: 'thought', step, text });
            }

            const toolCalls = result.toolCalls ?? [];
            if (toolCalls.length > 0) {
                messages.push({ role: 'assistant', content: text, toolCalls });
                for (const call of toolCalls) {
                    const observation = await this.runNativeCall(call, step, previousToolCalls);
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

            messages.push({ role: 'assistant', content: text });

            let parsed: Record<string, any> | null = null;
            try {
                parsed = this.parseOutput(this.extractFinalAnswer(text) || text);
            } catch (error) {
                this.emit({ type: 'parse_failed', step, error });
                if (error instanceof ValidationError && step < this.maxSteps - 1) {
                    messages.push({
                        role: 'user',
                        content: this.malformedAnswerMessage(error),
                    });
                    continue;
                }
                throw error;
            }

            return { ...parsed, steps: step + 1 };
        }

        throw this.exhaustedError();
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
            this.emit({ type: 'tool_error', step, tool: call.name, error });
            return `Error executing ${call.name}: ${error.message}`;
        }

        return this.runTool(call.name, call.arguments, step);
    }

    /** The tool declarations sent to the provider on every native turn. */
    private toolSpecs(): ToolSpec[] {
        return Object.entries(this.tools).map(([name, tool]) => ({
            name,
            description: tool.description,
            parameters: tool.parameters,
        }));
    }

    // ------------------------------------------------------------------ text

    /**
     * The text-prompting loop, used whenever native tool calling is unavailable.
     * Every model call is reported to `span` when tracing is on.
     */
    private async textLoop(
        inputs: Record<string, any>,
        options: LLMCallOptions | undefined,
        span: TraceSpan | undefined
    ): Promise<Record<string, any>> {
        let conversation = this.buildInitialPrompt(inputs);
        const previousToolCalls = new Set<string>();

        for (let step = 0; step < this.maxSteps; step++) {
            const prompt = conversation + '\n\nThought:';
            span?.startCall(prompt);
            const response = await this.lm.generate(prompt, options);
            span?.endCall(response);
            conversation += `\n\nThought: ${response}`;
            this.emit({ type: 'thought', step, text: response });

            // Tool use takes priority: a response can mention both an action and a
            // premature final answer, and the action is what advances the loop.
            const toolCall = this.extractToolCall(response);
            if (toolCall) {
                const toolCallKey = `${toolCall.tool}:${toolCall.input}`;
                if (previousToolCalls.has(toolCallKey)) {
                    this.emit({
                        type: 'repeated_tool_call',
                        step,
                        tool: toolCall.tool,
                        input: toolCall.input,
                    });
                    conversation +=
                        '\n\nObservation: You have already made this tool call. Please move to the next step.';
                    continue;
                }
                previousToolCalls.add(toolCallKey);
                this.emit({
                    type: 'tool_call',
                    step,
                    tool: toolCall.tool,
                    input: toolCall.input,
                });

                const observation = await this.executeTool(toolCall.tool, toolCall.input, step);
                conversation += `\n\nObservation: ${observation}`;
                continue;
            }

            if (!/final answer:/i.test(response)) {
                continue;
            }

            const rawAnswer = this.extractFinalAnswer(response);
            let parsed: Record<string, any> | null = null;
            try {
                parsed = this.parseOutput(rawAnswer);
            } catch (error) {
                this.emit({ type: 'parse_failed', step, error });
                // A malformed final answer is recoverable: tell the model what
                // shape it owes us and let it try again on the next step.
                if (error instanceof ValidationError && step < this.maxSteps - 1) {
                    conversation += `\n\nObservation: ${buildRepairObservation(error)}`;
                    continue;
                }
                throw error;
            }

            return { ...parsed, steps: step + 1 };
        }

        throw this.exhaustedError();
    }

    // --------------------------------------------------------------- shared

    protected parseOutput(rawOutput: unknown): Record<string, any> {
        if (!this.signature) {
            throw new Error('No signature provided for RespAct parsing');
        }
        const outputText =
            typeof rawOutput === 'string' ? rawOutput : JSON.stringify(rawOutput);
        return utilParseOutput(this.signature, outputText);
    }

    private exhaustedError(): Error {
        return new Error(
            `RespAct exceeded maximum steps (${this.maxSteps}) without producing a valid final answer`
        );
    }

    private malformedAnswerMessage(error: ValidationError): string {
        const fieldList = error.issues.map((issue) => issue.field).join(', ');
        return `Your Final Answer was missing or malformed for: ${fieldList}. Provide a Final Answer with every required field on its own "field: value" line.`;
    }

    private emit(event: RespActEvent): void {
        this.onEvent?.(event);
    }

    private questionOf(inputs: Record<string, any>): string {
        return String(inputs.question ?? JSON.stringify(inputs));
    }

    private outputFormatInstruction(): string {
        if (typeof this.signature === 'string' || !this.signature) return '';
        // getOutputFieldConfigs, not getOutputFields: a zod signature has no
        // getOutputFields, and would otherwise lose its field list here.
        const fieldNames = Object.keys(getOutputFieldConfigs(this.signature));
        if (fieldNames.length === 0) return '';

        let instruction =
            '\n\nWhen providing your Final Answer, include all of the following fields, each on its own line:\n\n';
        for (const field of fieldNames) {
            instruction += `${field}: [your response for ${field}]\n`;
        }
        return instruction;
    }

    private buildInitialPrompt(inputs: Record<string, any>): string {
        const toolDescriptions = Object.entries(this.tools)
            .map(([name, tool]) => `- ${name}: ${tool.description}${textInputHint(tool)}`)
            .join('\n');

        return `You have access to the following tools:
${toolDescriptions}

Question: ${this.questionOf(inputs)}

Use the available tools to gather what you need before answering.

Take one action at a time. After each action you will receive an observation; do not plan several actions ahead and do not write the Observation line yourself.

Use this format:
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [input to the tool]

When you have everything you need, respond with:
Thought: [why you now have enough]
Final Answer: [complete answer to the original question]${this.outputFormatInstruction()}

Begin.`;
    }

    /**
     * System prompt for the native path.
     *
     * The tools and their schemas travel in the request rather than the prompt,
     * so this only has to cover what the tool protocol does not: how the final
     * answer should be shaped.
     */
    private buildNativePrompt(): string {
        return `Answer the user's question. You have tools available; call them as needed to gather what you need before answering.

When you have everything you need, reply with the final answer and no tool call.${this.outputFormatInstruction()}`;
    }

    private extractToolCall(response: string): { tool: string; input: string } | null {
        const actionMatch = response.match(/Action:\s*(.+?)(?=\n|$)/m);
        const inputMatch = response.match(/Action Input:\s*(.+?)(?=\n|$)/m);

        if (actionMatch && inputMatch) {
            return { tool: actionMatch[1].trim(), input: inputMatch[1].trim() };
        }
        return null;
    }

    /** Run a tool from the text path, where arguments arrive as one string. */
    private async executeTool(toolName: string, input: string, step: number): Promise<string> {
        const tool = this.tools[toolName];
        if (!tool) {
            return this.unknownToolMessage(toolName);
        }

        let args: Record<string, unknown>;
        try {
            args = tool.typed ? parseTextArguments(tool, input) : { input };
        } catch (error) {
            this.emit({ type: 'tool_error', step, tool: toolName, error });
            return `Error executing ${toolName}: ${error instanceof Error ? error.message : String(error)}`;
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
            let callArgs: unknown = args;
            if (tool.validator) {
                const validated = tool.validator.safeParse(args);
                if (!validated.success) {
                    throw new Error(
                        `Invalid arguments: ${validated.error.issues
                            .map(
                                (issue) =>
                                    `${issue.path.join('.') || '(root)'}: ${issue.message}`
                            )
                            .join('; ')}`
                    );
                }
                callArgs = validated.data;
            }

            if (!tool.typed && typeof args.input !== 'string') {
                // An untyped tool is declared as taking one string called
                // `input`. A model that sends something else would otherwise
                // reach the tool as `undefined`.
                throw new Error(
                    `Expected a single string argument named "input". Received: ${JSON.stringify(args)}`
                );
            }

            const result = await (tool.typed
                ? tool.function(callArgs)
                : tool.function(args.input));
            const output = typeof result === 'string' ? result : stringifyResult(result);
            this.emit({ type: 'tool_result', step, tool: toolName, output });
            return output;
        } catch (error) {
            this.emit({ type: 'tool_error', step, tool: toolName, error });
            return `Error executing ${toolName}: ${error instanceof Error ? error.message : String(error)}`;
        }
    }

    private unknownToolMessage(toolName: string): string {
        return `Error: Tool '${toolName}' not found. Available tools: ${Object.keys(this.tools).join(', ')}`;
    }

    private extractFinalAnswer(response: string): string {
        // Keep everything after the marker: multi-field answers span lines.
        const match = response.match(/Final Answer:\s*([\s\S]*)$/i);
        return match ? match[1].trim() : '';
    }
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
 * Turn a text-mode `Action Input:` line into named arguments.
 *
 * A typed tool wants an object, but the text loop only ever produces a line of
 * text. JSON is the documented form; a single-property schema also accepts the
 * bare value, which is what models tend to write for a one-argument tool. That
 * bare value is coerced to the declared scalar type, so a `number` argument does
 * not fail validation purely because the loop only speaks text.
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
    const properties = tool.parameters.properties;
    if (!properties || typeof properties !== 'object') return {};
    return properties as Record<string, unknown>;
}

/** Tell the text-mode model what an `Action Input:` for a typed tool looks like. */
function textInputHint(tool: NormalizedTool): string {
    if (!tool.typed) return '';
    const names = propertyNames(tool);
    if (names.length === 0) return '';
    return ` (Action Input must be a JSON object with keys: ${names.join(', ')})`;
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
