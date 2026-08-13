import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { type Signature } from '../core/signature';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import type { SignatureOutput } from '../types/signature';
import { parseOutput as utilParseOutput } from '../utils/parsing';
import { ValidationError } from '../core/errors';

export interface ToolFunction {
    (...args: any[]): Promise<any> | any;
}

export interface ToolWithDescription {
    description: string;
    function: ToolFunction;
}

export type ToolDefinition = ToolFunction | ToolWithDescription;

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
}

export class RespAct<TSignature extends typeof Signature = typeof Signature> extends Module {
    private tools: Record<string, ToolWithDescription>;
    private maxSteps: number;
    private onEvent?: (event: RespActEvent) => void;

    constructor(signature: string | TSignature, options: RespActOptions) {
        super(signature, options.lm);

        this.tools = {};
        for (const [name, tool] of Object.entries(options.tools)) {
            this.tools[name] =
                typeof tool === 'function'
                    ? { description: `Tool: ${name}`, function: tool }
                    : tool;
        }
        this.maxSteps = options.maxSteps ?? 6;
        this.onEvent = options.onEvent;
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<
        Prediction<SignatureOutput<TSignature> & { steps: number }> &
            SignatureOutput<TSignature> & { steps: number }
    > {
        let conversation = this.buildInitialPrompt(inputs);
        const previousToolCalls = new Set<string>();

        for (let step = 0; step < this.maxSteps; step++) {
            const response = await this.lm.generate(conversation + '\n\nThought:', options);
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
                    const fieldList = error.issues.map((issue) => issue.field).join(', ');
                    conversation += `\n\nObservation: Your Final Answer was missing or malformed for: ${fieldList}. Provide a Final Answer with every required field on its own "field: value" line.`;
                    continue;
                }
                throw error;
            }

            const combinedOutput = { ...parsed, steps: step + 1 };
            return new Prediction(combinedOutput) as Prediction<
                SignatureOutput<TSignature> & { steps: number }
            > &
                SignatureOutput<TSignature> & { steps: number };
        }

        throw new Error(
            `RespAct exceeded maximum steps (${this.maxSteps}) without producing a valid final answer`
        );
    }

    protected parseOutput(rawOutput: unknown): Record<string, any> {
        if (!this.signature) {
            throw new Error('No signature provided for RespAct parsing');
        }
        const outputText =
            typeof rawOutput === 'string' ? rawOutput : JSON.stringify(rawOutput);
        return utilParseOutput(this.signature, outputText);
    }

    private emit(event: RespActEvent): void {
        this.onEvent?.(event);
    }

    private buildInitialPrompt(inputs: Record<string, any>): string {
        const toolDescriptions = Object.entries(this.tools)
            .map(([name, tool]) => `- ${name}: ${tool.description}`)
            .join('\n');

        let outputFormatInstruction = '';
        if (typeof this.signature !== 'string' && this.signature) {
            const fieldNames = Object.keys(this.signature.getOutputFields());
            if (fieldNames.length > 0) {
                outputFormatInstruction =
                    '\n\nWhen providing your Final Answer, include all of the following fields, each on its own line:\n\n';
                for (const field of fieldNames) {
                    outputFormatInstruction += `${field}: [your response for ${field}]\n`;
                }
            }
        }

        return `You have access to the following tools:
${toolDescriptions}

Question: ${inputs.question ?? JSON.stringify(inputs)}

Use the available tools to gather what you need before answering.

Take one action at a time. After each action you will receive an observation; do not plan several actions ahead and do not write the Observation line yourself.

Use this format:
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [input to the tool]

When you have everything you need, respond with:
Thought: [why you now have enough]
Final Answer: [complete answer to the original question]${outputFormatInstruction}

Begin.`;
    }

    private extractToolCall(response: string): { tool: string; input: string } | null {
        const actionMatch = response.match(/Action:\s*(.+?)(?=\n|$)/m);
        const inputMatch = response.match(/Action Input:\s*(.+?)(?=\n|$)/m);

        if (actionMatch && inputMatch) {
            return { tool: actionMatch[1].trim(), input: inputMatch[1].trim() };
        }
        return null;
    }

    private async executeTool(toolName: string, input: string, step: number): Promise<string> {
        if (!(toolName in this.tools)) {
            return `Error: Tool '${toolName}' not found. Available tools: ${Object.keys(this.tools).join(', ')}`;
        }

        try {
            const result = await this.tools[toolName].function(input);
            const output = String(result);
            this.emit({ type: 'tool_result', step, tool: toolName, output });
            return output;
        } catch (error) {
            this.emit({ type: 'tool_error', step, tool: toolName, error });
            return `Error executing ${toolName}: ${error instanceof Error ? error.message : String(error)}`;
        }
    }

    private extractFinalAnswer(response: string): string {
        // Keep everything after the marker: multi-field answers span lines.
        const match = response.match(/Final Answer:\s*([\s\S]*)$/i);
        return match ? match[1].trim() : '';
    }
}
