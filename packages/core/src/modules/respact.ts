import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { Signature } from '../core/signature';
import { SignatureOutput } from '../types/signature';

export interface ToolFunction {
    (...args: any[]): Promise<any> | any;
}

export interface ToolWithDescription {
    description: string;
    function: ToolFunction;
}

export type ToolDefinition = ToolFunction | ToolWithDescription;

export class RespAct<TSignature extends typeof Signature = typeof Signature> extends Module {
    private tools: Record<string, ToolWithDescription>;
    private maxSteps: number;

    constructor(
        signature: string | TSignature,
        options: {
            tools: Record<string, ToolDefinition>;
            maxSteps?: number;
        }
    ) {
        super(signature);
        // Normalize tools to always have descriptions
        this.tools = {};
        for (const [name, tool] of Object.entries(options.tools)) {
            if (typeof tool === 'function') {
                // Legacy support: function without description
                this.tools[name] = {
                    description: `Tool: ${name}`,
                    function: tool
                };
            } else {
                // New format: tool with description
                this.tools[name] = tool;
            }
        }
        this.maxSteps = options.maxSteps || 6;
    }

    async forward(
        inputs: Record<string, any>
    ): Promise<Prediction<SignatureOutput<TSignature> & { steps: number }> & SignatureOutput<TSignature> & { steps: number }> {
        let conversation = this.buildInitialPrompt(inputs);
        let finalAnswerData: SignatureOutput<TSignature> = {} as SignatureOutput<TSignature>;
        let previousToolCalls: string[] = []; // Track previous tool calls to avoid loops

        for (let step = 0; step < this.maxSteps; step++) {
            const response = await this.lm.generate(conversation + '\n\nThought:');
            conversation += `\n\nThought: ${response}`;

            // Check for tool usage FIRST (before checking for final answer)
            const toolCall = this.extractToolCall(response);
            if (toolCall) {
                // Check for repeated tool calls to prevent loops
                const toolCallKey = `${toolCall.tool}:${toolCall.input}`;
                if (previousToolCalls.includes(toolCallKey)) {
                    console.warn('🔄 WARNING: Detected repeated tool call, encouraging final answer...');
                    conversation += `\n\nObservation: You have already made this tool call. Please move to the next step.`;
                    continue;
                }
                previousToolCalls.push(toolCallKey);

                try {
                    const observation = await this.executeTool(toolCall.tool, toolCall.input);
                    conversation += `\n\nObservation: ${observation}`;
                    continue; // Continue to next step after tool execution
                } catch (error) {
                    conversation += `\n\nObservation: Error - ${error}`;
                    continue; // Continue even after error
                }
            }

            // Check for final answer only if no tool call was found
            if (response.toLowerCase().includes('final answer:')) {
                const rawAnswer = this.extractFinalAnswer(response);
                let parsedFromSignature: SignatureOutput<TSignature> = {} as SignatureOutput<TSignature>;

                try {
                    parsedFromSignature = this.parseOutput(rawAnswer) as SignatureOutput<TSignature>;
                } catch (error) {
                    console.warn('Failed to parse output from signature, using raw answer:', error);
                    parsedFromSignature = {} as SignatureOutput<TSignature>;
                }

                // For structured signatures, check if we have all required fields
                if (typeof this.signature !== 'string' && this.signature) {
                    const outputFields = this.signature.getOutputFields();
                    const requiredFields = Object.keys(outputFields);
                    const providedFields = Object.keys(parsedFromSignature).filter(key =>
                        parsedFromSignature[key] !== null && parsedFromSignature[key] !== undefined
                    );

                    // Temporarily relaxed validation - accept if we have at least some structured data
                    if (providedFields.length === 0) {
                        conversation += `\n\nObservation: Your Final Answer needs to include structured fields: ${requiredFields.join(', ')}. Please provide a Final Answer with the required format.`;
                        continue;
                    }
                }

                // Ensure we have meaningful output
                const parsedKeys = Object.keys(parsedFromSignature);
                const hasValidParsedData = parsedKeys.some(key => parsedFromSignature[key] !== null && parsedFromSignature[key] !== undefined);

                if (!hasValidParsedData || parsedKeys.length === 0) {
                    // If parsing failed or returned empty, create answer from raw
                    finalAnswerData = { answer: rawAnswer } as unknown as SignatureOutput<TSignature>;
                } else {
                    // If we have parsed data but missing answer field, add it
                    if (!('answer' in parsedFromSignature) && rawAnswer !== null) {
                        finalAnswerData = { ...parsedFromSignature, answer: rawAnswer } as unknown as SignatureOutput<TSignature>;
                    } else {
                        finalAnswerData = parsedFromSignature;
                    }
                }

                const combinedOutput = { ...finalAnswerData, steps: step + 1 };
                return new Prediction(combinedOutput) as Prediction<SignatureOutput<TSignature> & { steps: number }> & SignatureOutput<TSignature> & { steps: number };
            }
        }

        throw new Error(`RespAct exceeded maximum steps (${this.maxSteps}) without finding answer`);
    }

    protected parseOutput(rawOutput: any): Record<string, any> {
        if (!this.signature) {
            throw new Error('No signature provided for RespAct parsing');
        }

        // Ensure rawOutput is a string for parsing
        let outputText: string;
        if (typeof rawOutput === 'string') {
            outputText = rawOutput;
        } else {
            // Convert non-string values to string for parsing
            outputText = JSON.stringify(rawOutput);
        }

        const { parseOutput: utilParseOutput } = require('../utils/parsing');
        return utilParseOutput(this.signature, outputText);
    }

    private buildInitialPrompt(inputs: Record<string, any>): string {
        // Build tool descriptions
        const toolDescriptions = Object.entries(this.tools)
            .map(([name, tool]) => `- ${name}: ${tool.description}`)
            .join('\n');

        // Check if we have a structured signature
        let outputFormatInstruction = '';
        if (typeof this.signature !== 'string' && this.signature) {
            const outputFields = this.signature.getOutputFields();
            const fieldNames = Object.keys(outputFields);
            if (fieldNames.length > 0) {
                outputFormatInstruction = `\n\nIMPORTANT: When providing your Final Answer, you MUST provide ALL the following fields in this EXACT format:\n\n`;
                fieldNames.forEach(field => {
                    outputFormatInstruction += `${field}: [your response for ${field}]\n`;
                });
            }
        }

        return `You have access to the following tools:
${toolDescriptions}

Question: ${inputs.question || JSON.stringify(inputs)}

IMPORTANT: You MUST use the available tools to solve this question. Do not attempt to answer directly without using tools when tools are available for the task.

CRITICAL: Take ONE action at a time. After each action, you will receive an observation. Do NOT plan multiple actions in advance.

Work systematically:
1. Gather all necessary data using tools
2. Perform calculations if needed 
3. When you have ALL the information needed to answer the question completely, provide your Final Answer

Use this EXACT format (DO NOT generate the Observation line - it will be provided automatically):
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [input to the tool]

After you receive the Observation, you can then decide your next action. 

When you have gathered ALL necessary information through tool usage, provide:
Thought: [reasoning that you now have everything needed]
Final Answer: [complete answer to the original question using all gathered information]${outputFormatInstruction}

Begin! Remember: ONE action at a time, then decide if you need more information or can provide the final answer.`;
    }

    private extractToolCall(response: string): { tool: string; input: string } | null {
        // Look for Action and Action Input patterns, handling multiline responses
        const actionMatch = response.match(/Action:\s*(.+?)(?=\n|$)/m);
        const inputMatch = response.match(/Action Input:\s*(.+?)(?=\n|$)/m);

        if (actionMatch && inputMatch) {
            return {
                tool: actionMatch[1].trim(),
                input: inputMatch[1].trim()
            };
        }

        return null;
    }

    private async executeTool(toolName: string, input: string): Promise<string> {
        if (!(toolName in this.tools)) {
            return `Error: Tool '${toolName}' not found. Available tools: ${Object.keys(this.tools).join(', ')}`;
        }

        try {
            const result = await this.tools[toolName].function(input);
            return String(result);
        } catch (error) {
            return `Error executing ${toolName}: ${error}`;
        }
    }

    private extractFinalAnswer(response: string): any {
        const match = response.match(/Final Answer:\s*(.+?)$/m);
        if (match) {
            const answer = match[1].trim();

            // Try to parse as number
            const num = Number(answer);
            if (!isNaN(num)) return num;

            // Try to parse as JSON
            try {
                return JSON.parse(answer);
            } catch {
                return answer;
            }
        }

        return null;
    }
} 