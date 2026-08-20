import { BaseLM } from './core/base-lm';
import type { ChatMessage, LLMCallOptions, ModelCapabilities } from './types/language-model';

export interface MockLMOptions {
    /** Text replies returned by successive `generate`/`chat` calls. */
    responses?: string[];
    /** Objects returned by successive `generateStructured` calls. */
    structuredResponses?: unknown[];
    capabilities?: Partial<ModelCapabilities>;
}

/** Scripted language model for tests: replies in order, records every call. */
export class MockLM extends BaseLM {
    private responses: string[];
    private structuredResponses: unknown[];
    private capabilities: ModelCapabilities;

    /** Every chat call, in order, with the options it received. */
    readonly calls: Array<{ messages: ChatMessage[]; options?: LLMCallOptions }> = [];
    /** Every structured call, in order. */
    readonly structuredCalls: Array<{
        prompt: string;
        schema: unknown;
        options?: LLMCallOptions;
    }> = [];

    constructor(options: MockLMOptions = {}) {
        super('mock', 'mock-model');
        this.responses = [...(options.responses ?? [])];
        this.structuredResponses = [...(options.structuredResponses ?? [])];
        this.capabilities = {
            supportsStreaming: false,
            supportsStructuredOutput: false,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
            ...options.capabilities,
        };
    }

    setResponses(responses: string[]): void {
        this.responses = [...responses];
        this.calls.length = 0;
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        this.calls.push({ messages, options });
        if (this.responses.length === 0) {
            throw new Error('MockLM: no more scripted responses');
        }
        this.recordUsage({ promptTokens: 10, completionTokens: 5, latencyMs: 1 });
        return this.responses.shift()!;
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        this.structuredCalls.push({ prompt, schema, options });
        if (this.structuredResponses.length === 0) {
            // Fall back to the prompt-based path so tests can exercise either.
            return super.generateStructured<T>(prompt, schema, options);
        }
        this.recordUsage({ promptTokens: 10, completionTokens: 5, latencyMs: 1 });
        return this.structuredResponses.shift() as T;
    }

    getCapabilities(): ModelCapabilities {
        return this.capabilities;
    }

    /** The prompt text of the most recent chat call. */
    lastPrompt(): string {
        const last = this.calls.at(-1);
        return last?.messages.map((message) => message.content).join('\n') ?? '';
    }
}
