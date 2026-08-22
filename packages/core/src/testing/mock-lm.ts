import { BaseLM } from '../core/base-lm';
import { contentToText } from '../utils/content';
import type {
    ChatMessage,
    LLMCallOptions,
    ModelCapabilities,
    StreamChunk,
} from '../types/language-model';

export interface MockLMOptions {
    /** Text replies returned by successive `generate`/`chat` calls. */
    responses?: string[];
    /** Objects returned by successive `generateStructured` calls. */
    structuredResponses?: unknown[];
    capabilities?: Partial<ModelCapabilities>;
    /**
     * Characters per chunk emitted by `chatStream`/`generateStream`. Defaults to
     * word-sized pieces, which is roughly what a real provider sends.
     */
    chunkSize?: number;
}

/** Scripted language model for tests: replies in order, records every call. */
export class MockLM extends BaseLM {
    private responses: string[];
    private structuredResponses: unknown[];
    private capabilities: ModelCapabilities;
    private chunkSize?: number;

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
        this.chunkSize = options.chunkSize;
        this.capabilities = {
            supportsStreaming: true,
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
        // Copy: a caller that grows one `messages` array across a loop would
        // otherwise rewrite every call already recorded here.
        this.calls.push({ messages: messages.map((message) => ({ ...message })), options });
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

    /**
     * Stream the next scripted reply in pieces, ending with the same
     * `{ content: '', done: true, usage }` chunk the real providers send.
     */
    async *chatStream(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        const reply = await this.chat(messages, options);

        for (const piece of splitIntoChunks(reply, this.chunkSize)) {
            yield { content: piece, done: false };
        }

        yield {
            content: '',
            done: true,
            usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
        };
    }

    async *generateStream(
        prompt: string,
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        yield* this.chatStream([{ role: 'user', content: prompt }], options);
    }

    getCapabilities(): ModelCapabilities {
        return this.capabilities;
    }

    /** The prompt text of the most recent chat call. */
    lastPrompt(): string {
        const last = this.calls.at(-1);
        return last?.messages.map((message) => contentToText(message.content)).join('\n') ?? '';
    }
}

/** Split a reply into stream-sized pieces, preserving every character. */
function splitIntoChunks(text: string, chunkSize?: number): string[] {
    if (text === '') {
        return [];
    }

    if (chunkSize !== undefined && chunkSize > 0) {
        const chunks: string[] = [];
        for (let index = 0; index < text.length; index += chunkSize) {
            chunks.push(text.slice(index, index + chunkSize));
        }
        return chunks;
    }

    // Word-sized pieces, each carrying its trailing whitespace, so joining the
    // chunks reproduces the reply exactly.
    return text.match(/\S+\s*|\s+/g) ?? [text];
}
