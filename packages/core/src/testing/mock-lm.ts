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
    /**
     * Computes a reply from the request, for calls made once `responses` has
     * run out (or when none were scripted). Handy when a reply depends on the
     * prompt — echoing an input, say — or when a test should not care how
     * many calls a module makes.
     */
    respond?: (messages: ChatMessage[], options?: LLMCallOptions) => string;
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
    private respond?: (messages: ChatMessage[], options?: LLMCallOptions) => string;
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
        this.respond = options.respond;
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
        if (this.responses.length === 0 && !this.respond) {
            throw new Error('MockLM: no more scripted responses');
        }
        const reply =
            this.responses.length > 0
                ? this.responses.shift()!
                : this.respond!(messages, options);
        this.recordUsage({ promptTokens: 10, completionTokens: 5, latencyMs: 1 });
        return reply;
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
