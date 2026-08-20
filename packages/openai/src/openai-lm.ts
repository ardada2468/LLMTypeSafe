import {
    BaseLM,
    LMError,
    type ChatMessage,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
} from '@ts-dspy/core';
import OpenAI, { APIError } from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

/**
 * Current default. Confirm against `client.models.list()` if you need a specific
 * tier — model identifiers move faster than release cycles.
 */
export const DEFAULT_OPENAI_MODEL = 'gpt-5.2';

export interface OpenAIConfig {
    apiKey?: string;
    model?: string;
    organization?: string;
    project?: string;
    /** Override for Azure, a proxy, or any OpenAI-compatible endpoint. */
    baseURL?: string;
    /** Default per-request timeout in milliseconds. */
    timeout?: number;
    /** Default retry count. The SDK honours `retry-after` headers. */
    maxRetries?: number;
}

/** Context windows by model family, longest-prefix first. */
const CONTEXT_LENGTHS: Array<[prefix: string, length: number]> = [
    ['gpt-5', 400_000],
    ['o4', 200_000],
    ['o3', 200_000],
    ['o1', 200_000],
    ['gpt-4.1', 1_047_576],
    ['gpt-4o', 128_000],
    ['gpt-4-turbo', 128_000],
    ['gpt-4', 8_192],
    ['gpt-3.5', 16_385],
];

function contextLengthFor(model: string): number {
    for (const [prefix, length] of CONTEXT_LENGTHS) {
        if (model.startsWith(prefix)) return length;
    }
    return 128_000;
}

export class OpenAILM extends BaseLM {
    private readonly client: OpenAI;

    constructor(config: OpenAIConfig = {}) {
        super('openai', config.model ?? DEFAULT_OPENAI_MODEL);

        this.client = new OpenAI({
            apiKey: config.apiKey,
            organization: config.organization,
            project: config.project,
            baseURL: config.baseURL,
            timeout: config.timeout,
            maxRetries: config.maxRetries,
        });
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const startedAt = Date.now();

        try {
            const completion = await this.client.chat.completions.create(
                {
                    model: options?.model ?? this.model,
                    messages: toOpenAIMessages(messages),
                    ...samplingParams(options),
                },
                requestOptions(options)
            );

            this.recordUsage({
                promptTokens: completion.usage?.prompt_tokens ?? 0,
                completionTokens: completion.usage?.completion_tokens ?? 0,
                latencyMs: Date.now() - startedAt,
            });

            return completion.choices[0]?.message?.content ?? '';
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const startedAt = Date.now();

        try {
            const completion = await this.client.chat.completions.create(
                {
                    model: options?.model ?? this.model,
                    messages: toOpenAIMessages([{ role: 'user', content: prompt }]),
                    ...samplingParams(options),
                    response_format: {
                        type: 'json_schema',
                        json_schema: {
                            name: 'signature_output',
                            strict: true,
                            schema: schema as Record<string, unknown>,
                        },
                    },
                },
                requestOptions(options)
            );

            this.recordUsage({
                promptTokens: completion.usage?.prompt_tokens ?? 0,
                completionTokens: completion.usage?.completion_tokens ?? 0,
                latencyMs: Date.now() - startedAt,
            });

            const choice = completion.choices[0];
            if (choice?.finish_reason === 'length') {
                throw new LMError(
                    'openai',
                    'Structured response was truncated; raise maxTokens.'
                );
            }

            const content = choice?.message?.content ?? '';
            try {
                return JSON.parse(content) as T;
            } catch (cause) {
                throw new LMError(
                    'openai',
                    `Structured response was not valid JSON: ${content}`,
                    { cause }
                );
            }
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }
    }

    async *generateStream(
        prompt: string,
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        yield* this.chatStream([{ role: 'user', content: prompt }], options);
    }

    async *chatStream(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        const startedAt = Date.now();

        let stream;
        try {
            stream = await this.client.chat.completions.create(
                {
                    model: options?.model ?? this.model,
                    messages: toOpenAIMessages(messages),
                    ...samplingParams(options),
                    stream: true,
                    stream_options: { include_usage: true },
                },
                requestOptions(options)
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        let promptTokens = 0;
        let completionTokens = 0;

        for await (const chunk of stream) {
            if (chunk.usage) {
                promptTokens = chunk.usage.prompt_tokens ?? 0;
                completionTokens = chunk.usage.completion_tokens ?? 0;
            }
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
                yield { content, done: false };
            }
        }

        this.recordUsage({ promptTokens, completionTokens, latencyMs: Date.now() - startedAt });
        yield {
            content: '',
            done: true,
            usage: {
                promptTokens,
                completionTokens,
                totalTokens: promptTokens + completionTokens,
            },
        };
    }

    async listModels(): Promise<string[]> {
        const page = await this.client.models.list();
        return page.data.map((model) => model.id).sort();
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: true,
            supportsVision: true,
            maxContextLength: contextLengthFor(this.model),
            supportedFormats: ['text', 'json_object', 'json_schema'],
        };
    }

    /** @deprecated Use {@link getModelName}. */
    getModel(): string {
        return this.model;
    }
}

export function toOpenAIMessages(messages: ChatMessage[]): ChatCompletionMessageParam[] {
    return messages.map((message) => {
        switch (message.role) {
            case 'system':
                return { role: 'system', content: message.content };
            case 'assistant':
                return { role: 'assistant', content: message.content };
            case 'tool':
            case 'function':
                // The core ChatMessage shape has no tool_call_id, so a tool
                // result is surfaced as user content rather than dropped.
                return { role: 'user', content: message.content };
            default:
                return { role: 'user', content: message.content };
        }
    });
}

/**
 * Build sampling parameters.
 *
 * Only parameters the caller actually set are sent: reasoning models reject
 * non-default `temperature`/`top_p`, so defaulting them (the previous
 * implementation always sent `temperature: 0.7`) breaks those models outright.
 * `max_completion_tokens` replaces the deprecated `max_tokens`, which reasoning
 * models also reject.
 */
function samplingParams(options?: LLMCallOptions): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    if (options?.temperature !== undefined) params.temperature = options.temperature;
    if (options?.topP !== undefined) params.top_p = options.topP;
    if (options?.maxTokens !== undefined) params.max_completion_tokens = options.maxTokens;
    if (options?.frequencyPenalty !== undefined) {
        params.frequency_penalty = options.frequencyPenalty;
    }
    if (options?.presencePenalty !== undefined) {
        params.presence_penalty = options.presencePenalty;
    }
    if (options?.stopSequences) params.stop = options.stopSequences;
    return params;
}

/** Map ts-dspy call options onto the SDK's per-request options. */
function requestOptions(options?: LLMCallOptions): { timeout?: number; maxRetries?: number } {
    const request: { timeout?: number; maxRetries?: number } = {};
    if (options?.timeout !== undefined) request.timeout = options.timeout;
    if (options?.retries !== undefined) request.maxRetries = options.retries;
    return request;
}

function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    if (error instanceof APIError) {
        return new LMError('openai', error.message, { cause: error, status: error.status });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('openai', message, { cause: error });
}
