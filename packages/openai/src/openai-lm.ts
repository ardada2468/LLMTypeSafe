import {
    BaseLM,
    ContentFilterError,
    LMError,
    TimeoutError,
    classify,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
} from '@ts-dspy/core';
import OpenAI, { APIConnectionTimeoutError, APIError } from 'openai';
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
        return (await this.chatWithTools(messages, options)).content;
    }

    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
        const startedAt = Date.now();

        try {
            const completion = await this.client.chat.completions.create(
                {
                    model: options?.model ?? this.model,
                    messages: toOpenAIMessages(messages),
                    ...samplingParams(options),
                    ...toolParams(options),
                },
                requestOptions(options)
            );

            this.recordUsage({
                promptTokens: completion.usage?.prompt_tokens ?? 0,
                completionTokens: completion.usage?.completion_tokens ?? 0,
                latencyMs: Date.now() - startedAt,
            });

            const choice = completion.choices[0];
            assertNotFiltered(choice?.finish_reason);

            const toolCalls = fromOpenAIToolCalls(choice?.message?.tool_calls);

            return {
                content: choice?.message?.content ?? '',
                ...(toolCalls.length > 0 ? { toolCalls } : {}),
                finishReason: finishReasonOf(choice?.finish_reason),
            };
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
            assertNotFiltered(choice?.finish_reason);

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

        let promptTokens = 0;
        let completionTokens = 0;

        try {
            const stream = await this.client.chat.completions.create(
                {
                    model: options?.model ?? this.model,
                    messages: toOpenAIMessages(messages),
                    ...samplingParams(options),
                    stream: true,
                    stream_options: { include_usage: true },
                },
                requestOptions(options)
            );

            // The loop is inside the try so a mid-stream abort surfaces as an
            // LMError and is counted, rather than escaping as a raw SDK error.
            for await (const chunk of stream) {
                if (chunk.usage) {
                    promptTokens = chunk.usage.prompt_tokens ?? 0;
                    completionTokens = chunk.usage.completion_tokens ?? 0;
                }
                const choice = chunk.choices[0];
                if (choice?.finish_reason === 'content_filter') {
                    assertNotFiltered(choice.finish_reason);
                }
                const content = choice?.delta?.content;
                if (content) {
                    yield { content, done: false };
                }
            }
        } catch (error) {
            this.recordError();
            throw toLMError(error);
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
                if (message.toolCalls?.length) {
                    return {
                        role: 'assistant',
                        // The API rejects an empty string alongside tool_calls.
                        content: message.content || null,
                        tool_calls: message.toolCalls.map(toOpenAIToolCall),
                    };
                }
                return { role: 'assistant', content: message.content };
            case 'tool':
            case 'function':
                // `tool_call_id` is what pairs a result with its call. Without
                // one the API would reject the turn, so an uncorrelated result
                // is still surfaced as user content rather than dropped.
                if (message.toolCallId) {
                    return {
                        role: 'tool',
                        tool_call_id: message.toolCallId,
                        content: message.content,
                    };
                }
                return { role: 'user', content: message.content };
            default:
                return { role: 'user', content: message.content };
        }
    });
}

function toOpenAIToolCall(call: ToolCall, index: number) {
    return {
        // OpenAI requires an id; a call relayed from a provider without one
        // (Gemini) gets a placeholder. The position is part of it because two
        // parallel calls to the same tool would otherwise share an id, and
        // results would then pair up with the wrong call.
        id: call.id ?? `call_${index}_${call.name}`,
        type: 'function' as const,
        function: {
            name: call.name,
            // The wire format is a JSON *string*. Replay the provider's original
            // bytes when we have them, so a round trip is lossless.
            arguments: call.rawArguments ?? JSON.stringify(call.arguments ?? {}),
        },
    };
}

function fromOpenAIToolCalls(
    toolCalls:
        Array<{ id?: string; function?: { name?: string; arguments?: string } }> | undefined
): ToolCall[] {
    if (!toolCalls?.length) return [];

    return toolCalls
        .filter((call) => typeof call.function?.name === 'string')
        .map((call) => {
            const raw = call.function?.arguments ?? '';
            return {
                id: call.id,
                name: call.function!.name!,
                arguments: parseArguments(raw),
                rawArguments: raw,
            };
        });
}

/**
 * OpenAI sends arguments as a JSON string. A model can emit one that does not
 * parse; that is a tool-argument problem for the caller to report back to the
 * model, not a transport failure, so it yields empty arguments with the original
 * text preserved in `rawArguments`.
 */
function parseArguments(raw: string): Record<string, unknown> {
    if (!raw.trim()) return {};
    try {
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
            ? (parsed as Record<string, unknown>)
            : {};
    } catch {
        return {};
    }
}

/** Translate tool declarations into the Chat Completions request shape. */
function toolParams(options?: LLMCallOptions): Record<string, unknown> {
    if (!options?.tools?.length) return {};

    const params: Record<string, unknown> = {
        tools: options.tools.map((tool) => ({
            type: 'function',
            function: {
                name: tool.name,
                ...(tool.description ? { description: tool.description } : {}),
                parameters: tool.parameters,
            },
        })),
    };

    const choice = options.toolChoice;
    if (choice !== undefined) {
        params.tool_choice =
            typeof choice === 'string'
                ? choice
                : { type: 'function', function: { name: choice.name } };
    }

    return params;
}

function finishReasonOf(reason: string | null | undefined): FinishReason {
    switch (reason) {
        case 'stop':
            return 'stop';
        case 'tool_calls':
        case 'function_call':
            return 'tool_calls';
        case 'length':
            return 'length';
        case 'content_filter':
            return 'content_filter';
        default:
            return 'other';
    }
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
function requestOptions(options?: LLMCallOptions): {
    timeout?: number;
    maxRetries?: number;
    signal?: AbortSignal;
} {
    const request: { timeout?: number; maxRetries?: number; signal?: AbortSignal } = {};
    if (options?.timeout !== undefined) request.timeout = options.timeout;
    if (options?.retries !== undefined) request.maxRetries = options.retries;
    if (options?.signal !== undefined) request.signal = options.signal;
    return request;
}

/**
 * A filtered completion comes back as a normal 200 response whose only tell is
 * `finish_reason: 'content_filter'`. The previous implementation checked only
 * `'length'`, so a filtered reply was returned as an empty string in silence.
 */
function assertNotFiltered(finishReason: string | null | undefined): void {
    if (finishReason !== 'content_filter') return;
    throw new ContentFilterError('openai', 'Response was blocked by the content filter.', {
        category: 'content_filter',
    });
}

function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    // Checked before APIError: this is a subclass of it, and it carries no
    // status of its own to classify by.
    if (error instanceof APIConnectionTimeoutError) {
        return new TimeoutError('openai', error.message, { cause: error });
    }
    if (error instanceof APIError) {
        // `code` is what separates a context-length 400 from any other 400.
        const ErrorClass = classify(error.status, {
            code: error.code,
            type: error.type,
            message: error.message,
        });
        return new ErrorClass('openai', error.message, { cause: error, status: error.status });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('openai', message, { cause: error });
}
