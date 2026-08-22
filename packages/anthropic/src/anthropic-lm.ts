import {
    BaseLM,
    ContentFilterError,
    LMError,
    TimeoutError,
    classify,
    type ChatMessage,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
} from '@ts-dspy/core';
import Anthropic, { APIConnectionTimeoutError, APIError } from '@anthropic-ai/sdk';
import type { Message, MessageParam } from '@anthropic-ai/sdk/resources/messages';

/** Current Claude Opus. Model IDs are exact — never append a date suffix. */
export const DEFAULT_ANTHROPIC_MODEL = 'claude-opus-5';

/**
 * `max_tokens` is required by the Messages API, so a default is unavoidable.
 * Non-streaming requests stay under the SDK's HTTP timeout at this size;
 * streaming requests can afford considerably more.
 */
export const DEFAULT_MAX_TOKENS = 16_000;
export const DEFAULT_STREAMING_MAX_TOKENS = 64_000;

export interface AnthropicConfig {
    apiKey?: string;
    model?: string;
    /** Override for a gateway or proxy. */
    baseURL?: string;
    /** Default per-request timeout in milliseconds. */
    timeout?: number;
    /** Default retry count. The SDK retries 408/409/429/5xx and honours `retry-after`. */
    maxRetries?: number;
    /** Default `max_tokens` for non-streaming requests. */
    maxTokens?: number;
}

/**
 * Raised when Claude's safety classifiers decline a request.
 *
 * @deprecated Renamed to `ContentFilterError` in `@ts-dspy/core`, which every
 * provider now throws for the same condition. This is an alias of that class,
 * not a subclass of it, so two things changed: the constructor now takes
 * `(provider, message, options)` rather than `(category, explanation)`, and an
 * `instanceof` check now also matches an OpenAI or Gemini content filter. Check
 * `error.provider === 'anthropic'` if you need to tell them apart. The alias
 * will be removed in a future release.
 */
export const AnthropicRefusalError = ContentFilterError;
/** @deprecated Renamed to `ContentFilterError` in `@ts-dspy/core`. */
export type AnthropicRefusalError = ContentFilterError;

export class AnthropicLM extends BaseLM {
    private readonly client: Anthropic;
    private readonly defaultMaxTokens: number;

    constructor(config: AnthropicConfig = {}) {
        super('anthropic', config.model ?? DEFAULT_ANTHROPIC_MODEL);

        this.client = new Anthropic({
            apiKey: config.apiKey,
            baseURL: config.baseURL,
            timeout: config.timeout,
            maxRetries: config.maxRetries,
        });
        this.defaultMaxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS;
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const { system, messages: converted } = toAnthropicMessages(messages);
        const startedAt = Date.now();

        let message: Message;
        try {
            message = await this.client.messages.create(
                {
                    model: options?.model ?? this.model,
                    max_tokens: options?.maxTokens ?? this.defaultMaxTokens,
                    messages: converted,
                    ...(system ? { system } : {}),
                    ...samplingParams(options),
                },
                requestOptions(options)
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.recordUsage({
            promptTokens: message.usage?.input_tokens ?? 0,
            completionTokens: message.usage?.output_tokens ?? 0,
            latencyMs: Date.now() - startedAt,
        });

        this.assertNotRefused(message);
        return textOf(message);
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const startedAt = Date.now();

        let message: Message;
        try {
            message = await this.client.messages.create(
                {
                    model: options?.model ?? this.model,
                    max_tokens: options?.maxTokens ?? this.defaultMaxTokens,
                    messages: [{ role: 'user', content: prompt }],
                    output_config: {
                        format: {
                            type: 'json_schema',
                            schema: schema as Record<string, unknown>,
                        },
                    },
                    ...samplingParams(options),
                },
                requestOptions(options)
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.recordUsage({
            promptTokens: message.usage?.input_tokens ?? 0,
            completionTokens: message.usage?.output_tokens ?? 0,
            latencyMs: Date.now() - startedAt,
        });

        this.assertNotRefused(message);

        if (message.stop_reason === 'max_tokens') {
            throw new LMError(
                'anthropic',
                'Structured response was truncated; raise maxTokens.'
            );
        }

        const content = textOf(message);
        try {
            return JSON.parse(content) as T;
        } catch (cause) {
            throw new LMError(
                'anthropic',
                `Structured response was not valid JSON: ${content}`,
                { cause }
            );
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
        const { system, messages: converted } = toAnthropicMessages(messages);
        const startedAt = Date.now();

        const stream = this.client.messages.stream(
            {
                model: options?.model ?? this.model,
                max_tokens: options?.maxTokens ?? DEFAULT_STREAMING_MAX_TOKENS,
                messages: converted,
                ...(system ? { system } : {}),
                ...samplingParams(options),
            },
            requestOptions(options)
        );

        // `assertNotRefused` records the error itself, so it must stay outside
        // this try — inside it, the catch counted the same refusal twice.
        let final: Message;
        try {
            for await (const event of stream) {
                if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
                    yield { content: event.delta.text, done: false };
                }
            }
            final = await stream.finalMessage();
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.recordUsage({
            promptTokens: final.usage?.input_tokens ?? 0,
            completionTokens: final.usage?.output_tokens ?? 0,
            latencyMs: Date.now() - startedAt,
        });
        this.assertNotRefused(final);

        yield {
            content: '',
            done: true,
            usage: {
                promptTokens: final.usage?.input_tokens ?? 0,
                completionTokens: final.usage?.output_tokens ?? 0,
                totalTokens:
                    (final.usage?.input_tokens ?? 0) + (final.usage?.output_tokens ?? 0),
            },
        };
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: true,
            supportsVision: true,
            maxContextLength: 1_000_000,
            supportedFormats: ['text', 'json_schema'],
        };
    }

    /**
     * A declined request comes back as a normal 200 response, so this must run
     * before any attempt to read `content`.
     */
    private assertNotRefused(message: Message): void {
        if (message.stop_reason !== 'refusal') return;

        this.recordError();
        const details = message.stop_details as
            { category?: string | null; explanation?: string | null } | null | undefined;
        const category = details?.category ?? undefined;
        const explanation = details?.explanation ?? undefined;

        throw new ContentFilterError(
            'anthropic',
            `Request was declined by safety classifiers${category ? ` (${category})` : ''}` +
                `${explanation ? `: ${explanation}` : ''}`,
            { category }
        );
    }
}

function textOf(message: Message): string {
    return message.content
        .filter(
            (block): block is Extract<typeof block, { type: 'text' }> => block.type === 'text'
        )
        .map((block) => block.text)
        .join('');
}

/**
 * Convert ts-dspy messages into the Messages API shape.
 *
 * System messages become the top-level `system` parameter — Anthropic has no
 * system role inside `messages`. Consecutive same-role turns are merged, since
 * the API requires strict alternation.
 */
export function toAnthropicMessages(messages: ChatMessage[]): {
    system?: string;
    messages: MessageParam[];
} {
    const systemParts: string[] = [];
    const converted: MessageParam[] = [];

    for (const message of messages) {
        if (message.role === 'system') {
            systemParts.push(message.content);
            continue;
        }

        const role: 'user' | 'assistant' = message.role === 'assistant' ? 'assistant' : 'user';
        const previous = converted.at(-1);

        if (previous?.role === role && typeof previous.content === 'string') {
            previous.content = `${previous.content}\n\n${message.content}`;
        } else {
            converted.push({ role, content: message.content });
        }
    }

    return {
        system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
        messages: converted,
    };
}

/**
 * Build sampling parameters.
 *
 * `temperature` and `top_p` are never sent together — Anthropic advises against
 * it — so temperature wins when both are supplied.
 */
function samplingParams(options?: LLMCallOptions): Record<string, unknown> {
    const params: Record<string, unknown> = {};

    if (options?.temperature !== undefined) {
        params.temperature = options.temperature;
    } else if (options?.topP !== undefined) {
        params.top_p = options.topP;
    }

    if (options?.stopSequences) params.stop_sequences = options.stopSequences;
    return params;
}

function requestOptions(options?: LLMCallOptions): { timeout?: number; maxRetries?: number } {
    const request: { timeout?: number; maxRetries?: number } = {};
    if (options?.timeout !== undefined) request.timeout = options.timeout;
    if (options?.retries !== undefined) request.maxRetries = options.retries;
    return request;
}

function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    // Checked before APIError: this is a subclass of it, and a client-side
    // timeout carries neither a status nor a `type`. The `timeout_error` type
    // only ever covers a server-side gateway timeout.
    if (error instanceof APIConnectionTimeoutError) {
        return new TimeoutError('anthropic', error.message, { cause: error });
    }
    if (error instanceof APIError) {
        // `error.type` is a typed union here — the cleanest discriminator of
        // the three SDKs. There is no context-length member, though: an
        // over-long prompt arrives as a 400 `invalid_request_error`.
        const ErrorClass = classify(error.status, {
            type: error.type,
            message: error.message,
        });
        return new ErrorClass('anthropic', error.message, {
            cause: error,
            status: error.status,
        });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('anthropic', message, { cause: error });
}
