import {
    BaseLM,
    LMError,
    contentToText,
    normalizeImageSource,
    type ChatMessage,
    type ImageContentPart,
    type LLMCallOptions,
    type MessageContent,
    type ModelCapabilities,
    type StreamChunk,
} from '@ts-dspy/core';
import Anthropic, { APIError } from '@anthropic-ai/sdk';
import type {
    Base64ImageSource,
    ContentBlockParam,
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlockParam,
} from '@anthropic-ai/sdk/resources/messages';

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
 * The API returns HTTP 200 with `stop_reason: "refusal"` and no usable content,
 * so this must be checked before reading the response body.
 */
export class AnthropicRefusalError extends LMError {
    readonly category?: string;

    constructor(category?: string, explanation?: string) {
        super(
            'anthropic',
            `Request was declined by safety classifiers${category ? ` (${category})` : ''}` +
                `${explanation ? `: ${explanation}` : ''}`
        );
        this.category = category;
    }
}

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

        try {
            for await (const event of stream) {
                if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
                    yield { content: event.delta.text, done: false };
                }
            }

            const final = await stream.finalMessage();
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
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: true,
            supportsVision: supportsVisionFor(this.model),
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
        throw new AnthropicRefusalError(
            details?.category ?? undefined,
            details?.explanation ?? undefined
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

/** Claude models that take text only; every other current model reads images. */
const TEXT_ONLY_MODELS = [/^claude-3-5-haiku/, /^claude-2/, /^claude-instant/];

function supportsVisionFor(model: string): boolean {
    return !TEXT_ONLY_MODELS.some((pattern) => pattern.test(model));
}

/**
 * Convert ts-dspy messages into the Messages API shape.
 *
 * System messages become the top-level `system` parameter — Anthropic has no
 * system role inside `messages`, and that parameter is text-only, so an image
 * addressed to it is flattened to its placeholder rather than silently dropped.
 * Consecutive same-role turns are merged, since the API requires strict
 * alternation.
 */
export function toAnthropicMessages(messages: ChatMessage[]): {
    system?: string;
    messages: MessageParam[];
} {
    const systemParts: string[] = [];
    const converted: MessageParam[] = [];

    for (const message of messages) {
        if (message.role === 'system') {
            systemParts.push(contentToText(message.content));
            continue;
        }

        const role: 'user' | 'assistant' = message.role === 'assistant' ? 'assistant' : 'user';
        const content = toAnthropicContent(message.content);
        const previous = converted.at(-1);

        // Merging happens on block arrays, not by string concatenation. The
        // previous implementation merged only when both sides were strings,
        // which meant a text turn followed by an image turn was pushed as two
        // adjacent user messages — and the API rejects anything but strict
        // alternation.
        if (previous?.role === role) {
            previous.content = mergeBlocks(toBlocks(previous.content), toBlocks(content));
        } else {
            converted.push({ role, content });
        }
    }

    return {
        system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
        messages: converted,
    };
}

function toAnthropicContent(
    content: MessageContent
): string | Array<TextBlockParam | ImageBlockParam> {
    if (typeof content === 'string') return content;
    return content.map((part) =>
        part.type === 'text'
            ? { type: 'text' as const, text: part.text }
            : toAnthropicImage(part)
    );
}

/**
 * Images are either inline base64 with an explicit media type, or a URL the API
 * fetches. A `data:` URI handed in as a URL is rewritten to the base64 form,
 * which is the only shape `URLImageSource` will not accept.
 */
function toAnthropicImage(part: ImageContentPart): ImageBlockParam {
    const source = normalizeImageSource(part.source);
    if (source.kind === 'url') {
        return { type: 'image', source: { type: 'url', url: source.url } };
    }
    return {
        type: 'image',
        source: {
            type: 'base64',
            // The SDK narrows media_type to the four types the API accepts;
            // ours stays open so a new one needs no core release.
            media_type: source.mediaType as Base64ImageSource['media_type'],
            data: source.data,
        },
    };
}

function toBlocks(content: MessageParam['content']): ContentBlockParam[] {
    return typeof content === 'string' ? [{ type: 'text', text: content }] : [...content];
}

/**
 * Concatenate two turns' blocks, folding a text block that meets another text
 * block into one. Without the fold the two turns would abut with no separator;
 * the blank line is what the string merge used to provide.
 */
function mergeBlocks(
    left: ContentBlockParam[],
    right: ContentBlockParam[]
): ContentBlockParam[] {
    const last = left.at(-1);
    const first = right[0];
    if (last?.type === 'text' && first?.type === 'text') {
        return [
            ...left.slice(0, -1),
            { ...last, text: `${last.text}\n\n${first.text}` },
            ...right.slice(1),
        ];
    }
    return [...left, ...right];
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
    if (error instanceof APIError) {
        return new LMError('anthropic', error.message, {
            cause: error,
            status: error.status,
        });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('anthropic', message, { cause: error });
}
