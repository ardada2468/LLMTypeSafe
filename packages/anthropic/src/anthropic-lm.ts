import {
    BaseLM,
    ContentFilterError,
    LMError,
    TimeoutError,
    classify,
    contentToText,
    normalizeImageSource,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type ImageContentPart,
    type LLMCallOptions,
    type MessageContent,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
} from '@ts-dspy/core';
import Anthropic, { APIConnectionTimeoutError, APIError } from '@anthropic-ai/sdk';
import type {
    Base64ImageSource,
    ContentBlockParam,
    ImageBlockParam,
    Message,
    MessageParam,
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
        return (await this.chatWithTools(messages, options)).content;
    }

    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
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
                    ...toolParams(options),
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

        const toolCalls = toolCallsOf(message);
        return {
            content: textOf(message),
            ...(toolCalls.length > 0 ? { toolCalls } : {}),
            finishReason: finishReasonOf(message.stop_reason),
        };
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
                ...toolParams(options),
            },
            requestOptions(options)
        );

        // `assertNotRefused` records the error itself, so it must stay outside
        // this try — inside it, the catch counted the same refusal twice.
        let final: Message;
        try {
            for await (const event of stream) {
                if (event.type !== 'content_block_delta') continue;

                if (event.delta.type === 'text_delta') {
                    yield { content: event.delta.text, done: false };
                } else if (event.delta.type === 'input_json_delta') {
                    // Tool arguments stream as JSON fragments on their own event
                    // type. They are not text, so they travel in metadata rather
                    // than being spliced into `content`; the assembled calls also
                    // arrive whole on the final chunk.
                    yield {
                        content: '',
                        done: false,
                        metadata: {
                            toolInputDelta: {
                                index: event.index,
                                partialJson: event.delta.partial_json,
                            },
                        },
                    };
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

        const toolCalls = toolCallsOf(final);
        yield {
            content: '',
            done: true,
            ...(toolCalls.length > 0 ? { metadata: { toolCalls } } : {}),
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

/**
 * The assistant's prose.
 *
 * This deliberately keeps only `text` blocks: `tool_use` blocks are not text and
 * are surfaced separately by {@link toolCallsOf}, so a tool-calling turn returns
 * whatever the model said alongside the call rather than a JSON blob.
 */
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

/** Extract the `tool_use` blocks a turn requested. Anthropic sends `input` already parsed. */
function toolCallsOf(message: Message): ToolCall[] {
    return message.content
        .filter(
            (block): block is Extract<typeof block, { type: 'tool_use' }> =>
                block.type === 'tool_use'
        )
        .map((block) => ({
            id: block.id,
            name: block.name,
            arguments:
                block.input && typeof block.input === 'object' && !Array.isArray(block.input)
                    ? (block.input as Record<string, unknown>)
                    : {},
        }));
}

/**
 * Convert ts-dspy messages into the Messages API shape.
 *
 * System messages become the top-level `system` parameter — Anthropic has no
 * system role inside `messages`, and that parameter is text-only, so an image
 * addressed to it is flattened to its placeholder rather than silently dropped.
 * Consecutive same-role turns are merged, since the API requires strict
 * alternation.
 *
 * Tool traffic is structural rather than textual: an assistant turn carrying
 * tool calls becomes `text` + `tool_use` blocks, and a `tool` result turn becomes
 * a user turn holding a `tool_result` block keyed by `tool_use_id`. Merging
 * therefore happens at the block level whenever either side is block-shaped —
 * concatenating a tool result onto a plain user string, as the previous
 * implementation did, would have destroyed the correlation the API needs.
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
        const content = toAnthropicContent(message);
        const previous = converted.at(-1);

        if (previous?.role !== role) {
            converted.push({ role, content });
            continue;
        }

        // Two plain strings still merge as a string. Anything block-shaped — an
        // image, a tool call, a tool result — merges at the block level instead:
        // concatenating those as text would destroy the structure the API needs,
        // and refusing to merge would break its strict role alternation.
        if (typeof previous.content === 'string' && typeof content === 'string') {
            previous.content = `${previous.content}\n\n${content}`;
        } else {
            previous.content = mergeBlocks(toBlocks(previous.content), toBlocks(content));
        }
    }

    return {
        system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
        messages: converted,
    };
}

/**
 * A plain turn stays a string; anything carrying tool traffic or an image
 * becomes blocks.
 */
function toAnthropicContent(message: ChatMessage): string | ContentBlockParam[] {
    if (message.role === 'assistant' && message.toolCalls?.length) {
        const blocks: ContentBlockParam[] = [];
        const text = contentToText(message.content);
        if (text) {
            blocks.push({ type: 'text', text });
        }
        message.toolCalls.forEach((call, index) => {
            blocks.push({
                type: 'tool_use',
                // Anthropic requires an id; a call relayed from a provider
                // without one (Gemini) gets a placeholder. The position is part
                // of it because two parallel calls to the same tool would
                // otherwise share an id, mispairing their results.
                id: call.id ?? `toolu_${index}_${call.name}`,
                name: call.name,
                input: call.arguments ?? {},
            });
        });
        return blocks;
    }

    if (message.role === 'tool' || message.role === 'function') {
        const text = contentToText(message.content);
        // Without a `tool_use_id` there is nothing to correlate against, so the
        // result degrades to ordinary user text rather than a rejected request.
        if (!message.toolCallId) return text;
        return [
            {
                type: 'tool_result',
                tool_use_id: message.toolCallId,
                content: text,
            },
        ];
    }

    // An assistant turn cannot carry an image, so only user content keeps parts.
    if (typeof message.content === 'string') return message.content;
    if (message.role === 'assistant') return contentToText(message.content);
    return message.content.map((part) =>
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

/** Translate tool declarations into the Messages API request shape. */
function toolParams(options?: LLMCallOptions): Record<string, unknown> {
    if (!options?.tools?.length) return {};

    const params: Record<string, unknown> = {
        tools: options.tools.map((tool) => ({
            name: tool.name,
            ...(tool.description ? { description: tool.description } : {}),
            input_schema: tool.parameters,
        })),
    };

    const choice = options.toolChoice;
    if (choice !== undefined) {
        if (typeof choice === 'object') {
            params.tool_choice = { type: 'tool', name: choice.name };
        } else if (choice === 'required') {
            // Anthropic spells "you must call some tool" as `any`.
            params.tool_choice = { type: 'any' };
        } else {
            params.tool_choice = { type: choice };
        }
    }

    return params;
}

function finishReasonOf(reason: Message['stop_reason']): FinishReason {
    switch (reason) {
        case 'end_turn':
        case 'stop_sequence':
            return 'stop';
        case 'tool_use':
            return 'tool_calls';
        case 'max_tokens':
            return 'length';
        case 'refusal':
            return 'content_filter';
        default:
            return 'other';
    }
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
