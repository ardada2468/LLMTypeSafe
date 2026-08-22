import {
    BaseLM,
    LMError,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
} from '@ts-dspy/core';
import Anthropic, { APIError } from '@anthropic-ai/sdk';
import type {
    ContentBlockParam,
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

            const final = await stream.finalMessage();
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
        throw new AnthropicRefusalError(
            details?.category ?? undefined,
            details?.explanation ?? undefined
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
 * system role inside `messages`. Consecutive same-role turns are merged, since
 * the API requires strict alternation.
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
            systemParts.push(message.content);
            continue;
        }

        const role: 'user' | 'assistant' = message.role === 'assistant' ? 'assistant' : 'user';
        const content = toAnthropicContent(message);
        const previous = converted.at(-1);

        if (previous?.role !== role) {
            converted.push({ role, content });
            continue;
        }

        if (typeof previous.content === 'string' && typeof content === 'string') {
            previous.content = `${previous.content}\n\n${content}`;
        } else {
            previous.content = [...asBlocks(previous.content), ...asBlocks(content)];
        }
    }

    return {
        system: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
        messages: converted,
    };
}

/** A plain turn stays a string; anything carrying tool traffic becomes blocks. */
function toAnthropicContent(message: ChatMessage): string | ContentBlockParam[] {
    if (message.role === 'assistant' && message.toolCalls?.length) {
        const blocks: ContentBlockParam[] = [];
        if (message.content) {
            blocks.push({ type: 'text', text: message.content });
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
        // Without a `tool_use_id` there is nothing to correlate against, so the
        // result degrades to ordinary user text rather than a rejected request.
        if (!message.toolCallId) return message.content;
        return [
            {
                type: 'tool_result',
                tool_use_id: message.toolCallId,
                content: message.content,
            },
        ];
    }

    return message.content;
}

function asBlocks(content: string | ContentBlockParam[]): ContentBlockParam[] {
    if (typeof content !== 'string') return content;
    return content ? [{ type: 'text', text: content }] : [];
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
