import {
    BaseLM,
    ContentFilterError,
    LMError,
    TimeoutError,
    classify,
    contentToText,
    imageToUrl,
    mergeRequestBody,
    providerOptionsFor,
    retryAfterMsFrom,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type ImageContentPart,
    type MessageContent,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
    type UsageStats,
} from '@ts-dspy/core';
import OpenAI, { APIConnectionTimeoutError, APIError, APIUserAbortError } from 'openai';
import type {
    ChatCompletionContentPart,
    ChatCompletionContentPartImage,
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletionCreateParamsStreaming,
    ChatCompletionMessageParam,
} from 'openai/resources/chat/completions';
import type { CompletionUsage } from 'openai/resources/completions';
import type { ReasoningEffort } from 'openai/resources/shared';

/**
 * Current default. Confirm against `client.models.list()` if you need a specific
 * tier — model identifiers move faster than release cycles.
 */
export const DEFAULT_OPENAI_MODEL = 'gpt-5.2';

/** How hard a reasoning model thinks before answering. Not every model takes every value. */
export type OpenAIReasoningEffort = Exclude<ReasoningEffort, null>;

/**
 * Per-call options read from `providerOptions.openai`.
 *
 * `reasoningEffort` is translated to `reasoning_effort`; every other key is
 * merged into the Chat Completions request body unchanged — e.g.
 * `{ openai: { service_tier: 'flex', user: 'u-1' } }`.
 */
export interface OpenAIProviderOptions {
    reasoningEffort?: OpenAIReasoningEffort;
    [key: string]: unknown;
}

export interface OpenAIConfig {
    apiKey?: string;
    model?: string;
    /** `null` stops the SDK reading `OPENAI_ORG_ID` from the environment. */
    organization?: string | null;
    /** `null` stops the SDK reading `OPENAI_PROJECT_ID` from the environment. */
    project?: string | null;
    /** Override for Azure, a proxy, or any OpenAI-compatible endpoint. */
    baseURL?: string;
    /** Default per-request timeout in milliseconds. */
    timeout?: number;
    /**
     * Default retry count (SDK default: 2). Retries run inside the SDK, which
     * honours `retry-after` headers. Overridden per call by `options.retries`.
     */
    maxRetries?: number;
    /**
     * Default `reasoning_effort` for reasoning models (`gpt-5*`, `o*`). Sent
     * only when set; overridden per call by `providerOptions.openai.reasoningEffort`.
     * Leave unset for models without reasoning, which reject the parameter.
     */
    reasoningEffort?: OpenAIReasoningEffort;
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

/**
 * Model families that take text only. Everything else in the current lineup
 * accepts images, so this is a deny-list: a new vision model works on the day it
 * ships, and the flag stops claiming vision for models that never had it.
 */
const TEXT_ONLY_PREFIXES = [
    'gpt-3.5',
    'gpt-4-0', // gpt-4-0314 / gpt-4-0613, before vision
    'gpt-4-32k',
    'o1-mini',
    'o1-preview',
    'o3-mini',
    'text-',
];

function supportsVisionFor(model: string): boolean {
    return !TEXT_ONLY_PREFIXES.some((prefix) => model.startsWith(prefix));
}

export class OpenAILM extends BaseLM {
    private readonly client: OpenAI;
    private readonly reasoningEffort?: OpenAIReasoningEffort;
    /** Constructor settings that change what a given prompt returns. */
    protected readonly scope: Record<string, unknown>;

    constructor(config: OpenAIConfig = {}) {
        super('openai', config.model ?? DEFAULT_OPENAI_MODEL);

        this.reasoningEffort = config.reasoningEffort;
        this.scope = {
            baseURL: config.baseURL ?? null,
            organization: config.organization ?? null,
            project: config.project ?? null,
            ...(config.reasoningEffort ? { reasoningEffort: config.reasoningEffort } : {}),
        };
        this.client = new OpenAI({
            apiKey: config.apiKey,
            organization: config.organization,
            project: config.project,
            baseURL: config.baseURL,
            timeout: config.timeout,
            maxRetries: config.maxRetries,
        });
    }

    /**
     * A proxy or an alternate organization can answer the same prompt
     * differently, so two clients configured that way must not share cache
     * entries.
     */
    protected cacheScope(): unknown {
        return this.scope;
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
                this.buildRequest(messages, options, toolParams(options)),
                requestOptions(options)
            );

            this.recordUsage({
                ...usageOf(completion.usage),
                latencyMs: Date.now() - startedAt,
                model: options?.model,
            });

            const choice = usableChoice(completion.choices);

            const toolCalls = fromOpenAIToolCalls(choice?.message?.tool_calls);
            const reasoning = reasoningOf(choice?.message);

            return {
                content: choice?.message?.content ?? '',
                ...(toolCalls.length > 0 ? { toolCalls } : {}),
                finishReason: finishReasonOf(choice?.finish_reason),
                ...(reasoning ? { reasoning } : {}),
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
                this.buildRequest([{ role: 'user', content: prompt }], options, {
                    response_format: {
                        type: 'json_schema',
                        json_schema: {
                            name: 'signature_output',
                            strict: true,
                            schema: schema as Record<string, unknown>,
                        },
                    },
                }),
                requestOptions(options)
            );

            this.recordUsage({
                ...usageOf(completion.usage),
                latencyMs: Date.now() - startedAt,
                model: options?.model,
            });

            const choice = usableChoice(completion.choices);

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

        let usage = usageOf(undefined);
        let finishReason: string | null | undefined;
        const pendingCalls: Array<{ id?: string; name: string; arguments: string }> = [];

        try {
            const stream = await this.client.chat.completions.create(
                {
                    ...this.buildRequest(messages, options, toolParams(options)),
                    stream: true,
                    stream_options: { include_usage: true },
                } as ChatCompletionCreateParamsStreaming,
                requestOptions(options)
            );

            // The loop is inside the try so a mid-stream abort surfaces as an
            // LMError and is counted, rather than escaping as a raw SDK error.
            for await (const chunk of stream) {
                if (chunk.usage) usage = usageOf(chunk.usage);
                const choice = chunk.choices[0];
                if (choice?.finish_reason === 'content_filter') {
                    assertNotFiltered(choice.finish_reason);
                }
                if (choice?.finish_reason) finishReason = choice.finish_reason;
                // Tool calls arrive in fragments keyed by index: the id and
                // name once, the argument JSON a few characters at a time.
                for (const delta of choice?.delta?.tool_calls ?? []) {
                    const call = (pendingCalls[delta.index] ??= { name: '', arguments: '' });
                    if (delta.id) call.id = delta.id;
                    if (delta.function?.name) call.name += delta.function.name;
                    if (delta.function?.arguments) call.arguments += delta.function.arguments;
                }
                const reasoning = reasoningOf(choice?.delta);
                if (reasoning) {
                    // Reasoning is not answer text, so it travels in metadata.
                    yield { content: '', done: false, metadata: { reasoningDelta: reasoning } };
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

        this.recordUsage({
            ...usage,
            latencyMs: Date.now() - startedAt,
            model: options?.model,
        });
        const toolCalls = fromOpenAIToolCalls(
            pendingCalls
                .filter((call) => call !== undefined)
                .map((call) => ({
                    id: call.id,
                    function: { name: call.name, arguments: call.arguments },
                }))
        );
        yield {
            content: '',
            done: true,
            usage: streamUsage(usage),
            // As the other providers do: the assembled calls and why the model
            // stopped, so a truncated stream is not mistaken for a complete one.
            metadata: {
                finishReason: finishReasonOf(finishReason),
                ...(toolCalls.length > 0 ? { toolCalls } : {}),
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
            supportsVision: supportsVisionFor(this.model),
            maxContextLength: contextLengthFor(this.model),
            supportedFormats: ['text', 'json_object', 'json_schema'],
        };
    }

    /** @deprecated Use {@link getModelName}. */
    getModel(): string {
        return this.model;
    }

    /**
     * Assemble a Chat Completions body: model, messages, sampling, reasoning
     * effort, the operation's own fields, then the raw `providerOptions.openai`
     * keys on top.
     */
    private buildRequest(
        messages: ChatMessage[],
        options: LLMCallOptions | undefined,
        extra: Record<string, unknown> = {}
    ): ChatCompletionCreateParamsNonStreaming {
        const { reasoningEffort, ...raw } = providerOptionsFor(
            options,
            'openai'
        ) as OpenAIProviderOptions;
        const effort = reasoningEffort ?? this.reasoningEffort;

        const body: Record<string, unknown> = {
            model: options?.model ?? this.model,
            messages: toOpenAIMessages(messages),
            ...samplingParams(options),
            ...(effort ? { reasoning_effort: effort } : {}),
            ...extra,
        };
        return mergeRequestBody(body, raw) as unknown as ChatCompletionCreateParamsNonStreaming;
    }
}

interface OpenAIUsage {
    promptTokens: number;
    completionTokens: number;
    cachedTokens?: number;
    reasoningTokens?: number;
}

/**
 * Token usage, including the cached-prompt and reasoning breakdowns. Both are
 * subsets of the prompt and completion totals respectively, exactly as the
 * API reports them.
 */
function usageOf(usage: CompletionUsage | null | undefined): OpenAIUsage {
    const cached = usage?.prompt_tokens_details?.cached_tokens;
    const reasoning = usage?.completion_tokens_details?.reasoning_tokens;
    return {
        promptTokens: usage?.prompt_tokens ?? 0,
        completionTokens: usage?.completion_tokens ?? 0,
        ...(cached !== undefined ? { cachedTokens: cached } : {}),
        ...(reasoning !== undefined ? { reasoningTokens: reasoning } : {}),
    };
}

function streamUsage(usage: OpenAIUsage): Partial<UsageStats> {
    return { ...usage, totalTokens: usage.promptTokens + usage.completionTokens };
}

/**
 * Reasoning text, where the server returns any. OpenAI itself does not on
 * Chat Completions; DeepSeek, vLLM and others send `reasoning_content`, and
 * OpenRouter sends `reasoning`.
 */
function reasoningOf(message: unknown): string | undefined {
    if (!message || typeof message !== 'object') return undefined;
    const { reasoning_content: content, reasoning } = message as {
        reasoning_content?: unknown;
        reasoning?: unknown;
    };
    if (typeof content === 'string' && content) return content;
    if (typeof reasoning === 'string' && reasoning) return reasoning;
    return undefined;
}

/**
 * Convert ts-dspy messages into the Chat Completions shape.
 *
 * `ChatCompletionMessageParam` is a discriminated union in which only the
 * `user` variant accepts image parts, so the conversion is role-aware: system
 * and assistant content is flattened to text (an image there would be rejected
 * by the API), and only user turns keep their parts.
 */
export function toOpenAIMessages(messages: ChatMessage[]): ChatCompletionMessageParam[] {
    return messages.map((message) => {
        switch (message.role) {
            case 'system':
                return { role: 'system', content: contentToText(message.content) };
            case 'assistant':
                if (message.toolCalls?.length) {
                    return {
                        role: 'assistant',
                        // The API rejects an empty string alongside tool_calls.
                        content: contentToText(message.content) || null,
                        tool_calls: message.toolCalls.map(toOpenAIToolCall),
                    };
                }
                return { role: 'assistant', content: contentToText(message.content) };
            case 'tool':
            case 'function':
                // `tool_call_id` is what pairs a result with its call. Without
                // one the API would reject the turn, so an uncorrelated result
                // is still surfaced as user content rather than dropped.
                if (message.toolCallId) {
                    return {
                        role: 'tool',
                        tool_call_id: message.toolCallId,
                        content: contentToText(message.content),
                    };
                }
                return { role: 'user', content: toOpenAIUserContent(message.content) };
            default:
                return { role: 'user', content: toOpenAIUserContent(message.content) };
        }
    });
}

/** User content: a plain string stays a string, parts become the SDK's part union. */
function toOpenAIUserContent(content: MessageContent): string | ChatCompletionContentPart[] {
    if (typeof content === 'string') return content;
    return content.map((part) =>
        part.type === 'text' ? { type: 'text' as const, text: part.text } : toOpenAIImage(part)
    );
}

/**
 * Images travel in `image_url.url`, which accepts an `https://` URL or a
 * `data:image/…;base64,…` URI — so a base64 source is rendered as a data URI.
 */
function toOpenAIImage(part: ImageContentPart): ChatCompletionContentPartImage {
    return {
        type: 'image_url',
        image_url: {
            url: imageToUrl(part.source),
            ...(part.detail ? { detail: part.detail } : {}),
        },
    };
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

/**
 * Throw when the reply carries no usable choice, or the model refused.
 *
 * A refusal arrives as `finish_reason: 'stop'` with `content: null` and the
 * explanation in `message.refusal`; read as an ordinary reply it is an empty
 * string, which a structured call then reports as invalid JSON and a repair
 * loop retries for nothing. An empty `choices` array is what Azure returns for
 * a prompt its filter blocked.
 */
function usableChoice<C extends { finish_reason?: string | null; message?: unknown }>(
    choices: readonly C[] | undefined
): C {
    const choice = choices?.[0];
    if (!choice) {
        throw new LMError('openai', 'The response contained no choices.');
    }
    assertNotFiltered(choice.finish_reason);
    const refusal = (choice.message as { refusal?: unknown } | undefined)?.refusal;
    if (typeof refusal === 'string' && refusal !== '') {
        throw new ContentFilterError('openai', `The model refused: ${refusal}`, {
            category: 'refusal',
        });
    }
    return choice;
}

/** Map an OpenAI SDK failure onto the core error hierarchy. */
export function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    // Checked before APIError: this is a subclass of it, and it carries no
    // status of its own to classify by.
    // A call cancelled through its signal is the other half of TimeoutError's
    // contract ("timed out or was aborted").
    if (error instanceof APIConnectionTimeoutError || error instanceof APIUserAbortError) {
        return new TimeoutError('openai', error.message, { cause: error });
    }
    if (error instanceof APIError) {
        // `code` is what separates a context-length 400 from any other 400.
        const ErrorClass = classify(error.status, {
            code: error.code,
            type: error.type,
            message: error.message,
        });
        return new ErrorClass('openai', error.message, {
            cause: error,
            status: error.status,
            retryAfterMs: retryAfterMsFrom(error.headers),
        });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('openai', message, { cause: error });
}
