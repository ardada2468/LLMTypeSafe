import {
    BaseLM,
    ContentFilterError,
    LMError,
    TimeoutError,
    classify,
    retryAfterMsFrom,
    contentToText,
    mergeRequestBody,
    normalizeImageSource,
    providerOptionsFor,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type ImageContentPart,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
    type UsageStats,
} from '@ts-dspy/core';
import Anthropic, {
    APIConnectionTimeoutError,
    APIError,
    APIUserAbortError,
} from '@anthropic-ai/sdk';
import type {
    Base64ImageSource,
    ContentBlockParam,
    ImageBlockParam,
    Message,
    MessageCreateParamsNonStreaming,
    MessageParam,
    OutputConfig,
    TextBlockParam,
    ThinkingConfigParam,
    Usage,
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

/**
 * Extended thinking.
 *
 * With `budgetTokens` this sends `{ type: 'enabled', budget_tokens }`, the
 * fixed-budget form that only older models (before Claude 4.6) accept —
 * current models reject it with a 400. Without it, it sends
 * `{ type: 'adaptive' }`, which lets the model decide how much to think and is
 * what every current model expects; steer depth with `effort` instead.
 */
export interface AnthropicThinking {
    /** Fixed thinking budget, at least 1024. Older models only. */
    budgetTokens?: number;
    /**
     * Whether thinking text comes back. Current models default to `'omitted'`
     * (empty thinking text), so set `'summarized'` to receive `reasoning`.
     */
    display?: 'summarized' | 'omitted';
}

/** `output_config.effort`: how much the model spends on thinking and output. */
export type AnthropicEffort = NonNullable<OutputConfig['effort']>;

/**
 * Per-call options read from `providerOptions.anthropic`.
 *
 * `thinking`, `effort` and `cachePrompt` mean what they do on
 * {@link AnthropicConfig}; `thinking` also accepts the raw API shape (an object
 * with a `type`), sent unchanged. Every other key is merged into the Messages
 * request body as-is — e.g. `{ anthropic: { metadata: { user_id: 'u-1' } } }`.
 */
export interface AnthropicProviderOptions {
    thinking?: AnthropicThinking | ThinkingConfigParam | false;
    effort?: AnthropicEffort;
    cachePrompt?: boolean;
    [key: string]: unknown;
}

export interface AnthropicConfig {
    apiKey?: string;
    model?: string;
    /** Override for a gateway or proxy. */
    baseURL?: string;
    /** Default per-request timeout in milliseconds. */
    timeout?: number;
    /**
     * Default retry count (SDK default: 2). The SDK retries 408/409/429/5xx and
     * honours `retry-after`. Overridden per call by `options.retries`.
     */
    maxRetries?: number;
    /** Default `max_tokens` for non-streaming requests. */
    maxTokens?: number;
    /**
     * Default extended-thinking setting; `false` sends `{ type: 'disabled' }`.
     * Unset leaves the parameter out, so the model's own default applies.
     * While thinking is on, `temperature` and `top_p` are not sent.
     */
    thinking?: AnthropicThinking | false;
    /** Default `output_config.effort`. */
    effort?: AnthropicEffort;
    /**
     * Mark the system prompt — and, in a multi-turn conversation, the turn
     * before the last — with `cache_control: { type: 'ephemeral' }`, so a
     * repeated prefix (instructions, few-shot demos) is read from Anthropic's
     * prompt cache. Cache reads show up as `cachedTokens` in usage. Prefixes
     * below the model's minimum cacheable length are silently not cached.
     */
    cachePrompt?: boolean;
}

/**
 * Raised when Claude's safety classifiers decline a request.
 *
 * @deprecated Prefer `ContentFilterError` from `@ts-dspy/core`, which every
 * provider throws for the same condition. This is a **subclass** of it, so
 * `catch (e) { if (e instanceof ContentFilterError) … }` handles all three
 * providers while `instanceof AnthropicRefusalError` still means Anthropic
 * specifically. The constructor did change, though: it now takes
 * `(provider, message, options)` rather than `(category, explanation)`. This
 * subclass will be removed in a future release.
 */
export class AnthropicRefusalError extends ContentFilterError {}

export class AnthropicLM extends BaseLM {
    private readonly client: Anthropic;
    private readonly defaultMaxTokens: number;
    private readonly streamingMaxTokens: number;
    private readonly baseURL?: string;
    private readonly thinking?: AnthropicThinking | false;
    private readonly effort?: AnthropicEffort;
    private readonly cachePrompt: boolean;

    constructor(config: AnthropicConfig = {}) {
        super('anthropic', config.model ?? DEFAULT_ANTHROPIC_MODEL);

        this.client = new Anthropic({
            apiKey: config.apiKey,
            baseURL: config.baseURL,
            timeout: config.timeout,
            maxRetries: config.maxRetries,
        });
        this.defaultMaxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS;
        // An explicit cap is a cap: it applies to streaming too, where the
        // default is otherwise the roomier one.
        this.streamingMaxTokens = config.maxTokens ?? DEFAULT_STREAMING_MAX_TOKENS;
        this.baseURL = config.baseURL;
        this.thinking = config.thinking;
        this.effort = config.effort;
        this.cachePrompt = config.cachePrompt ?? false;
    }

    /**
     * `maxTokens` is part of the request, so a client that truncates at 64
     * tokens must not serve a cache entry recorded by one that allows 8192.
     */
    protected cacheScope(): unknown {
        return {
            maxTokens: this.defaultMaxTokens,
            baseURL: this.baseURL ?? null,
            // Spread only when set, so existing cache keys stay valid.
            ...(this.thinking !== undefined ? { thinking: this.thinking } : {}),
            ...(this.effort ? { effort: this.effort } : {}),
            ...(this.cachePrompt ? { cachePrompt: true } : {}),
        };
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        return (await this.chatWithTools(messages, options)).content;
    }

    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
        const startedAt = Date.now();

        let message: Message;
        try {
            message = await this.client.messages.create(
                this.buildRequest(messages, options, {
                    maxTokens: this.defaultMaxTokens,
                    extra: toolParams(options, String(options?.model ?? this.model)),
                }),
                requestOptions(options)
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.recordUsage({
            ...usageOf(message.usage),
            latencyMs: Date.now() - startedAt,
            model: options?.model,
        });

        this.assertNotRefused(message);

        const toolCalls = toolCallsOf(message);
        const reasoning = reasoningOf(message);
        return {
            content: textOf(message),
            ...(toolCalls.length > 0 ? { toolCalls } : {}),
            finishReason: finishReasonOf(message.stop_reason),
            ...(reasoning ? { reasoning } : {}),
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
                this.buildRequest([{ role: 'user', content: prompt }], options, {
                    maxTokens: this.defaultMaxTokens,
                    extra: {
                        output_config: {
                            format: {
                                type: 'json_schema',
                                schema: toAnthropicSchema(schema) as Record<string, unknown>,
                            },
                        },
                    },
                }),
                requestOptions(options)
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.recordUsage({
            ...usageOf(message.usage),
            latencyMs: Date.now() - startedAt,
            model: options?.model,
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
        const startedAt = Date.now();

        const stream = this.client.messages.stream(
            this.buildRequest(messages, options, {
                maxTokens: this.streamingMaxTokens,
                extra: toolParams(options, String(options?.model ?? this.model)),
            }),
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
                } else if (event.delta.type === 'thinking_delta') {
                    // Thinking is not answer text, so it travels in metadata.
                    yield {
                        content: '',
                        done: false,
                        metadata: { reasoningDelta: event.delta.thinking },
                    };
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

        const usage = usageOf(final.usage);
        this.recordUsage({
            ...usage,
            latencyMs: Date.now() - startedAt,
            model: options?.model,
        });
        this.assertNotRefused(final);

        const toolCalls = toolCallsOf(final);
        const reasoning = reasoningOf(final);
        const metadata = {
            finishReason: finishReasonOf(final.stop_reason),
            ...(toolCalls.length > 0 ? { toolCalls } : {}),
            ...(reasoning ? { reasoning } : {}),
        };
        yield {
            content: '',
            done: true,
            ...(Object.keys(metadata).length > 0 ? { metadata } : {}),
            usage: streamUsage(usage),
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
     * Assemble a Messages request: the converted conversation, `max_tokens`,
     * thinking, effort, prompt-cache markers and sampling, then the
     * operation's own fields, then the raw `providerOptions.anthropic` keys.
     */
    private buildRequest(
        messages: ChatMessage[],
        options: LLMCallOptions | undefined,
        request: { maxTokens: number; extra?: Record<string, unknown> }
    ): MessageCreateParamsNonStreaming {
        const {
            thinking: callThinking,
            effort: callEffort,
            cachePrompt: callCachePrompt,
            ...raw
        } = providerOptionsFor(options, 'anthropic') as AnthropicProviderOptions;

        const thinking = toThinkingParam(
            callThinking !== undefined ? callThinking : this.thinking
        );
        const effort = callEffort ?? this.effort;
        const cachePrompt = callCachePrompt ?? this.cachePrompt;

        const { system, messages: converted } = toAnthropicMessages(messages);

        let maxTokens = options?.maxTokens ?? request.maxTokens;
        // The API requires `budget_tokens < max_tokens`; leave the answer room.
        if (thinking?.type === 'enabled' && maxTokens <= thinking.budget_tokens) {
            maxTokens = thinking.budget_tokens + MIN_ANSWER_TOKENS;
        }

        const body: Record<string, unknown> = {
            model: options?.model ?? this.model,
            max_tokens: maxTokens,
            messages: cachePrompt ? markConversationPrefix(converted) : converted,
            ...(system ? { system: cachePrompt ? cachedSystem(system) : system } : {}),
            // Thinking requires the default sampling: the API rejects a
            // non-default temperature or top_p alongside it. Current models
            // reject the parameters outright, thinking or not.
            ...((thinking && thinking.type !== 'disabled') ||
            !acceptsSamplingParams(String(options?.model ?? this.model))
                ? {}
                : samplingParams(options)),
            ...(options?.stopSequences ? { stop_sequences: options.stopSequences } : {}),
            ...(thinking ? { thinking } : {}),
            ...request.extra,
        };
        const withEffort = effort
            ? mergeRequestBody(body, { output_config: { effort } })
            : body;
        return mergeRequestBody(withEffort, raw) as unknown as MessageCreateParamsNonStreaming;
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

        throw new AnthropicRefusalError(
            'anthropic',
            `Request was declined by safety classifiers${category ? ` (${category})` : ''}` +
                `${explanation ? `: ${explanation}` : ''}`,
            { category }
        );
    }
}

/** Room left for the answer when `max_tokens` is raised above a thinking budget. */
const MIN_ANSWER_TOKENS = 1024;

/** Translate the ts-dspy thinking option into the API's `thinking` parameter. */
function toThinkingParam(
    thinking: AnthropicThinking | ThinkingConfigParam | false | undefined
): ThinkingConfigParam | undefined {
    if (thinking === undefined) return undefined;
    if (thinking === false) return { type: 'disabled' };
    if ('type' in thinking) return thinking;
    const display = thinking.display ? { display: thinking.display } : {};
    if (thinking.budgetTokens !== undefined) {
        return { type: 'enabled', budget_tokens: thinking.budgetTokens, ...display };
    }
    return { type: 'adaptive', ...display };
}

/** The system prompt as one text block carrying a cache breakpoint. */
function cachedSystem(system: string): TextBlockParam[] {
    return [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }];
}

/**
 * Put a cache breakpoint on the last block of the turn before the final one,
 * so a conversation or a run of few-shot turns is cached up to — but not
 * including — the newest message, which differs on every call. A single-turn
 * request is left alone: caching it would only pay the cache-write premium.
 * The caller's array is not mutated.
 */
function markConversationPrefix(messages: MessageParam[]): MessageParam[] {
    if (messages.length < 2) return messages;
    const index = messages.length - 2;
    const blocks = toBlocks(messages[index].content);
    const last = blocks.at(-1);
    // Thinking blocks cannot carry a breakpoint; nothing to mark then.
    if (!last || last.type === 'thinking' || last.type === 'redacted_thinking') {
        return messages;
    }
    const marked = [...messages];
    marked[index] = {
        ...messages[index],
        content: [...blocks.slice(0, -1), { ...last, cache_control: { type: 'ephemeral' } }],
    };
    return marked;
}

interface AnthropicUsage {
    promptTokens: number;
    completionTokens: number;
    cachedTokens?: number;
    reasoningTokens?: number;
}

/**
 * Token usage. Anthropic's `input_tokens` excludes cache reads and writes, so
 * the prompt total adds them back — `cachedTokens` (cache reads) is then a
 * subset of `promptTokens`, as it is for every provider.
 */
function usageOf(usage: Usage | null | undefined): AnthropicUsage {
    const cacheRead = usage?.cache_read_input_tokens ?? undefined;
    const cacheWrite = usage?.cache_creation_input_tokens ?? 0;
    const thinking = usage?.output_tokens_details?.thinking_tokens;
    return {
        promptTokens: (usage?.input_tokens ?? 0) + cacheWrite + (cacheRead ?? 0),
        completionTokens: usage?.output_tokens ?? 0,
        ...(cacheRead !== undefined ? { cachedTokens: cacheRead } : {}),
        ...(thinking !== undefined ? { reasoningTokens: thinking } : {}),
    };
}

function streamUsage(usage: AnthropicUsage): Partial<UsageStats> {
    return { ...usage, totalTokens: usage.promptTokens + usage.completionTokens };
}

/** Text of the `thinking` blocks, which never counts as the answer. */
function reasoningOf(message: Message): string {
    return message.content
        .filter(
            (block): block is Extract<typeof block, { type: 'thinking' }> =>
                block.type === 'thinking'
        )
        .map((block) => block.thinking)
        .join('');
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

/**
 * Whether a model still takes `temperature` / `top_p`. Claude Opus 4.7 and
 * everything since answers them with a 400, so they are sent only to the
 * models known to accept them — an unrecognised (newer) id gets none, which
 * every model accepts. Modules such as `BestOfN` set a temperature themselves,
 * so sending it on would fail calls the user never configured.
 */
export function acceptsSamplingParams(model: string): boolean {
    if (/claude-(?:instant|2|3)/.test(model)) return true;
    const four = /claude-(?:opus|sonnet|haiku)-4(?:[-.@](\d+))?/.exec(model);
    if (!four) return false;
    const minor = four[1];
    // No minor is 4.0; eight digits are a snapshot date, also 4.0.
    return minor === undefined || minor.length >= 8 || Number(minor) <= 6;
}

/**
 * Whether a model accepts a forced tool choice (`any` or a named tool). The
 * listed models answer one with a 400; they get `auto` instead, which leaves
 * the decision to the model.
 */
export function acceptsForcedToolChoice(model: string): boolean {
    return !/claude-(?:opus-5-5|sonnet-5-5|(?:fable|mythos)-5-[12])/.test(model);
}

const UNSUPPORTED_SCHEMA_KEYWORDS = new Set([
    'minimum',
    'maximum',
    'exclusiveMinimum',
    'exclusiveMaximum',
    'multipleOf',
    'minLength',
    'maxLength',
    'maxItems',
    'uniqueItems',
]);

/**
 * Remove the JSON Schema constraints structured output does not support —
 * numeric bounds, string lengths, array limits beyond `minItems` of 0 or 1 —
 * which the API rejects with a 400. Zod emits them for something as ordinary
 * as `z.number().int()`. Nothing is lost: the reply is validated against the
 * full signature afterwards.
 */
export function toAnthropicSchema(schema: unknown, parentIsProperties = false): unknown {
    if (Array.isArray(schema)) return schema.map((item) => toAnthropicSchema(item));
    if (schema === null || typeof schema !== 'object') return schema;

    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(schema)) {
        // Inside `properties` the keys are field names, not keywords: a field
        // called `minimum` must survive.
        if (!parentIsProperties) {
            if (UNSUPPORTED_SCHEMA_KEYWORDS.has(key)) continue;
            if (key === 'minItems' && value !== 0 && value !== 1) continue;
        }
        const isPropertyMap =
            !parentIsProperties &&
            (key === 'properties' || key === '$defs' || key === 'definitions');
        out[key] = toAnthropicSchema(value, isPropertyMap);
    }
    return out;
}

/** Translate tool declarations into the Messages API request shape. */
function toolParams(options?: LLMCallOptions, model = ''): Record<string, unknown> {
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
        if (
            (typeof choice === 'object' || choice === 'required') &&
            !acceptsForcedToolChoice(model)
        ) {
            params.tool_choice = { type: 'auto' };
        } else if (typeof choice === 'object') {
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
    if (error instanceof APIConnectionTimeoutError || error instanceof APIUserAbortError) {
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
            retryAfterMs: retryAfterMsFrom(error.headers),
        });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('anthropic', message, { cause: error });
}
