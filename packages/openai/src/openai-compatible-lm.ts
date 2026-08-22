import {
    BaseLM,
    LMError,
    type ChatMessage,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
} from '@ts-dspy/core';
import { OpenAILM, type OpenAIConfig } from './openai-lm';

/**
 * Placeholder credential for servers that require an `Authorization` header but
 * ignore its value — Ollama, LM Studio, and most self-hosted vLLM deployments.
 *
 * The `openai` SDK throws at construction when no key is present, so a user
 * pointing at `localhost` with no `OPENAI_API_KEY` set gets an auth failure
 * before a single request leaves the process. Defaulting the key removes that
 * trap without pretending to be a real credential.
 */
export const PLACEHOLDER_API_KEY = 'not-needed';

/**
 * Conservative context window used when the caller does not declare one.
 *
 * Under-reporting is the safe direction: nothing in the library truncates on
 * this number, so a low value is merely pessimistic, whereas OpenAI's
 * `gpt-*`-prefix table would report 128k for a model that in fact holds 4k.
 */
export const DEFAULT_COMPATIBLE_CONTEXT_LENGTH = 8_192;

/** Base URLs for endpoints known to speak the OpenAI chat-completions API. */
export const OPENAI_COMPATIBLE_BASE_URLS = {
    ollama: 'http://localhost:11434/v1',
    lmstudio: 'http://localhost:1234/v1',
    vllm: 'http://localhost:8000/v1',
    groq: 'https://api.groq.com/openai/v1',
    together: 'https://api.together.xyz/v1',
    openrouter: 'https://openrouter.ai/api/v1',
} as const;

export interface OpenAICompatibleConfig extends Omit<OpenAIConfig, 'baseURL' | 'model'> {
    /** Required. The endpoint's OpenAI-compatible root, usually ending in `/v1`. */
    baseURL: string;
    /** Required. Endpoint model identifiers share nothing with OpenAI's. */
    model: string;
    /**
     * Defaults to {@link PLACEHOLDER_API_KEY} for servers that ignore the value.
     *
     * Unlike {@link OpenAILM}, `OPENAI_API_KEY` is deliberately *not* picked up
     * from the environment: `baseURL` here is by definition somewhere other than
     * `api.openai.com`, and quietly forwarding an OpenAI credential to whatever
     * host the caller named is a leak, not a convenience. Pass the endpoint's own
     * key — `process.env.GROQ_API_KEY` and friends — explicitly.
     */
    apiKey?: string;
    /**
     * Defaults to `true`: `stream: true` is part of the compatible surface.
     *
     * Set it to `false` for a server that rejects the streaming request anyway —
     * some older builds and strict proxies refuse the `stream_options` field the
     * OpenAI SDK path sends. Streaming then falls back to one plain call.
     */
    supportsStreaming?: boolean;
    /**
     * Defaults to `false`. Most compatible servers reject
     * `response_format: { type: 'json_schema', strict: true }` outright, so
     * claiming support makes every `Predict` call fail. Turn it on only for an
     * endpoint you have confirmed, such as vLLM's guided decoding.
     */
    supportsStructuredOutput?: boolean;
    /** Defaults to `false`. */
    supportsFunctionCalling?: boolean;
    /** Defaults to `false`. */
    supportsVision?: boolean;
    /** Defaults to {@link DEFAULT_COMPATIBLE_CONTEXT_LENGTH}. */
    maxContextLength?: number;
}

/**
 * Provider for any endpoint that speaks the OpenAI chat-completions API:
 * Ollama, LM Studio, vLLM, Groq, Together, OpenRouter, and the rest.
 *
 * The wire format is identical to {@link OpenAILM}, so this reuses it whole and
 * only corrects the four defaults that are wrong away from `api.openai.com`:
 * the key, the model, the advertised capabilities, and the context window.
 *
 * ```ts
 * const lm = new OpenAICompatibleLM({
 *     baseURL: 'http://localhost:11434/v1',
 *     model: 'llama3.2',
 * });
 * ```
 *
 * Capabilities default to the conservative answer because `Predict` branches on
 * `supportsStructuredOutput`: with it wrongly `true`, every call ships a strict
 * JSON-schema `response_format` that most servers reject, and the provider looks
 * broken rather than merely unsupported. With it `false`, structured output goes
 * through the prompt-based fallback in `BaseLM`, which works everywhere.
 */
export class OpenAICompatibleLM extends OpenAILM {
    private readonly capabilities: ModelCapabilities;

    constructor(config: OpenAICompatibleConfig) {
        if (!config?.baseURL) {
            throw new LMError(
                'openai',
                'OpenAICompatibleLM requires a baseURL, e.g. http://localhost:11434/v1'
            );
        }
        if (!config.model) {
            throw new LMError(
                'openai',
                'OpenAICompatibleLM requires a model; endpoint model ids differ from OpenAI ones.'
            );
        }

        super({
            ...config,
            apiKey: config.apiKey ?? PLACEHOLDER_API_KEY,
        });

        const structured = config.supportsStructuredOutput ?? false;
        this.capabilities = {
            supportsStreaming: config.supportsStreaming ?? true,
            supportsStructuredOutput: structured,
            supportsFunctionCalling: config.supportsFunctionCalling ?? false,
            supportsVision: config.supportsVision ?? false,
            maxContextLength: config.maxContextLength ?? DEFAULT_COMPATIBLE_CONTEXT_LENGTH,
            supportedFormats: structured ? ['text', 'json_object', 'json_schema'] : ['text'],
        };
    }

    getCapabilities(): ModelCapabilities {
        return {
            ...this.capabilities,
            supportedFormats: [...this.capabilities.supportedFormats],
        };
    }

    /**
     * Native JSON-schema mode when the endpoint has it, prompt-based JSON when
     * it does not. Calling this directly must not send a `response_format` the
     * server will reject, so the capability is honoured here too and not only
     * in `Predict`.
     */
    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        if (this.capabilities.supportsStructuredOutput) {
            return super.generateStructured<T>(prompt, schema, options);
        }
        // Reach past OpenAILM to the base class's prompt-based implementation
        // rather than restating its JSON extraction, which tolerates code fences.
        return BaseLM.prototype.generateStructured.call(
            this,
            prompt,
            schema,
            options
        ) as Promise<T>;
    }

    /**
     * Streams when the endpoint can, and otherwise emits the whole reply as a
     * single chunk so callers written against the streaming API keep working.
     */
    async *chatStream(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        if (this.capabilities.supportsStreaming) {
            yield* super.chatStream(messages, options);
            return;
        }

        const before = this.getUsage();
        const content = await this.chat(messages, options);
        const after = this.getUsage();

        if (content) {
            yield { content, done: false };
        }

        // Token counts come from diffing the instance-wide counters, which is
        // only sound when nothing else on this LM completed in between — one
        // shared instance serving concurrent calls is the documented pattern.
        // Report no usage rather than another call's tokens.
        if ((after.requestCount ?? 0) - (before.requestCount ?? 0) !== 1) {
            yield { content: '', done: true };
            return;
        }

        const promptTokens = after.promptTokens - before.promptTokens;
        const completionTokens = after.completionTokens - before.completionTokens;
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
}
