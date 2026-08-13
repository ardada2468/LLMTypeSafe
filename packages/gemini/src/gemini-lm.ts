import {
    BaseLM,
    LMError,
    type ChatMessage,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
} from '@ts-dspy/core';
import {
    GoogleGenAI,
    HarmBlockThreshold,
    HarmCategory,
    type Content,
    type GenerateContentConfig,
    type GenerateContentResponse,
    type SafetySetting,
} from '@google/genai';

/** Current default. Gemini 2.x models have reached end of life. */
export const DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash';

export interface GeminiConfig {
    /** Gemini API key. Not required when `vertexai` is true and ADC is configured. */
    apiKey?: string;
    model?: string;
    /** Route through Vertex AI instead of the Gemini API. */
    vertexai?: boolean;
    /** GCP project, for Vertex AI. */
    project?: string;
    /** GCP location, for Vertex AI. */
    location?: string;
    /** Override the API endpoint, e.g. for a proxy. */
    baseUrl?: string;
    /**
     * Safety thresholds. Defaults to `BLOCK_MEDIUM_AND_ABOVE` across all four
     * harm categories — the previous implementation configured only harassment
     * and silently left the rest at their service defaults.
     */
    safetySettings?: SafetySetting[];
}

const DEFAULT_SAFETY_SETTINGS: SafetySetting[] = [
    HarmCategory.HARM_CATEGORY_HARASSMENT,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
].map((category) => ({ category, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE }));

/** Context windows by model family; the 1M default matches current Gemini models. */
function contextLengthFor(model: string): number {
    if (model.includes('flash-lite')) return 1_000_000;
    if (model.includes('flash')) return 1_000_000;
    if (model.includes('pro')) return 1_000_000;
    return 1_000_000;
}

export class GeminiLM extends BaseLM {
    private readonly client: GoogleGenAI;
    private readonly safetySettings: SafetySetting[];

    constructor(config: GeminiConfig = {}) {
        super('gemini', config.model ?? DEFAULT_GEMINI_MODEL);

        this.client = new GoogleGenAI({
            apiKey: config.apiKey,
            vertexai: config.vertexai,
            project: config.project,
            location: config.location,
            ...(config.baseUrl ? { httpOptions: { baseUrl: config.baseUrl } } : {}),
        });
        this.safetySettings = config.safetySettings ?? DEFAULT_SAFETY_SETTINGS;
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const { contents, systemInstruction } = toGeminiContents(messages);
        const response = await this.send(contents, systemInstruction, options);
        return response.text ?? '';
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const { contents, systemInstruction } = toGeminiContents([
            { role: 'user', content: prompt },
        ]);

        const response = await this.send(contents, systemInstruction, options, {
            responseMimeType: 'application/json',
            responseJsonSchema: schema,
        });

        const text = response.text ?? '';
        try {
            return JSON.parse(text) as T;
        } catch (cause) {
            throw new LMError('gemini', `Structured response was not valid JSON: ${text}`, {
                cause,
            });
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
        const { contents, systemInstruction } = toGeminiContents(messages);
        const startedAt = Date.now();

        let stream;
        try {
            stream = await this.client.models.generateContentStream({
                model: options?.model ?? this.model,
                contents,
                config: this.buildConfig(systemInstruction, options),
            });
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        let last: GenerateContentResponse | undefined;
        for await (const chunk of stream) {
            last = chunk;
            const text = chunk.text;
            if (text) {
                yield { content: text, done: false };
            }
        }

        this.recordUsageFrom(last, startedAt);
        yield { content: '', done: true, usage: usageFrom(last) };
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

    private async send(
        contents: Content[],
        systemInstruction: string | undefined,
        options?: LLMCallOptions,
        extraConfig?: Partial<GenerateContentConfig>
    ): Promise<GenerateContentResponse> {
        const startedAt = Date.now();

        let response: GenerateContentResponse;
        try {
            response = await this.client.models.generateContent({
                model: options?.model ?? this.model,
                contents,
                config: { ...this.buildConfig(systemInstruction, options), ...extraConfig },
            });
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        // Check the block reason before touching `text`. The previous
        // implementation read `response.text()` first, which threw on blocked
        // responses and made this branch unreachable.
        const blockReason = response.promptFeedback?.blockReason;
        if (blockReason) {
            this.recordError();
            throw new LMError('gemini', `Request blocked by safety filters: ${blockReason}`);
        }

        this.recordUsageFrom(response, startedAt);
        return response;
    }

    private buildConfig(
        systemInstruction: string | undefined,
        options?: LLMCallOptions
    ): GenerateContentConfig {
        const config: GenerateContentConfig = {
            safetySettings: this.safetySettings,
        };

        if (systemInstruction) config.systemInstruction = systemInstruction;
        if (options?.maxTokens !== undefined) config.maxOutputTokens = options.maxTokens;
        if (options?.temperature !== undefined) config.temperature = options.temperature;
        if (options?.topP !== undefined) config.topP = options.topP;
        if (options?.stopSequences) config.stopSequences = options.stopSequences;
        if (options?.frequencyPenalty !== undefined) {
            config.frequencyPenalty = options.frequencyPenalty;
        }
        if (options?.presencePenalty !== undefined) {
            config.presencePenalty = options.presencePenalty;
        }
        if (options?.timeout !== undefined) {
            config.abortSignal = AbortSignal.timeout(options.timeout);
        }

        return config;
    }

    private recordUsageFrom(
        response: GenerateContentResponse | undefined,
        startedAt: number
    ): void {
        const usage = response?.usageMetadata;
        this.recordUsage({
            promptTokens: usage?.promptTokenCount ?? 0,
            completionTokens: usage?.candidatesTokenCount ?? 0,
            latencyMs: Date.now() - startedAt,
        });
    }
}

function usageFrom(response: GenerateContentResponse | undefined) {
    const usage = response?.usageMetadata;
    return {
        promptTokens: usage?.promptTokenCount ?? 0,
        completionTokens: usage?.candidatesTokenCount ?? 0,
        totalTokens: usage?.totalTokenCount ?? 0,
    };
}

/**
 * Convert ts-dspy messages to Gemini contents.
 *
 * System messages become a separate `systemInstruction`, since Gemini has no
 * system role in `contents`. This does not mutate the caller's array — the
 * previous implementation called `messages.pop()`, destroying the last turn of
 * any array a caller reused.
 */
export function toGeminiContents(messages: ChatMessage[]): {
    contents: Content[];
    systemInstruction?: string;
} {
    const systemParts: string[] = [];
    const contents: Content[] = [];

    for (const message of messages) {
        if (message.role === 'system') {
            systemParts.push(message.content);
            continue;
        }
        contents.push({
            role: message.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: message.content }],
        });
    }

    return {
        contents,
        systemInstruction: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    };
}

function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    const message = error instanceof Error ? error.message : String(error);
    const status =
        typeof error === 'object' && error !== null && 'status' in error
            ? Number((error as { status: unknown }).status)
            : undefined;
    return new LMError('gemini', message, { cause: error, status });
}
