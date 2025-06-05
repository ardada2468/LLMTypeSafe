import { ILanguageModel, LLMCallOptions, ChatMessage, UsageStats, ModelCapabilities } from '@ts-dspy/core';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';

export interface GeminiConfig {
    apiKey: string;
    model?: string;
}

export class GeminiLM implements ILanguageModel {
    private client: GoogleGenerativeAI;
    private config: Required<GeminiConfig>;
    private usage: UsageStats = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        totalCost: 0,
    };

    constructor(config: GeminiConfig) {
        this.config = {
            model: 'gemini-2.0-flash',
            ...config,
        };
        this.client = new GoogleGenerativeAI(this.config.apiKey);
    }

    async generate(prompt: string, options?: LLMCallOptions): Promise<string> {
        const messages: ChatMessage[] = [{ role: 'user', content: prompt }];
        return this.chat(messages, options);
    }

    async generateStructured<T>(prompt: string, schema: any, options?: LLMCallOptions): Promise<T> {
        const structuredPrompt = `${prompt}\n\nRespond with valid JSON matching this schema:\n${JSON.stringify(
            schema,
            null,
            2
        )}`;
        const response = await this.generate(structuredPrompt, options);

        try {
            return JSON.parse(response) as T;
        } catch (error) {
            throw new Error(`Failed to parse structured output: ${error}`);
        }
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const model = this.client.getGenerativeModel({
            model: this.config.model,
        });
        const lastMessage = messages.pop();
        if (!lastMessage) {
            return '';
        }

        const chat = model.startChat({
            history: messages.map((msg) => ({
                role: msg.role === 'assistant' ? 'model' : msg.role,
                parts: [{ text: msg.content }],
            })),
            generationConfig: {
                maxOutputTokens: options?.maxTokens,
                temperature: options?.temperature,
                topP: options?.topP,
                stopSequences: options?.stopSequences,
            },
            safetySettings: [
                {
                    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                },
            ],
        });

        const result = await chat.sendMessage(lastMessage.content);

        const response = result.response;
        const responseText = response.text();

        if (result.response.promptFeedback?.blockReason) {
            throw new Error(`Gemini API Error: ${result.response.promptFeedback.blockReason}`);
        }

        // Unfortunately, the new Gemini API does not provide token usage stats yet.
        // We will have to manually estimate or leave it as 0.

        return responseText;
    }

    getUsage(): UsageStats {
        console.log('Gemini does not provide token usage stats yet.');
        return { ...this.usage };
    }

    resetUsage(): void {
        this.usage = {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            totalCost: 0,
        };
    }

    setModel(model: string): void {
        this.config.model = model;
    }

    getModel(): string {
        return this.config.model;
    }

    getModelName(): string {
        return this.config.model;
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: false,
            supportsStructuredOutput: true,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 32768, // Gemini 1.0 Pro has a 32k context window
            supportedFormats: ['json_object'],
        };
    }
} 