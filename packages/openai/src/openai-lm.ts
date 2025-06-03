import { ILanguageModel, LLMCallOptions, ChatMessage, UsageStats } from '@ts-dspy/core';

export interface OpenAIConfig {
    apiKey: string;
    model?: string;
    organization?: string;
    baseURL?: string;
}

export class OpenAILM implements ILanguageModel {
    private config: Required<OpenAIConfig>;
    private usage: UsageStats = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        totalCost: 0
    };

    constructor(config: OpenAIConfig) {
        this.config = {
            model: 'gpt-4',
            organization: '',
            baseURL: 'https://api.openai.com/v1',
            ...config
        };
    }

    async generate(prompt: string, options?: LLMCallOptions): Promise<string> {
        const messages: ChatMessage[] = [{ role: 'user', content: prompt }];
        return this.chat(messages, options);
    }

    async generateStructured<T>(prompt: string, schema: any, options?: LLMCallOptions): Promise<T> {
        const structuredPrompt = `${prompt}\n\nRespond with valid JSON matching this schema:\n${JSON.stringify(schema, null, 2)}`;
        const response = await this.generate(structuredPrompt, options);

        try {
            return JSON.parse(response) as T;
        } catch (error) {
            throw new Error(`Failed to parse structured output: ${error}`);
        }
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiKey}`
        };

        if (this.config.organization) {
            headers['OpenAI-Organization'] = this.config.organization;
        }

        const body = {
            model: this.config.model,
            messages,
            temperature: options?.temperature ?? 0.7,
            max_tokens: options?.maxTokens,
            top_p: options?.topP,
            frequency_penalty: options?.frequencyPenalty,
            presence_penalty: options?.presencePenalty,
            stop: options?.stopSequences
        };

        // Remove undefined values
        Object.keys(body).forEach(key => {
            if ((body as any)[key] === undefined) {
                delete (body as any)[key];
            }
        });

        const response = await fetch(`${this.config.baseURL}/chat/completions`, {
            method: 'POST',
            headers,
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            let errorMessage = `OpenAI API Error: ${response.status} ${response.statusText}`;
            try {
                const error = await response.json();
                errorMessage = `OpenAI API Error: ${error.error?.message || response.statusText}`;
            } catch {
                // Use the default error message if JSON parsing fails
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();

        // Update usage statistics
        if (data.usage) {
            this.usage.promptTokens += data.usage.prompt_tokens || 0;
            this.usage.completionTokens += data.usage.completion_tokens || 0;
            this.usage.totalTokens += data.usage.total_tokens || 0;

            // Estimate cost (rough pricing for GPT-3.5-turbo)
            const inputCost = (data.usage.prompt_tokens || 0) * 0.0015 / 1000;
            const outputCost = (data.usage.completion_tokens || 0) * 0.002 / 1000;
            this.usage.totalCost = (this.usage.totalCost || 0) + inputCost + outputCost;
        }

        return data.choices?.[0]?.message?.content || '';
    }

    getUsage(): UsageStats {
        return { ...this.usage };
    }

    resetUsage(): void {
        this.usage = {
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0,
            totalCost: 0
        };
    }

    // Additional utility methods
    setModel(model: string): void {
        this.config.model = model;
    }

    getModel(): string {
        return this.config.model;
    }

    setBaseURL(baseURL: string): void {
        this.config.baseURL = baseURL;
    }
} 