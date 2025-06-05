import { GeminiLM, GeminiConfig } from './gemini-lm';
import { ChatMessage } from '@ts-dspy/core';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Mock the GoogleGenerativeAI class
jest.mock('@google/generative-ai', () => {
    const mockChatSession = {
        sendMessage: jest.fn(),
    };
    const mockModel: any = {
        startChat: jest.fn(() => mockChatSession),
        getGenerativeModel: jest.fn(),
    };
    mockModel.getGenerativeModel.mockImplementation(() => mockModel);

    return {
        GoogleGenerativeAI: jest.fn(() => mockModel),
        HarmCategory: {
            HARM_CATEGORY_HARASSMENT: 'HARM_CATEGORY_HARASSMENT',
        },
        HarmBlockThreshold: {
            BLOCK_MEDIUM_AND_ABOVE: 'BLOCK_MEDIUM_AND_ABOVE',
        }
    };
});

const mockGoogleGenerativeAI = GoogleGenerativeAI as jest.Mock;
const mockStartChat = new (mockGoogleGenerativeAI)().getGenerativeModel().startChat;
const mockSendMessage = mockStartChat().sendMessage;

describe('GeminiLM', () => {
    const apiKey = 'test-api-key';
    let lm: GeminiLM;

    beforeEach(() => {
        jest.clearAllMocks();
        const mockModel = new (mockGoogleGenerativeAI)().getGenerativeModel();
        mockStartChat.mockClear();
        mockSendMessage.mockClear();
        (mockModel.startChat as jest.Mock).mockReturnValue({ sendMessage: mockSendMessage });

        lm = new GeminiLM({ apiKey });
    });

    describe('constructor', () => {
        it('should initialize with API key and default model', () => {
            expect(lm.getModel()).toBe('gemini-2.0-flash');
            expect(mockGoogleGenerativeAI).toHaveBeenCalledWith(apiKey);
        });

        it('should allow overriding default model', () => {
            const customModel = 'gemini-1.5-pro-latest';
            const customLm = new GeminiLM({ apiKey, model: customModel });
            expect(customLm.getModel()).toBe(customModel);
        });
    });

    describe('generate', () => {
        it('should call chat method with the prompt', async () => {
            const prompt = 'Hello, world!';
            const responseText = 'Response from Gemini';

            mockSendMessage.mockResolvedValue({
                response: {
                    text: () => responseText,
                    promptFeedback: {},
                },
            });

            const result = await lm.generate(prompt);
            expect(result).toBe(responseText);
            expect(mockStartChat).toHaveBeenCalled();
            expect(mockSendMessage).toHaveBeenCalledWith(prompt);
        });
    });

    describe('chat', () => {
        it('should handle a conversation history', async () => {
            const messages: ChatMessage[] = [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi there!' },
                { role: 'user', content: 'How are you?' },
            ];
            const responseText = 'I am a large language model.';

            mockSendMessage.mockResolvedValue({
                response: {
                    text: () => responseText,
                    promptFeedback: {},
                },
            });

            const result = await lm.chat(messages);

            expect(result).toBe(responseText);
            expect(mockStartChat).toHaveBeenCalledWith(expect.objectContaining({
                history: [
                    { role: 'user', parts: [{ text: 'Hello' }] },
                    { role: 'model', parts: [{ text: 'Hi there!' }] },
                ],
            }));
            expect(mockSendMessage).toHaveBeenCalledWith('How are you?');
        });

        it('should throw an error if the response is blocked', async () => {
            const messages: ChatMessage[] = [{ role: 'user', content: 'A harmful prompt' }];

            mockSendMessage.mockResolvedValue({
                response: {
                    text: () => '',
                    promptFeedback: { blockReason: 'SAFETY' },
                },
            });

            await expect(lm.chat(messages)).rejects.toThrow('Gemini API Error: SAFETY');
        });
    });

    describe('generateStructured', () => {
        it('should correctly format prompt and parse valid JSON response', async () => {
            const prompt = "Extract user details.";
            const schema = { type: "object", properties: { name: { type: "string" }, age: { type: "number" } } };
            const mockApiResponse = { name: "John Doe", age: 30 };
            const responseText = JSON.stringify(mockApiResponse);

            mockSendMessage.mockResolvedValue({
                response: {
                    text: () => responseText,
                    promptFeedback: {},
                },
            });

            const result = await lm.generateStructured<{ name: string, age: number }>(prompt, schema);
            expect(result).toEqual(mockApiResponse);

            const lastMessage = mockSendMessage.mock.calls[0][0];
            expect(lastMessage).toContain(prompt);
            expect(lastMessage).toContain("Respond with valid JSON matching this schema:");
            expect(lastMessage).toContain(JSON.stringify(schema, null, 2));
        });

        it('should throw error if API response is not valid JSON', async () => {
            const prompt = "Extract user details.";
            const schema = { type: "object", properties: { name: { type: "string" } } };

            mockSendMessage.mockResolvedValue({
                response: {
                    text: () => "This is not JSON",
                    promptFeedback: {},
                },
            });

            await expect(lm.generateStructured(prompt, schema)).rejects.toThrow(/Failed to parse structured output/);
        });
    });

    describe('Usage Tracking', () => {
        it('should return zero usage as Gemini API does not provide it', () => {
            const usage = lm.getUsage();
            expect(usage.promptTokens).toBe(0);
            expect(usage.completionTokens).toBe(0);
            expect(usage.totalTokens).toBe(0);
            expect(usage.totalCost).toBe(0);
        });

        it('should remain zero after a call', async () => {
            mockSendMessage.mockResolvedValue({
                response: {
                    text: () => "some response",
                    promptFeedback: {},
                },
            });

            await lm.generate("test");
            const usage = lm.getUsage();
            expect(usage.totalTokens).toBe(0);
        });

        it('should reset usage to zero', () => {
            // It starts at zero, so this just confirms the method exists and works
            lm.resetUsage();
            const usage = lm.getUsage();
            expect(usage.totalTokens).toBe(0);
        });
    });

    describe('Configuration', () => {
        it('should be able to get and set the model', () => {
            lm.setModel('gemini-1.5-flash');
            expect(lm.getModel()).toBe('gemini-1.5-flash');
        });
    });
}); 