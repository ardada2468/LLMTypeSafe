import { z } from 'zod';
import {
    RespAct,
    type ToolFunction,
    type ToolWithDescription,
    type RespActEvent,
} from './respact';
import { Signature, OutputField, InputField } from '../core/signature';
import { BaseLM } from '../core/base-lm';
import { MockLM } from '../test-utils';
import type {
    ChatMessage,
    ChatResult,
    LLMCallOptions,
    ModelCapabilities,
} from '../types/language-model';

/**
 * A language model that advertises native tool calling and replies from a
 * script of `ChatResult`s. `MockLM` deliberately reports
 * `supportsFunctionCalling: false`, which is what keeps the rest of this suite
 * on the text path.
 */
class ToolCallingLM extends BaseLM {
    readonly turns: ChatMessage[][] = [];
    readonly options: Array<LLMCallOptions | undefined> = [];

    constructor(private readonly script: ChatResult[]) {
        super('fake', 'fake-model');
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        return (await this.chatWithTools(messages, options)).content;
    }

    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
        // Snapshot: RespAct appends to the same array across turns.
        this.turns.push(messages.map((message) => ({ ...message })));
        this.options.push(options);
        if (this.script.length === 0) {
            throw new Error('ToolCallingLM: no more scripted results');
        }
        return this.script.shift()!;
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: false,
            supportsStructuredOutput: false,
            supportsFunctionCalling: true,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
        };
    }
}

// These tests drive the real Module base class and the real parser. The previous
// suite mocked both, so it could not catch parsing or validation regressions.

describe('RespAct', () => {
    describe('tool normalization', () => {
        it('accepts bare functions and synthesizes a description', () => {
            const tool: ToolFunction = (input: string) => `Result: ${input}`;
            const agent = new RespAct('question -> answer', {
                tools: { testTool: tool },
                lm: new MockLM(),
            });

            const tools = (agent as any).tools;
            expect(tools.testTool.description).toBe('Tool: testTool');
            expect(tools.testTool.function).toBe(tool);
        });

        it('accepts tools that carry their own description', () => {
            const tool: ToolWithDescription = {
                description: 'A test tool that processes input',
                function: (input: string) => `Processed: ${input}`,
            };
            const agent = new RespAct('question -> answer', {
                tools: { advancedTool: tool },
                lm: new MockLM(),
            });

            const tools = (agent as any).tools;
            expect(tools.advancedTool.description).toBe('A test tool that processes input');
            expect(tools.advancedTool.function).toBe(tool.function);
        });

        it('handles both shapes side by side', () => {
            const agent = new RespAct('question -> answer', {
                tools: {
                    legacy: (input: string) => `Legacy: ${input}`,
                    modern: {
                        description: 'Modern tool with description',
                        function: (input: string) => `Modern: ${input}`,
                    },
                },
                lm: new MockLM(),
            });

            const tools = (agent as any).tools;
            expect(tools.legacy.description).toBe('Tool: legacy');
            expect(tools.modern.description).toBe('Modern tool with description');
        });
    });

    describe('prompt building', () => {
        it('lists every tool with its description', () => {
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    calculate: {
                        description: 'Performs mathematical calculations',
                        function: () => 0,
                    },
                    search: {
                        description: 'Searches for information online',
                        function: (query: string) => `Search results for: ${query}`,
                    },
                },
            });

            const prompt = (agent as any).buildInitialPrompt({ question: 'Test question' });

            expect(prompt).toContain('- calculate: Performs mathematical calculations');
            expect(prompt).toContain('- search: Searches for information online');
            expect(prompt).toContain('Question: Test question');
        });

        it('spells out required output fields for class signatures', () => {
            class Answer extends Signature {
                @InputField({ description: 'the question' })
                question!: string;

                @OutputField({ description: 'the answer' })
                answer!: string;

                @OutputField({ description: 'confidence', type: 'number' })
                confidence!: number;
            }

            const agent = new RespAct(Answer, { tools: {}, lm: new MockLM() });
            const prompt = (agent as any).buildInitialPrompt({ question: 'Q' });

            expect(prompt).toContain('answer: [your response for answer]');
            expect(prompt).toContain('confidence: [your response for confidence]');
        });
    });

    describe('tool execution', () => {
        it('awaits async tools and returns their result', async () => {
            const tool = vi.fn().mockResolvedValue('Tool result');
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: { testTool: { description: 'Test tool', function: tool } },
            });

            const result = await (agent as any).executeTool('testTool', 'test input', 0);

            expect(tool).toHaveBeenCalledWith('test input');
            expect(result).toBe('Tool result');
        });

        it('stringifies results from synchronous tools', async () => {
            const tool = vi.fn().mockReturnValue('Legacy result');
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: { legacyTool: tool },
            });

            const result = await (agent as any).executeTool('legacyTool', 'test input', 0);

            expect(tool).toHaveBeenCalledWith('test input');
            expect(result).toBe('Legacy result');
        });

        it('turns a thrown tool error into an observation', async () => {
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    failingTool: {
                        description: 'A tool that fails',
                        function: () => {
                            throw new Error('Tool failed');
                        },
                    },
                },
            });

            const result = await (agent as any).executeTool('failingTool', 'input', 0);
            expect(result).toContain('Error executing failingTool: Tool failed');
        });

        it('reports the available tools when asked for an unknown one', async () => {
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: { knownTool: { description: 'A known tool', function: () => 'result' } },
            });

            const result = await (agent as any).executeTool('unknownTool', 'input', 0);
            expect(result).toContain("Error: Tool 'unknownTool' not found");
            expect(result).toContain('Available tools: knownTool');
        });
    });

    describe('reasoning loop', () => {
        it('calls a tool, then returns the validated final answer', async () => {
            const calculator = vi.fn().mockReturnValue(42);
            const lm = new MockLM({
                responses: [
                    'I need to calculate. Action: calculator\nAction Input: 6 * 7',
                    'Final Answer: answer: The result is 42',
                ],
            });

            const agent = new RespAct('question -> answer', {
                tools: {
                    calculator: {
                        description: 'Performs arithmetic',
                        function: calculator,
                    },
                },
                maxSteps: 3,
                lm,
            });

            const result = await agent.forward({ question: 'What is 6 times 7?' });

            expect(calculator).toHaveBeenCalledWith('6 * 7');
            expect(result.answer).toBe('The result is 42');
            expect(result.steps).toBe(2);
        });

        it('validates typed output fields on the final answer', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'the answer' })
                answer!: string;

                @OutputField({ description: 'confidence', type: 'number' })
                confidence!: number;
            }

            const lm = new MockLM({
                responses: ['Final Answer: answer: Paris\nconfidence: 0.9'],
            });
            const agent = new RespAct(Scored, { tools: {}, maxSteps: 2, lm });

            const result = await agent.forward({ question: 'Capital of France?' });

            expect(result.answer).toBe('Paris');
            expect(result.confidence).toBe(0.9);
            expect(typeof result.confidence).toBe('number');
        });

        it('asks again when the final answer is missing a required field', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'the answer' })
                answer!: string;

                @OutputField({ description: 'confidence', type: 'number' })
                confidence!: number;
            }

            const lm = new MockLM({
                responses: [
                    'Final Answer: answer: Paris',
                    'Final Answer: answer: Paris\nconfidence: 0.8',
                ],
            });
            const agent = new RespAct(Scored, { tools: {}, maxSteps: 4, lm });

            const result = await agent.forward({ question: 'Capital of France?' });

            expect(result.confidence).toBe(0.8);
            expect(result.steps).toBe(2);
            expect(lm.lastPrompt()).toContain('missing or malformed for: confidence');
        });

        it('does not repeat an identical tool call', async () => {
            const tool = vi.fn().mockReturnValue('data');
            const lm = new MockLM({
                responses: [
                    'Action: fetch\nAction Input: same',
                    'Action: fetch\nAction Input: same',
                    'Final Answer: answer: done',
                ],
            });
            const events: RespActEvent[] = [];

            const agent = new RespAct('question -> answer', {
                tools: { fetch: tool },
                maxSteps: 5,
                lm,
                onEvent: (event) => events.push(event),
            });

            const result = await agent.forward({ question: 'Q' });

            expect(tool).toHaveBeenCalledTimes(1);
            expect(result.answer).toBe('done');
            expect(events.some((event) => event.type === 'repeated_tool_call')).toBe(true);
        });

        it('passes call options through to the language model', async () => {
            const lm = new MockLM({ responses: ['Final Answer: answer: ok'] });
            const agent = new RespAct('question -> answer', { tools: {}, lm });

            await agent.forward({ question: 'Q' }, { temperature: 0.1, timeout: 5000 });

            expect(lm.calls[0].options).toEqual({ temperature: 0.1, timeout: 5000 });
        });

        it('throws when the loop runs out of steps', async () => {
            const lm = new MockLM({ responses: ['thinking...', 'still thinking...'] });
            const agent = new RespAct('question -> answer', { tools: {}, maxSteps: 2, lm });

            await expect(agent.forward({ question: 'Q' })).rejects.toThrow(
                'RespAct exceeded maximum steps (2)'
            );
        });

        it('emits events for each stage of the loop', async () => {
            const lm = new MockLM({
                responses: ['Action: echo\nAction Input: hi', 'Final Answer: answer: hi'],
            });
            const events: RespActEvent[] = [];

            const agent = new RespAct('question -> answer', {
                tools: { echo: (input: string) => input },
                lm,
                onEvent: (event) => events.push(event),
            });

            await agent.forward({ question: 'Q' });

            const types = events.map((event) => event.type);
            expect(types).toContain('thought');
            expect(types).toContain('tool_call');
            expect(types).toContain('tool_result');
        });
    });

    describe('typed tools', () => {
        it('derives a JSON Schema from a Zod argument schema', () => {
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: ({ a, b }: { a: number; b: number }) => a + b,
                    },
                },
            });

            const schema = (agent as any).tools.add.parameters;
            expect(schema.type).toBe('object');
            expect(Object.keys(schema.properties)).toEqual(['a', 'b']);
            expect(schema.required).toEqual(['a', 'b']);
        });

        it('synthesizes a single-string schema for an untyped tool', () => {
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: { echo: (input: string) => input },
            });

            const schema = (agent as any).tools.echo.parameters;
            expect(Object.keys(schema.properties)).toEqual(['input']);
            expect((agent as any).tools.echo.typed).toBe(false);
        });

        it('accepts a raw JSON Schema without wrapping it in Zod', () => {
            const parameters = {
                type: 'object',
                properties: { city: { type: 'string' } },
                required: ['city'],
            };
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    weather: { description: 'Weather', parameters, function: () => 'sunny' },
                },
            });

            expect((agent as any).tools.weather.parameters).toBe(parameters);
            expect((agent as any).tools.weather.validator).toBeUndefined();
        });

        it('parses a JSON Action Input into named arguments in text mode', async () => {
            const add = vi.fn(({ a, b }: { a: number; b: number }) => a + b);
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: add,
                    },
                },
            });

            const result = await (agent as any).executeTool('add', '{"a": 2, "b": 3}', 0);

            expect(add).toHaveBeenCalledWith({ a: 2, b: 3 });
            expect(result).toBe('5');
        });

        it('accepts a bare value in text mode when the schema has one property', async () => {
            const shout = vi.fn(({ text }: { text: string }) => text.toUpperCase());
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    shout: {
                        description: 'Uppercase text',
                        parameters: z.object({ text: z.string() }),
                        function: shout,
                    },
                },
            });

            expect(await (agent as any).executeTool('shout', 'hello', 0)).toBe('HELLO');
            expect(shout).toHaveBeenCalledWith({ text: 'hello' });
        });

        it('turns a Zod validation failure into a correctable observation', async () => {
            const add = vi.fn();
            const events: RespActEvent[] = [];
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                onEvent: (event) => events.push(event),
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: add,
                    },
                },
            });

            const result = await (agent as any).executeTool('add', '{"a": 2}', 0);

            expect(add).not.toHaveBeenCalled();
            expect(result).toContain('Invalid arguments');
            expect(result).toContain('b');
            expect(events.some((event) => event.type === 'tool_error')).toBe(true);
        });

        it('coerces a bare value to the declared scalar type in text mode', async () => {
            const double = vi.fn(({ n }: { n: number }) => n * 2);
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    double: {
                        description: 'Double a number',
                        parameters: z.object({ n: z.number() }),
                        function: double,
                    },
                },
            });

            // The text loop only ever produces a line of text, so a numeric
            // argument would otherwise fail validation on every attempt.
            expect(await (agent as any).executeTool('double', '21', 0)).toBe('42');
            expect(double).toHaveBeenCalledWith({ n: 21 });
        });

        it('lists the required argument keys in the text-mode prompt', () => {
            const agent = new RespAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: () => 0,
                    },
                },
            });

            const prompt = (agent as any).buildInitialPrompt({ question: 'Q' });
            expect(prompt).toContain('Action Input must be a JSON object with keys: a, b');
        });
    });

    describe('native tool calling', () => {
        it('calls a multi-argument tool and returns the final answer', async () => {
            const add = vi.fn(({ a, b }: { a: number; b: number }) => a + b);
            const lm = new ToolCallingLM([
                {
                    content: 'Let me add those.',
                    toolCalls: [{ id: 'call_1', name: 'add', arguments: { a: 6, b: 7 } }],
                    finishReason: 'tool_calls',
                },
                { content: 'The result is 13', finishReason: 'stop' },
            ]);

            const agent = new RespAct('question -> answer', {
                lm,
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: add,
                    },
                },
            });

            const result = await agent.forward({ question: 'What is 6 plus 7?' });

            expect(add).toHaveBeenCalledWith({ a: 6, b: 7 });
            expect(result.answer).toBe('The result is 13');
            expect(result.steps).toBe(2);
        });

        it('advertises every tool schema on each request', async () => {
            const lm = new ToolCallingLM([{ content: 'done', finishReason: 'stop' }]);
            const agent = new RespAct('question -> answer', {
                lm,
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: () => 0,
                    },
                },
            });

            await agent.forward({ question: 'Q' });

            expect(lm.options[0]?.tools).toEqual([
                {
                    name: 'add',
                    description: 'Add two numbers',
                    parameters: expect.objectContaining({ type: 'object' }),
                },
            ]);
        });

        it('records the tool result as a correlated tool turn', async () => {
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [{ id: 'call_9', name: 'echo', arguments: { input: 'hi' } }],
                },
                { content: 'hi', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', {
                lm,
                tools: { echo: (input: string) => input },
            });

            await agent.forward({ question: 'Q' });

            const secondTurn = lm.turns[1];
            expect(secondTurn.at(-2)).toMatchObject({
                role: 'assistant',
                toolCalls: [{ id: 'call_9', name: 'echo' }],
            });
            expect(secondTurn.at(-1)).toEqual({
                role: 'tool',
                name: 'echo',
                toolCallId: 'call_9',
                content: 'hi',
            });
        });

        it('passes a bare string through to an untyped tool', async () => {
            const echo = vi.fn((input: string) => `echo:${input}`);
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [{ id: 'c1', name: 'echo', arguments: { input: 'hi' } }],
                },
                { content: 'echo:hi', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', { lm, tools: { echo } });

            await agent.forward({ question: 'Q' });

            expect(echo).toHaveBeenCalledWith('hi');
        });

        it('runs parallel tool calls from a single turn', async () => {
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [
                        { id: 'c1', name: 'left', arguments: { input: 'a' } },
                        { id: 'c2', name: 'right', arguments: { input: 'b' } },
                    ],
                },
                { content: 'ab', finishReason: 'stop' },
            ]);
            const left = vi.fn((input: string) => `L${input}`);
            const right = vi.fn((input: string) => `R${input}`);

            const agent = new RespAct('question -> answer', { lm, tools: { left, right } });
            await agent.forward({ question: 'Q' });

            expect(left).toHaveBeenCalledWith('a');
            expect(right).toHaveBeenCalledWith('b');
            const toolTurns = lm.turns[1].filter((message) => message.role === 'tool');
            expect(toolTurns.map((message) => message.content)).toEqual(['La', 'Rb']);
        });

        it('does not repeat an identical native tool call', async () => {
            const fetchTool = vi.fn(() => 'data');
            const lm = new ToolCallingLM([
                { content: '', toolCalls: [{ id: 'c1', name: 'fetch', arguments: { q: 1 } }] },
                { content: '', toolCalls: [{ id: 'c2', name: 'fetch', arguments: { q: 1 } }] },
                { content: 'done', finishReason: 'stop' },
            ]);
            const events: RespActEvent[] = [];

            const agent = new RespAct('question -> answer', {
                lm,
                onEvent: (event) => events.push(event),
                tools: {
                    fetch: {
                        description: 'Fetch',
                        parameters: z.object({ q: z.number() }),
                        function: fetchTool,
                    },
                },
            });

            const result = await agent.forward({ question: 'Q' });

            expect(fetchTool).toHaveBeenCalledTimes(1);
            expect(result.answer).toBe('done');
            expect(events.some((event) => event.type === 'repeated_tool_call')).toBe(true);
        });

        it('emits the same event surface as the text loop', async () => {
            const lm = new ToolCallingLM([
                {
                    content: 'thinking',
                    toolCalls: [{ id: 'c1', name: 'echo', arguments: { input: 'hi' } }],
                },
                { content: 'hi', finishReason: 'stop' },
            ]);
            const events: RespActEvent[] = [];

            const agent = new RespAct('question -> answer', {
                lm,
                tools: { echo: (input: string) => input },
                onEvent: (event) => events.push(event),
            });

            await agent.forward({ question: 'Q' });

            // The final turn's prose is a thought too, exactly as in text mode.
            expect(events.map((event) => event.type)).toEqual([
                'thought',
                'tool_call',
                'tool_result',
                'thought',
            ]);
            expect(events[1]).toMatchObject({ tool: 'echo', input: 'hi' });
        });

        it('asks again when the final answer is missing a required field', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'the answer' })
                answer!: string;

                @OutputField({ description: 'confidence', type: 'number' })
                confidence!: number;
            }

            const lm = new ToolCallingLM([
                { content: 'answer: Paris', finishReason: 'stop' },
                { content: 'answer: Paris\nconfidence: 0.8', finishReason: 'stop' },
            ]);
            const agent = new RespAct(Scored, {
                lm,
                tools: { echo: (input: string) => input },
            });

            const result = await agent.forward({ question: 'Capital of France?' });

            expect(result.confidence).toBe(0.8);
            const nudge = lm.turns[1].at(-1);
            expect(nudge?.role).toBe('user');
            expect(nudge?.content).toContain('missing or malformed for: confidence');
        });

        it('surfaces a failing tool as an observation rather than throwing', async () => {
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [{ id: 'c1', name: 'boom', arguments: { input: 'x' } }],
                },
                { content: 'recovered', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', {
                lm,
                tools: {
                    boom: () => {
                        throw new Error('kaboom');
                    },
                },
            });

            const result = await agent.forward({ question: 'Q' });

            expect(result.answer).toBe('recovered');
            expect(lm.turns[1].at(-1)?.content).toContain('Error executing boom: kaboom');
        });

        it('falls back to the text loop when the model cannot call tools', async () => {
            const add = vi.fn(({ a, b }: { a: number; b: number }) => a + b);
            const lm = new MockLM({
                responses: [
                    'Action: add\nAction Input: {"a": 6, "b": 7}',
                    'Final Answer: answer: The result is 13',
                ],
            });

            const agent = new RespAct('question -> answer', {
                lm,
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: add,
                    },
                },
            });

            const result = await agent.forward({ question: 'What is 6 plus 7?' });

            expect(add).toHaveBeenCalledWith({ a: 6, b: 7 });
            expect(result.answer).toBe('The result is 13');
        });

        it('stays on the text loop when the model inherited chatWithTools', async () => {
            // The flag existed long before anything read it, so a custom LM can
            // report it while inheriting BaseLM's text-only `chatWithTools`.
            // Taking the native path there would silently never call a tool.
            const lm = new MockLM({
                capabilities: { supportsFunctionCalling: true },
                responses: ['Action: echo\nAction Input: hi', 'Final Answer: answer: hi'],
            });
            const echo = vi.fn((input: string) => input);
            const agent = new RespAct('question -> answer', { lm, tools: { echo } });

            const result = await agent.forward({ question: 'Q' });

            expect(echo).toHaveBeenCalledWith('hi');
            expect(result.answer).toBe('hi');
        });

        it('does not record an empty assistant turn when the model says nothing', async () => {
            const lm = new ToolCallingLM([
                { content: '', finishReason: 'stop' },
                { content: 'done', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', {
                lm,
                tools: { echo: (input: string) => input },
            });

            await agent.forward({ question: 'Q' });

            // Providers reject a non-final assistant turn with empty content,
            // so the nudge goes in on its own.
            expect(lm.turns[1].some((message) => message.role === 'assistant')).toBe(false);
            expect(lm.turns[1].at(-1)?.role).toBe('user');
        });

        it('reports unparseable provider arguments instead of running the tool', async () => {
            const echo = vi.fn((input: string) => input);
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [
                        {
                            id: 'c1',
                            name: 'echo',
                            arguments: {},
                            rawArguments: '{not json',
                        },
                    ],
                },
                { content: 'recovered', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', { lm, tools: { echo } });

            await agent.forward({ question: 'Q' });

            expect(echo).not.toHaveBeenCalled();
            expect(lm.turns[1].at(-1)?.content).toContain('not valid JSON');
        });

        it('rejects a non-string argument for an untyped tool', async () => {
            const echo = vi.fn((input: string) => input);
            const lm = new ToolCallingLM([
                { content: '', toolCalls: [{ id: 'c1', name: 'echo', arguments: { q: 1 } }] },
                { content: 'recovered', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', { lm, tools: { echo } });

            await agent.forward({ question: 'Q' });

            expect(echo).not.toHaveBeenCalled();
            expect(lm.turns[1].at(-1)?.content).toContain('Expected a single string argument');
        });

        it('honours forceTextMode on a tool-calling model', async () => {
            const lm = new ToolCallingLM([
                { content: ' Final Answer: answer: done', finishReason: 'stop' },
            ]);
            const agent = new RespAct('question -> answer', {
                lm,
                forceTextMode: true,
                tools: { echo: (input: string) => input },
            });

            const result = await agent.forward({ question: 'Q' });

            expect(result.answer).toBe('done');
            // The text loop sends one user turn and declares no tools.
            expect(lm.turns[0]).toHaveLength(1);
            expect(lm.options[0]?.tools).toBeUndefined();
        });
    });
});
