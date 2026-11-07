import { z } from 'zod';
import { ReAct, type ToolFunction, type ToolWithDescription, type ReActEvent } from './react';
import { RespAct } from './respact';
import { Predict } from './predict';
import { ChainOfThought } from './chain-of-thought';
import { Signature, OutputField, InputField, signature } from '../core/signature';
import { BaseLM } from '../core/base-lm';
import { ChatAdapter } from '../adapters/chat';
import { MockLM } from '../test-utils';
import { TsDspyError } from '../core/errors';
import { configure } from '../core/config';
import { clearHistory, inspectHistory } from '../core/trace';
import type {
    ChatMessage,
    ChatResult,
    LLMCallOptions,
    ModelCapabilities,
} from '../types/language-model';
import type { LMCall } from '../types/module';

/**
 * A language model that advertises native tool calling and replies from a
 * script of `ChatResult`s. `MockLM` deliberately reports
 * `supportsFunctionCalling: false`, which is what keeps the rest of this suite
 * on the adapter (text) path.
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
        // Snapshot: ReAct appends to the same array across turns.
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

// These tests drive the real Module base class, adapter and parser.
//
// Rewritten for 0.7 (RespAct -> ReAct): every step is now a Predict over the
// signature's inputs plus `trajectory`, producing next_thought /
// next_tool_name / next_tool_args through the adapter, and a ChainOfThought
// extract step produces the outputs. The scripted replies changed from the
// old "Thought / Action / Final Answer" protocol accordingly, running out of
// steps now proceeds to the extract step instead of throwing MaxStepsError, and
// a malformed native final answer goes to the extract step instead of a nudge.

/** One step's reply, in ChatAdapter's marker format. */
function step(thought: string, tool: string, args: Record<string, unknown> | string): string {
    const argText = typeof args === 'string' ? args : JSON.stringify(args);
    return (
        `[[ ## next_thought ## ]]\n${thought}\n\n` +
        `[[ ## next_tool_name ## ]]\n${tool}\n\n` +
        `[[ ## next_tool_args ## ]]\n${argText}\n\n[[ ## completed ## ]]`
    );
}

const finish = (thought = 'I have what I need.') => step(thought, 'finish', {});

/** The extract step's reply: reasoning, then the signature's outputs. */
function extract(outputs: Record<string, unknown>, reasoning = 'From the trajectory.'): string {
    return [
        `[[ ## reasoning ## ]]\n${reasoning}`,
        ...Object.entries(outputs).map(([name, value]) => `[[ ## ${name} ## ]]\n${value}`),
        '[[ ## completed ## ]]',
    ].join('\n\n');
}

describe('ReAct', () => {
    describe('tool normalization', () => {
        // Tools are normalised into `Tool` instances internally (stored as
        // `{ tool, typed }`), so these reach through `.tool`.
        it('accepts bare functions and synthesizes a description', async () => {
            const tool: ToolFunction = (input: string) => `Result: ${input}`;
            const agent = new ReAct('question -> answer', {
                tools: { testTool: tool },
                lm: new MockLM(),
            });

            const tools = (agent as any).tools;
            expect(tools.testTool.tool.description).toBe('Tool: testTool');
            expect(await tools.testTool.tool.call({ input: 'x' })).toBe('Result: x');
        });

        it('accepts tools that carry their own description', async () => {
            const tool: ToolWithDescription = {
                description: 'A test tool that processes input',
                function: (input: string) => `Processed: ${input}`,
            };
            const agent = new ReAct('question -> answer', {
                tools: { advancedTool: tool },
                lm: new MockLM(),
            });

            const tools = (agent as any).tools;
            expect(tools.advancedTool.tool.description).toBe(
                'A test tool that processes input'
            );
            expect(await tools.advancedTool.tool.call({ input: 'x' })).toBe('Processed: x');
        });

        it('handles both shapes side by side', () => {
            const agent = new ReAct('question -> answer', {
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
            expect(tools.legacy.tool.description).toBe('Tool: legacy');
            expect(tools.modern.tool.description).toBe('Modern tool with description');
        });

        it('reserves the name finish', () => {
            expect(
                () => new ReAct('question -> answer', { tools: { finish: () => 'x' } })
            ).toThrow(TsDspyError);
        });
    });

    describe('signature and prompt', () => {
        const calculator = {
            description: 'Performs mathematical calculations',
            function: () => 0,
        };

        it('lists every tool, and finish, in the step instructions', async () => {
            const lm = new MockLM({ responses: [finish(), extract({ answer: 'x' })] });
            const agent = new ReAct('question -> answer', {
                lm,
                tools: {
                    calculate: calculator,
                    search: {
                        description: 'Searches for information online',
                        function: (query: string) => `Search results for: ${query}`,
                    },
                },
            });

            await agent.forward({ question: 'Test question' });

            const system = String(lm.calls[0].messages[0].content);
            expect(system).toContain(
                '(1) calculate, whose description is <desc>Performs mathematical calculations</desc>'
            );
            expect(system).toContain('(2) search, whose description is');
            expect(system).toContain('(3) finish, whose description is <desc>Marks the task');
            // The tool name is an enum of the tools plus finish.
            expect(system).toContain('one of: calculate; search; finish');
            expect(lm.calls[0].messages.at(-1)?.content).toContain(
                '[[ ## question ## ]]\nTest question'
            );
        });

        it('builds its step signature from the inputs plus trajectory', () => {
            const agent = new ReAct('question -> answer', { lm: new MockLM(), tools: {} });
            const stepSignature = (agent.react as any).signature;
            const extractSignature = (agent.extract as any).signature;

            expect(Object.keys(stepSignature.getInputFields())).toEqual([
                'question',
                'trajectory',
            ]);
            expect(Object.keys(stepSignature.getOutputFields())).toEqual([
                'next_thought',
                'next_tool_name',
                'next_tool_args',
            ]);
            expect(Object.keys(extractSignature.getOutputFields())).toEqual(['answer']);
        });

        it('works with a signature that has no question input', async () => {
            class Brief extends Signature {
                static description = 'Write a briefing.';

                @InputField({ description: 'the topic' })
                topic!: string;

                @InputField({ description: 'who will read it' })
                audience!: string;

                @OutputField({ description: 'the briefing' })
                summary!: string;
            }
            const lookup = vi.fn().mockReturnValue('Rates rose 0.25%.');
            const lm = new MockLM({
                responses: [
                    step('Look it up.', 'lookup', { input: 'interest rates' }),
                    finish(),
                    extract({ summary: 'Rates rose a quarter point.' }),
                ],
            });

            const result = await new ReAct(Brief, { lm, tools: { lookup } }).forward({
                topic: 'interest rates',
                audience: 'executives',
            });

            expect(lookup).toHaveBeenCalledWith('interest rates');
            expect(result.summary).toBe('Rates rose a quarter point.');
            const firstTurn = String(lm.calls[0].messages.at(-1)?.content);
            expect(firstTurn).toContain('[[ ## topic ## ]]\ninterest rates');
            expect(firstTurn).toContain('[[ ## audience ## ]]\nexecutives');
            expect(String(lm.calls[0].messages[0].content)).toContain('Write a briefing.');
        });

        it('works with a zod signature', async () => {
            const Lookup = signature({
                input: z.object({ city: z.string() }),
                output: z.object({ temperature: z.number() }),
            });
            const lm = new MockLM({
                responses: [
                    step('Check the weather.', 'weather', { input: 'Oslo' }),
                    finish(),
                    extract({ temperature: '-3' }),
                ],
            });

            const result = await new ReAct(Lookup, {
                lm,
                tools: { weather: () => '-3C' },
            }).forward({ city: 'Oslo' });

            expect(result.temperature).toBe(-3);
            expect(result.steps).toBe(2);
        });

        it('renders an object input as JSON rather than [object Object]', async () => {
            const lm = new MockLM({ responses: [finish(), extract({ answer: 'ok' })] });
            const agent = new ReAct('question -> answer', { tools: {}, lm });

            await agent.forward({ question: { city: 'Paris' } });

            expect(lm.lastPrompt()).toContain('{"city":"Paris"}');
            expect(lm.lastPrompt()).not.toContain('[object Object]');
        });

        it('exposes its step and extract predictors', () => {
            const agent = new ReAct('question -> answer', { lm: new MockLM(), tools: {} });

            expect(agent.react).toBeInstanceOf(Predict);
            expect(agent.extract).toBeInstanceOf(ChainOfThought);
            expect(agent.namedPredictors().map(([path]) => path)).toEqual(['react', 'extract']);
        });

        it('keeps RespAct as a deprecated alias', () => {
            const agent = new RespAct('question -> answer', { lm: new MockLM(), tools: {} });
            expect(agent).toBeInstanceOf(ReAct);
        });
    });

    describe('tool execution', () => {
        it('awaits async tools and returns their result', async () => {
            const tool = vi.fn().mockResolvedValue('Tool result');
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: { testTool: { description: 'Test tool', function: tool } },
            });

            const result = await (agent as any).executeTool('testTool', 'test input', 0);

            expect(tool).toHaveBeenCalledWith('test input');
            expect(result).toBe('Tool result');
        });

        it('stringifies results from synchronous tools', async () => {
            const tool = vi.fn().mockReturnValue('Legacy result');
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: { legacyTool: tool },
            });

            const result = await (agent as any).executeTool('legacyTool', 'test input', 0);

            expect(tool).toHaveBeenCalledWith('test input');
            expect(result).toBe('Legacy result');
        });

        it('turns a thrown tool error into an observation', async () => {
            const agent = new ReAct('question -> answer', {
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
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: { knownTool: { description: 'A known tool', function: () => 'result' } },
            });

            const result = await (agent as any).executeTool('unknownTool', 'input', 0);
            expect(result).toContain("Error: Tool 'unknownTool' not found");
            expect(result).toContain('Available tools: knownTool');
        });
    });

    describe('reasoning loop', () => {
        it('calls a tool, finishes, then extracts the validated answer', async () => {
            const calculator = vi.fn().mockReturnValue(42);
            const lm = new MockLM({
                responses: [
                    step('I need to calculate.', 'calculator', { input: '6 * 7' }),
                    finish(),
                    extract({ answer: 'The result is 42' }, 'The calculator said 42.'),
                ],
            });

            const agent = new ReAct('question -> answer', {
                tools: {
                    calculator: { description: 'Performs arithmetic', function: calculator },
                },
                maxSteps: 3,
                lm,
            });

            const result = await agent.forward({ question: 'What is 6 times 7?' });

            expect(calculator).toHaveBeenCalledWith('6 * 7');
            expect(result.answer).toBe('The result is 42');
            expect(result.reasoning).toBe('The calculator said 42.');
            expect(result.steps).toBe(2);
            expect(result.trajectory).toEqual({
                thought_0: 'I need to calculate.',
                tool_name_0: 'calculator',
                tool_args_0: { input: '6 * 7' },
                observation_0: '42',
                thought_1: 'I have what I need.',
                tool_name_1: 'finish',
                tool_args_1: {},
                observation_1: 'Completed.',
            });
        });

        it('feeds the trajectory to later steps and to the extract step', async () => {
            const lm = new MockLM({
                responses: [
                    step('Look.', 'lookup', { input: 'x' }),
                    finish(),
                    extract({ answer: 'found' }),
                ],
            });
            await new ReAct('question -> answer', {
                lm,
                tools: { lookup: () => 'the observation' },
            }).forward({ question: 'Q' });

            for (const call of lm.calls.slice(1)) {
                expect(String(call.messages.at(-1)?.content)).toContain(
                    'observation_0: the observation'
                );
            }
            // The extract step is a ChainOfThought over the original outputs.
            expect(String(lm.calls[2].messages[0].content)).toContain('`reasoning`');
        });

        it('accepts a bare value for a single-argument tool', async () => {
            const shout = vi.fn(({ text }: { text: string }) => text.toUpperCase());
            const lm = new MockLM({
                responses: [
                    step('Shout.', 'shout', 'hello'),
                    finish(),
                    extract({ answer: 'HELLO' }),
                ],
            });

            await new ReAct('question -> answer', {
                lm,
                tools: {
                    shout: {
                        description: 'Uppercase text',
                        parameters: z.object({ text: z.string() }),
                        function: shout,
                    },
                },
            }).forward({ question: 'Q' });

            expect(shout).toHaveBeenCalledWith({ text: 'hello' });
        });

        it('validates typed output fields in the extract step', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'the answer' })
                answer!: string;

                @OutputField({ description: 'confidence', type: 'number' })
                confidence!: number;
            }

            const lm = new MockLM({
                responses: [finish(), extract({ answer: 'Paris', confidence: '0.9' })],
            });
            const agent = new ReAct(Scored, { tools: {}, maxSteps: 2, lm });

            const result = await agent.forward({ question: 'Capital of France?' });

            expect(result.answer).toBe('Paris');
            expect(result.confidence).toBe(0.9);
        });

        it('does not repeat an identical tool call', async () => {
            const tool = vi.fn().mockReturnValue('data');
            const lm = new MockLM({
                responses: [
                    step('Fetch.', 'fetch', { input: 'same' }),
                    step('Fetch again.', 'fetch', { input: 'same' }),
                    finish(),
                    extract({ answer: 'done' }),
                ],
            });
            const events: ReActEvent[] = [];

            const agent = new ReAct('question -> answer', {
                tools: { fetch: tool },
                maxSteps: 5,
                lm,
                onEvent: (event) => events.push(event),
            });

            const result = await agent.forward({ question: 'Q' });

            expect(tool).toHaveBeenCalledTimes(1);
            expect(result.answer).toBe('done');
            expect(result.trajectory.observation_1).toContain('already made this tool call');
            expect(events.some((event) => event.type === 'repeated_tool_call')).toBe(true);
        });

        it('passes call options through to the language model', async () => {
            const lm = new MockLM({ responses: [finish(), extract({ answer: 'ok' })] });
            const agent = new ReAct('question -> answer', { tools: {}, lm });

            await agent.forward({ question: 'Q' }, { temperature: 0.1, timeout: 5000 });

            expect(lm.calls[0].options).toEqual({ temperature: 0.1, timeout: 5000 });
            expect(lm.calls[1].options).toEqual({ temperature: 0.1, timeout: 5000 });
        });

        it('proceeds to the extract step when the steps run out', async () => {
            const lm = new MockLM({
                responses: [
                    step('Look.', 'lookup', { input: 'a' }),
                    step('Look more.', 'lookup', { input: 'b' }),
                    extract({ answer: 'best guess' }),
                ],
            });
            const agent = new ReAct('question -> answer', {
                tools: { lookup: (input: string) => `saw ${input}` },
                maxSteps: 2,
                lm,
            });

            // DSPy's behaviour: no MaxStepsError, the trajectory so far is used.
            const result = await agent.forward({ question: 'Q' });

            expect(result.answer).toBe('best guess');
            expect(result.steps).toBe(2);
            expect(lm.calls).toHaveLength(3);
            expect(String(lm.calls[2].messages.at(-1)?.content)).toContain(
                'observation_1: saw b'
            );
        });

        it('ends the loop and extracts when a step cannot be parsed', async () => {
            const lm = new MockLM({
                responses: ['I will just think out loud.', extract({ answer: '42' })],
            });
            const events: ReActEvent[] = [];
            const agent = new ReAct('question -> answer', {
                tools: {},
                maxSteps: 3,
                lm,
                // No JSON retry, so the unparseable reply ends the loop at once.
                adapter: new ChatAdapter({ jsonFallback: false }),
                onEvent: (event) => events.push(event),
            });

            const result = await agent.forward({ question: 'Q' });

            expect(result.answer).toBe('42');
            expect(result.steps).toBe(0);
            expect(events.map((event) => event.type)).toEqual(['parse_failed']);
        });

        it('emits events for each stage of the loop', async () => {
            const lm = new MockLM({
                responses: [
                    step('Echo.', 'echo', { input: 'hi' }),
                    finish(),
                    extract({ answer: 'hi' }),
                ],
            });
            const events: ReActEvent[] = [];

            const agent = new ReAct('question -> answer', {
                tools: { echo: (input: string) => input },
                lm,
                onEvent: (event) => events.push(event),
            });

            await agent.forward({ question: 'Q' });

            expect(events.map((event) => event.type)).toEqual([
                'thought',
                'tool_call',
                'tool_result',
                'thought',
            ]);
        });
    });

    describe('typed tools', () => {
        it('derives a JSON Schema from a Zod argument schema', () => {
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: ({ a, b }: { a: number; b: number }) => a + b,
                    },
                },
            });

            const schema = (agent as any).tools.add.tool.parameters;
            expect(schema.type).toBe('object');
            expect(Object.keys(schema.properties)).toEqual(['a', 'b']);
            expect(schema.required).toEqual(['a', 'b']);
        });

        it('synthesizes a single-string schema for an untyped tool', () => {
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: { echo: (input: string) => input },
            });

            const schema = (agent as any).tools.echo.tool.parameters;
            expect(Object.keys(schema.properties)).toEqual(['input']);
            expect((agent as any).tools.echo.typed).toBe(false);
        });

        it('accepts a raw JSON Schema without wrapping it in Zod', () => {
            const parameters = {
                type: 'object',
                properties: { city: { type: 'string' } },
                required: ['city'],
            };
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    weather: { description: 'Weather', parameters, function: () => 'sunny' },
                },
            });

            expect((agent as any).tools.weather.tool.parameters).toBe(parameters);
            expect((agent as any).tools.weather.tool.validator).toBeUndefined();
        });

        it('parses JSON arguments into named arguments', async () => {
            const add = vi.fn(({ a, b }: { a: number; b: number }) => a + b);
            const agent = new ReAct('question -> answer', {
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

        it('turns a Zod validation failure into a correctable observation', async () => {
            const add = vi.fn();
            const events: ReActEvent[] = [];
            const agent = new ReAct('question -> answer', {
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

        it('coerces a bare value to the declared scalar type', async () => {
            const double = vi.fn(({ n }: { n: number }) => n * 2);
            const agent = new ReAct('question -> answer', {
                lm: new MockLM(),
                tools: {
                    double: {
                        description: 'Double a number',
                        parameters: z.object({ n: z.number() }),
                        function: double,
                    },
                },
            });

            expect(await (agent as any).executeTool('double', '21', 0)).toBe('42');
            expect(double).toHaveBeenCalledWith({ n: 21 });
        });

        it('lists each tool’s arguments in the step instructions', async () => {
            const lm = new MockLM({ responses: [finish(), extract({ answer: 'x' })] });
            await new ReAct('question -> answer', {
                lm,
                tools: {
                    add: {
                        description: 'Add two numbers',
                        parameters: z.object({ a: z.number(), b: z.number() }),
                        function: () => 0,
                    },
                },
            }).forward({ question: 'Q' });

            expect(String(lm.calls[0].messages[0].content)).toContain(
                'It takes arguments {"a":{"type":"number"},"b":{"type":"number"}}'
            );
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

            const agent = new ReAct('question -> answer', {
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
            expect(result.trajectory).toMatchObject({
                tool_name_0: 'add',
                tool_args_0: { a: 6, b: 7 },
                observation_0: '13',
            });
        });

        it('opens the conversation with the adapter’s messages for the signature', async () => {
            const lm = new ToolCallingLM([{ content: 'done', finishReason: 'stop' }]);
            await new ReAct('question -> answer', {
                lm,
                tools: { echo: (input: string) => input },
            }).forward({ question: 'Q' });

            const [system, user] = lm.turns[0];
            expect(system.role).toBe('system');
            expect(String(system.content)).toContain('[[ ## answer ## ]]');
            expect(String(system.content)).toContain('You have tools available');
            expect(user.role).toBe('user');
            expect(String(user.content)).toContain('[[ ## question ## ]]\nQ');
        });

        it('advertises every tool schema on each request, without finish', async () => {
            const lm = new ToolCallingLM([{ content: 'done', finishReason: 'stop' }]);
            const agent = new ReAct('question -> answer', {
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
            const agent = new ReAct('question -> answer', {
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
            const agent = new ReAct('question -> answer', { lm, tools: { echo } });

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

            const agent = new ReAct('question -> answer', { lm, tools: { left, right } });
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
            const events: ReActEvent[] = [];

            const agent = new ReAct('question -> answer', {
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

        it('emits the same event surface as the adapter loop', async () => {
            const lm = new ToolCallingLM([
                {
                    content: 'thinking',
                    toolCalls: [{ id: 'c1', name: 'echo', arguments: { input: 'hi' } }],
                },
                { content: 'hi', finishReason: 'stop' },
            ]);
            const events: ReActEvent[] = [];

            const agent = new ReAct('question -> answer', {
                lm,
                tools: { echo: (input: string) => input },
                onEvent: (event) => events.push(event),
            });

            await agent.forward({ question: 'Q' });

            expect(events.map((event) => event.type)).toEqual([
                'thought',
                'tool_call',
                'tool_result',
                'thought',
            ]);
            expect(events[1]).toMatchObject({ tool: 'echo', input: 'hi' });
        });

        it('hands a final answer that misses a field to the extract step', async () => {
            class Scored extends Signature {
                @OutputField({ description: 'the answer' })
                answer!: string;

                @OutputField({ description: 'confidence', type: 'number' })
                confidence!: number;
            }

            const lm = new ToolCallingLM([
                { content: 'answer: Paris', finishReason: 'stop' },
                {
                    content: 'reasoning: it said Paris\nanswer: Paris\nconfidence: 0.8',
                    finishReason: 'stop',
                },
            ]);
            const events: ReActEvent[] = [];
            const agent = new ReAct(Scored, {
                lm,
                tools: { echo: (input: string) => input },
                onEvent: (event) => events.push(event),
            });

            const result = await agent.forward({ question: 'Capital of France?' });

            expect(result.confidence).toBe(0.8);
            expect(events.some((event) => event.type === 'parse_failed')).toBe(true);
            // The extract call carries the native exchange as its trajectory.
            expect(String(lm.turns[1].at(-1)?.content)).toContain('thought_0: answer: Paris');
            expect(lm.options[1]?.tools).toBeUndefined();
        });

        it('extracts from the trajectory when the native steps run out', async () => {
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [{ id: 'c1', name: 'echo', arguments: { input: 'a' } }],
                },
                { content: 'reasoning: r\nanswer: a', finishReason: 'stop' },
            ]);
            const agent = new ReAct('question -> answer', {
                lm,
                maxSteps: 1,
                tools: { echo: (input: string) => input },
            });

            const result = await agent.forward({ question: 'Q' });

            expect(result.answer).toBe('a');
            expect(result.steps).toBe(1);
        });

        it('surfaces a failing tool as an observation rather than throwing', async () => {
            const lm = new ToolCallingLM([
                {
                    content: '',
                    toolCalls: [{ id: 'c1', name: 'boom', arguments: { input: 'x' } }],
                },
                { content: 'recovered', finishReason: 'stop' },
            ]);
            const agent = new ReAct('question -> answer', {
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

        it('traces content-part messages as text rather than [object Object]', async () => {
            configure({ tracing: true });
            clearHistory();
            try {
                const lm = new ToolCallingLM([
                    {
                        content: '',
                        toolCalls: [{ id: 'c1', name: 'echo', arguments: { input: 'hi' } }],
                    },
                    { content: 'answer: hi', finishReason: 'stop' },
                ]);
                // The loop reuses one message array across turns; swap the user
                // turn for content parts, as a multimodal conversation would carry.
                const original = lm.chatWithTools.bind(lm);
                lm.chatWithTools = async (messages, options) => {
                    messages[1] = {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'part one, ' },
                            { type: 'text', text: 'part two' },
                        ],
                    };
                    return original(messages, options);
                };
                const agent = new ReAct('question -> answer', {
                    lm,
                    tools: { echo: (input: string) => input },
                });

                const result = await agent.forward({ question: 'Q' });

                const calls: LMCall[] = result.trace?.calls ?? [];
                const prompts = calls.map((call) => call.prompt);
                expect(prompts).toHaveLength(2);
                expect(prompts[1]).toContain('user: part one, part two');
                expect(prompts.join('\n')).not.toContain('[object Object]');
                expect(inspectHistory().length).toBeGreaterThan(0);
            } finally {
                configure({ tracing: false });
                clearHistory();
            }
        });

        it('uses the adapter loop when the model cannot call tools', async () => {
            const add = vi.fn(({ a, b }: { a: number; b: number }) => a + b);
            const lm = new MockLM({
                responses: [
                    step('Add.', 'add', { a: 6, b: 7 }),
                    finish(),
                    extract({ answer: 'The result is 13' }),
                ],
            });

            const agent = new ReAct('question -> answer', {
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

        it('stays on the adapter loop when the model inherited chatWithTools', async () => {
            // The flag existed long before anything read it, so a custom LM can
            // report it while inheriting BaseLM's text-only `chatWithTools`.
            // Taking the native path there would silently never call a tool.
            const lm = new MockLM({
                capabilities: { supportsFunctionCalling: true },
                responses: [
                    step('Echo.', 'echo', { input: 'hi' }),
                    finish(),
                    extract({ answer: 'hi' }),
                ],
            });
            const echo = vi.fn((input: string) => input);
            const agent = new ReAct('question -> answer', { lm, tools: { echo } });

            const result = await agent.forward({ question: 'Q' });

            expect(echo).toHaveBeenCalledWith('hi');
            expect(result.answer).toBe('hi');
        });

        it('does not record an empty assistant turn when the model says nothing', async () => {
            const lm = new ToolCallingLM([
                { content: '', finishReason: 'stop' },
                { content: 'done', finishReason: 'stop' },
            ]);
            const agent = new ReAct('question -> answer', {
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
            const agent = new ReAct('question -> answer', { lm, tools: { echo } });

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
            const agent = new ReAct('question -> answer', { lm, tools: { echo } });

            await agent.forward({ question: 'Q' });

            expect(echo).not.toHaveBeenCalled();
            expect(lm.turns[1].at(-1)?.content).toContain('Expected a single string argument');
        });

        it('honours forceTextMode on a tool-calling model', async () => {
            const lm = new ToolCallingLM([
                { content: finish(), finishReason: 'stop' },
                { content: extract({ answer: 'done' }), finishReason: 'stop' },
            ]);
            const agent = new ReAct('question -> answer', {
                lm,
                forceTextMode: true,
                tools: { echo: (input: string) => input },
            });

            const result = await agent.forward({ question: 'Q' });

            expect(result.answer).toBe('done');
            // The adapter loop sends system + user turns and declares no tools.
            expect(lm.turns[0].map((message) => message.role)).toEqual(['system', 'user']);
            expect(lm.options[0]?.tools).toBeUndefined();
        });
    });
});
