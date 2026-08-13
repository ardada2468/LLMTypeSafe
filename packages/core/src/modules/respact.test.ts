import {
    RespAct,
    type ToolFunction,
    type ToolWithDescription,
    type RespActEvent,
} from './respact';
import { Signature, OutputField, InputField } from '../core/signature';
import { MockLM } from '../test-utils';

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
});
