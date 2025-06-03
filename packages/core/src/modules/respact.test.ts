import { RespAct, ToolFunction, ToolWithDescription } from './respact';
import { Signature } from '../core/signature';

// Mock LM for testing
class MockLM {
    private responses: string[] = [];
    private responseIndex = 0;

    setResponses(responses: string[]) {
        this.responses = responses;
        this.responseIndex = 0;
    }

    async generate(prompt: string): Promise<string> {
        if (this.responseIndex >= this.responses.length) {
            throw new Error('No more mock responses available');
        }
        return this.responses[this.responseIndex++];
    }
}

// Mock module configuration
jest.mock('../core/module', () => {
    return {
        Module: class {
            protected signature: any;
            protected lm: MockLM;

            constructor(signature: any) {
                this.signature = signature;
                this.lm = new MockLM();
            }

            protected parseOutput(rawOutput: any): Record<string, any> {
                if (typeof rawOutput === 'string') {
                    // Simple parsing for test
                    if (rawOutput.includes(':')) {
                        const parts = rawOutput.split(':');
                        return { [parts[0].trim()]: parts[1].trim() };
                    }
                }
                return { answer: rawOutput };
            }
        }
    };
});

describe('RespAct', () => {
    let mockLM: MockLM;

    beforeEach(() => {
        mockLM = new MockLM();
    });

    describe('Constructor and Tool Handling', () => {
        it('should accept legacy tools (functions only)', () => {
            const mockTool: ToolFunction = (input: string) => `Result: ${input}`;

            const agent = new RespAct('question -> answer', {
                tools: {
                    testTool: mockTool
                }
            });

            expect(agent).toBeInstanceOf(RespAct);
            // Access private field for testing
            const tools = (agent as any).tools;
            expect(tools.testTool).toBeDefined();
            expect(tools.testTool.description).toBe('Tool: testTool');
            expect(tools.testTool.function).toBe(mockTool);
        });

        it('should accept new tools with descriptions', () => {
            const mockTool: ToolWithDescription = {
                description: 'A test tool that processes input',
                function: (input: string) => `Processed: ${input}`
            };

            const agent = new RespAct('question -> answer', {
                tools: {
                    advancedTool: mockTool
                }
            });

            const tools = (agent as any).tools;
            expect(tools.advancedTool).toBeDefined();
            expect(tools.advancedTool.description).toBe('A test tool that processes input');
            expect(tools.advancedTool.function).toBe(mockTool.function);
        });

        it('should handle mixed tool types', () => {
            const legacyTool: ToolFunction = (input: string) => `Legacy: ${input}`;
            const newTool: ToolWithDescription = {
                description: 'Modern tool with description',
                function: (input: string) => `Modern: ${input}`
            };

            const agent = new RespAct('question -> answer', {
                tools: {
                    legacy: legacyTool,
                    modern: newTool
                }
            });

            const tools = (agent as any).tools;
            expect(tools.legacy.description).toBe('Tool: legacy');
            expect(tools.modern.description).toBe('Modern tool with description');
        });
    });

    describe('Prompt Building with Tool Descriptions', () => {
        it('should include tool descriptions in the prompt', () => {
            const tools = {
                calculate: {
                    description: 'Performs mathematical calculations',
                    function: (expr: string) => eval(expr)
                } as ToolWithDescription,
                search: {
                    description: 'Searches for information online',
                    function: (query: string) => `Search results for: ${query}`
                } as ToolWithDescription
            };

            const agent = new RespAct('question -> answer', { tools });
            const prompt = (agent as any).buildInitialPrompt({ question: 'Test question' });

            expect(prompt).toContain('- calculate: Performs mathematical calculations');
            expect(prompt).toContain('- search: Searches for information online');
            expect(prompt).toContain('Question: Test question');
        });

        it('should include legacy tool descriptions in the prompt', () => {
            const legacyTool: ToolFunction = (input: string) => `Result: ${input}`;

            const agent = new RespAct('question -> answer', {
                tools: { oldTool: legacyTool }
            });

            const prompt = (agent as any).buildInitialPrompt({ question: 'Test question' });
            expect(prompt).toContain('- oldTool: Tool: oldTool');
        });
    });

    describe('Tool Execution', () => {
        it('should execute tools with descriptions correctly', async () => {
            const mockTool = jest.fn().mockResolvedValue('Tool result');
            const agent = new RespAct('question -> answer', {
                tools: {
                    testTool: {
                        description: 'Test tool description',
                        function: mockTool
                    }
                }
            });

            const result = await (agent as any).executeTool('testTool', 'test input');

            expect(mockTool).toHaveBeenCalledWith('test input');
            expect(result).toBe('Tool result');
        });

        it('should execute legacy tools correctly', async () => {
            const mockTool = jest.fn().mockReturnValue('Legacy result');
            const agent = new RespAct('question -> answer', {
                tools: { legacyTool: mockTool }
            });

            const result = await (agent as any).executeTool('legacyTool', 'test input');

            expect(mockTool).toHaveBeenCalledWith('test input');
            expect(result).toBe('Legacy result');
        });

        it('should handle tool errors gracefully', async () => {
            const failingTool = () => {
                throw new Error('Tool failed');
            };

            const agent = new RespAct('question -> answer', {
                tools: {
                    failingTool: {
                        description: 'A tool that fails',
                        function: failingTool
                    }
                }
            });

            const result = await (agent as any).executeTool('failingTool', 'input');
            expect(result).toContain('Error executing failingTool: Error: Tool failed');
        });

        it('should handle unknown tools', async () => {
            const agent = new RespAct('question -> answer', {
                tools: {
                    knownTool: {
                        description: 'A known tool',
                        function: () => 'result'
                    }
                }
            });

            const result = await (agent as any).executeTool('unknownTool', 'input');
            expect(result).toContain("Error: Tool 'unknownTool' not found");
            expect(result).toContain('Available tools: knownTool');
        });
    });

    describe('Integration Tests', () => {
        it('should complete a full workflow with described tools', async () => {
            const calculatorTool = {
                description: 'Performs arithmetic calculations like addition, multiplication, etc.',
                function: jest.fn().mockReturnValue(42)
            };

            const agent = new RespAct('question -> answer', {
                tools: { calculator: calculatorTool },
                maxSteps: 3
            });

            // Mock the LM responses
            const mockLM = (agent as any).lm;
            mockLM.setResponses([
                'I need to calculate the result. Action: calculator\nAction Input: 6 * 7',
                'Final Answer: The result is 42'
            ]);

            const result = await agent.forward({ question: 'What is 6 times 7?' });

            expect(calculatorTool.function).toHaveBeenCalledWith('6 * 7');
            expect(result.answer).toBe('The result is 42');
            expect(result.steps).toBe(2);
        });

        it('should work with legacy tools in integration', async () => {
            const legacyTool = jest.fn().mockReturnValue('Legacy result');

            const agent = new RespAct('question -> answer', {
                tools: { legacy: legacyTool },
                maxSteps: 3
            });

            const mockLM = (agent as any).lm;
            mockLM.setResponses([
                'Action: legacy\nAction Input: test',
                'Final Answer: Got legacy result'
            ]);

            const result = await agent.forward({ question: 'Test legacy tools' });

            expect(legacyTool).toHaveBeenCalledWith('test');
            expect(result.answer).toBe('Got legacy result');
        });
    });

    describe('Backward Compatibility', () => {
        it('should maintain full backward compatibility with existing code', () => {
            // This test ensures that existing code using the old format still works
            const oldStyleTools = {
                tool1: (input: string) => `Result 1: ${input}`,
                tool2: (input: string) => `Result 2: ${input}`
            };

            expect(() => {
                new RespAct('question -> answer', {
                    tools: oldStyleTools
                });
            }).not.toThrow();
        });

        it('should generate appropriate descriptions for legacy tools', () => {
            const legacyTools = {
                calculate: (expr: string) => eval(expr),
                fetchData: async (url: string) => `Data from ${url}`
            };

            const agent = new RespAct('question -> answer', {
                tools: legacyTools
            });

            const tools = (agent as any).tools;
            expect(tools.calculate.description).toBe('Tool: calculate');
            expect(tools.fetchData.description).toBe('Tool: fetchData');
        });
    });
}); 