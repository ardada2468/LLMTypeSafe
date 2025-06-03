/**
 * Test script for enhanced tool descriptions
 * This demonstrates the new tool format without requiring API keys
 */

import { RespAct } from '../packages/core/src/modules/respact';

// Mock LM that simulates intelligent tool usage based on descriptions
class MockLM {
    async generate(prompt: string): Promise<string> {
        console.log('\n🤖 AI Received Prompt:');
        console.log('='.repeat(40));

        // Extract tool descriptions from prompt
        const toolLines = prompt.split('\n').filter(line => line.startsWith('- '));
        console.log('Available tools:');
        toolLines.forEach(line => console.log('  ' + line));
        console.log('='.repeat(40));

        // Simulate intelligent responses based on the question
        if (prompt.includes('15 plus 8') && prompt.includes('multiply')) {
            return 'I need to add 15 and 8 first. Action: add\nAction Input: 15,8';
        } else if (prompt.includes('23') && prompt.includes('multiply')) {
            return 'Now I need to multiply 23 by 3. Action: multiply\nAction Input: 23,3';
        } else if (prompt.includes('stock price') && prompt.includes('AAPL')) {
            return 'I need to get the stock price. Action: getStockPrice\nAction Input: AAPL';
        } else if (prompt.includes('recommendation')) {
            return 'I need to get a recommendation. Action: getRecommendation\nAction Input: AAPL';
        } else {
            return 'Final Answer: I have completed the requested calculations.';
        }
    }
}

// Mock Module class
class MockModule {
    protected signature: any;
    protected lm: MockLM;

    constructor(signature: any) {
        this.signature = signature;
        this.lm = new MockLM();
    }

    protected parseOutput(rawOutput: any): Record<string, any> {
        if (typeof rawOutput === 'string') {
            return { answer: rawOutput };
        }
        return { answer: rawOutput };
    }
}

// Temporarily override Module for testing
const originalModule = require('../packages/core/src/core/module').Module;
require('../packages/core/src/core/module').Module = MockModule;

async function testEnhancedTools() {
    console.log('🎯 Testing Enhanced Tool Descriptions');
    console.log('=====================================\n');

    // Test 1: Simple Math Tools
    console.log('📝 Test 1: Simple Math with Enhanced Descriptions');
    console.log('-'.repeat(50));

    const mathAgent = new RespAct('question -> answer', {
        tools: {
            add: {
                description: "Adds two numbers together. Input format: 'number1,number2' (e.g., '5,3'). Use this for addition operations when you need to add two numbers.",
                function: (input: string) => {
                    console.log(`🔧 ADD TOOL: Processing "${input}"`);
                    const [a, b] = input.split(',').map(x => parseFloat(x.trim()));
                    const result = a + b;
                    console.log(`   ✅ ${a} + ${b} = ${result}`);
                    return result;
                }
            },

            multiply: {
                description: "Multiplies two numbers together. Input format: 'number1,number2' (e.g., '4,7'). Use this for multiplication operations when you need to multiply two numbers.",
                function: (input: string) => {
                    console.log(`🔧 MULTIPLY TOOL: Processing "${input}"`);
                    const [a, b] = input.split(',').map(x => parseFloat(x.trim()));
                    const result = a * b;
                    console.log(`   ✅ ${a} × ${b} = ${result}`);
                    return result;
                }
            }
        },
        maxSteps: 3
    });

    try {
        const result = await mathAgent.forward({
            question: "What is 15 plus 8, then multiply that result by 3?"
        });
        console.log('\n✅ Final Result:', result.answer);
        console.log('📊 Steps taken:', result.steps);
    } catch (error: any) {
        console.log('❌ Error:', error.message);
    }

    // Test 2: Compare with Legacy Format
    console.log('\n\n📝 Test 2: Comparison with Legacy Format');
    console.log('-'.repeat(50));

    const legacyAgent = new RespAct('question -> answer', {
        tools: {
            // Legacy format - just functions
            add: (input: string) => {
                console.log(`🔧 LEGACY ADD: ${input}`);
                const [a, b] = input.split(',').map(x => parseFloat(x.trim()));
                return a + b;
            },
            multiply: (input: string) => {
                console.log(`🔧 LEGACY MULTIPLY: ${input}`);
                const [a, b] = input.split(',').map(x => parseFloat(x.trim()));
                return a * b;
            }
        },
        maxSteps: 3
    });

    console.log('\n🔍 Notice: Legacy tools automatically get generic descriptions');
    const legacyPrompt = (legacyAgent as any).buildInitialPrompt({ question: 'Test' });
    const legacyToolSection = legacyPrompt.split('\n\nQuestion:')[0];
    console.log(legacyToolSection);

    console.log('\n🎯 Key Benefits Demonstrated:');
    console.log('• Enhanced tools have detailed descriptions');
    console.log('• AI understands WHEN to use each tool');
    console.log('• AI understands HOW to format inputs');
    console.log('• Legacy tools are automatically supported');
    console.log('• Better tool selection and usage');
}

// Restore original module
function cleanup() {
    require('../packages/core/src/core/module').Module = originalModule;
}

// Run the test
if (require.main === module) {
    testEnhancedTools()
        .then(() => {
            console.log('\n✨ Enhanced Tools Test Completed!');
            cleanup();
        })
        .catch((error) => {
            console.error('❌ Test failed:', error);
            cleanup();
        });
}

export { testEnhancedTools }; 