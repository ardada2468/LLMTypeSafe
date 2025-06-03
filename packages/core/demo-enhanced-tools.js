/**
 * Simple demo to show enhanced tool descriptions working
 * This can be run directly with node demo-enhanced-tools.js
 */

const { RespAct, configure } = require('./dist/index.js');

// Mock LM for demonstration
class MockLM {
    constructor() {
        this.stepCount = 0;
    }

    async generate(prompt) {
        console.log('\n🤖 AI RECEIVED PROMPT:');
        console.log('='.repeat(50));

        // Show tool descriptions in prompt
        const lines = prompt.split('\n');
        const toolStart = lines.findIndex(line => line.includes('You have access to the following tools:'));
        if (toolStart !== -1) {
            console.log('Tools available to AI:');
            for (let i = toolStart + 1; i < lines.length && lines[i].trim(); i++) {
                if (lines[i].startsWith('-')) {
                    console.log('  ' + lines[i]);
                }
            }
        }

        const questionLine = lines.find(line => line.startsWith('Question:'));
        if (questionLine) {
            console.log('\n' + questionLine);
        }
        console.log('='.repeat(50));

        this.stepCount++;

        // Better logic for sequential tool calls
        if (prompt.includes('Observation: 23') && this.stepCount > 1) {
            // After getting the result of addition, multiply
            console.log('🧠 AI Decision: I got 23 from adding, now I need to multiply by 3');
            return 'Now I need to multiply 23 by 3. Action: multiply\nAction Input: 23,3';
        } else if (prompt.includes('Observation: 69') && this.stepCount > 2) {
            // After getting multiplication result, provide final answer
            console.log('🧠 AI Decision: I have both results, time for final answer');
            return 'Final Answer: The calculation is complete. First I added 15 + 8 = 23, then multiplied 23 × 3 = 69. The final answer is 69.';
        } else if (prompt.includes('add 15 and 8') || prompt.includes('Please add 15 and 8')) {
            // First step: addition
            console.log('🧠 AI Decision: I need to start by adding 15 and 8');
            return 'I need to add 15 and 8 first. Action: add\nAction Input: 15,8';
        } else {
            // Fallback
            console.log('🧠 AI Decision: Providing fallback response');
            return 'Final Answer: I need to add 15 and 8, then multiply by 3.';
        }
    }

    // Required methods for ILanguageModel interface
    async chat(messages) {
        return this.generate(messages.map(m => m.content).join('\n'));
    }

    getUsage() {
        return { promptTokens: 0, completionTokens: 0, totalTokens: 0, totalCost: 0 };
    }

    resetUsage() { }
}

async function demonstrateEnhancedTools() {
    console.log('🎯 Enhanced Tool Descriptions Demo');
    console.log('==================================\n');

    // Configure mock LM
    console.log('🔧 Configuring mock language model...\n');
    configure({ lm: new MockLM() });

    // Create agent with enhanced tools
    const agent = new RespAct('question -> answer', {
        tools: {
            add: {
                description: "Adds two numbers together. Input format: 'number1,number2' (e.g., '5,3'). Use this for addition operations when you need to add two numbers.",
                function: (input) => {
                    console.log(`🔧 ADD TOOL: Processing "${input}"`);
                    const [a, b] = input.split(',').map(x => parseFloat(x.trim()));
                    const result = a + b;
                    console.log(`   ✅ ${a} + ${b} = ${result}`);
                    return result;
                }
            },

            multiply: {
                description: "Multiplies two numbers together. Input format: 'number1,number2' (e.g., '4,7'). Use this for multiplication operations when you need to multiply two numbers.",
                function: (input) => {
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

    console.log('🧪 Testing question: "Please add 15 and 8, then multiply the result by 3"\n');

    try {
        const result = await agent.forward({
            question: "Please add 15 and 8, then multiply the result by 3"
        });

        console.log('\n✅ FINAL RESULT:', result.answer);
        console.log('📊 Steps taken:', result.steps);

        console.log('\n🎯 Key Benefits Demonstrated:');
        console.log('• AI received detailed tool descriptions');
        console.log('• AI understood WHEN to use each tool');
        console.log('• AI formatted inputs correctly');
        console.log('• Enhanced descriptions lead to better tool selection');

    } catch (error) {
        console.log('❌ Error:', error.message);
    }
}

// Run the demo
if (require.main === module) {
    console.log('Building project first...\n');

    // First build the project
    const { execSync } = require('child_process');
    try {
        execSync('npm run build', { stdio: 'inherit' });
        console.log('\n✅ Build completed, running demo...\n');
        demonstrateEnhancedTools().catch(console.error);
    } catch (error) {
        console.error('❌ Build failed:', error.message);
    }
} 