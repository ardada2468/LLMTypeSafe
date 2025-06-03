/**
 * Simple demo to show enhanced tool descriptions in action
 */

const { RespAct, configure } = require('./dist/index.js');

// Mock LM that demonstrates tool description usage
class MockLM {
    async generate(prompt) {
        console.log('\n🤖 MOCK AI PROCESSING...');

        // Extract and display tool descriptions
        const lines = prompt.split('\n');
        const toolStart = lines.findIndex(line => line.includes('You have access to the following tools:'));

        if (toolStart !== -1) {
            console.log('\n📋 TOOL DESCRIPTIONS RECEIVED BY AI:');
            console.log('='.repeat(60));
            for (let i = toolStart + 1; i < lines.length; i++) {
                if (lines[i].startsWith('- ')) {
                    console.log('  ' + lines[i]);
                } else if (lines[i].trim() === '' || lines[i].startsWith('Question:')) {
                    break;
                }
            }
            console.log('='.repeat(60));
        }

        // Simple demonstration of tool usage
        if (prompt.includes('calculate 2 + 3')) {
            console.log('🧠 AI Decision: I need to use the math tool to add 2 + 3');
            return 'I need to calculate this. Action: math\nAction Input: 2 + 3';
        } else if (prompt.includes('Observation: 5')) {
            console.log('🧠 AI Decision: I got the result, providing final answer');
            return 'Final Answer: The result of 2 + 3 is 5.';
        } else {
            console.log('🧠 AI Decision: Providing direct response');
            return 'Final Answer: I can help with calculations using the available tools.';
        }
    }

    async chat(messages) { return this.generate(messages.map(m => m.content).join('\n')); }
    getUsage() { return { promptTokens: 0, completionTokens: 0, totalTokens: 0, totalCost: 0 }; }
    resetUsage() { }
}

async function runDemo() {
    console.log('🎯 Enhanced Tool Descriptions Demo');
    console.log('====================================\n');

    // Configure the mock LM
    configure({ lm: new MockLM() });

    // Create agent with enhanced tool descriptions
    const agent = new RespAct('question -> answer', {
        tools: {
            math: {
                description: "Performs mathematical calculations. Provide a mathematical expression like '2 + 3' or '10 * 4'. Use this tool whenever you need to calculate numbers, solve arithmetic problems, or perform mathematical operations.",
                function: (expression) => {
                    console.log(`\n🔧 MATH TOOL EXECUTED:`);
                    console.log(`   Input: "${expression}"`);

                    try {
                        const result = eval(expression.replace(/[^0-9+\-*/().\s]/g, ''));
                        console.log(`   Output: ${result}`);
                        return result;
                    } catch (error) {
                        console.log(`   Error: ${error.message}`);
                        return `Error: ${error.message}`;
                    }
                }
            },

            // Legacy tool for comparison
            oldTool: (input) => {
                console.log(`🔧 LEGACY TOOL: ${input}`);
                return "Legacy result";
            }
        },
        maxSteps: 3
    });

    console.log('🧪 Testing: "Please calculate 2 + 3"\n');

    try {
        const result = await agent.forward({
            question: "Please calculate 2 + 3"
        });

        console.log('\n✅ SUCCESS!');
        console.log('📊 Result:', result.answer);
        console.log('📈 Steps:', result.steps);

        console.log('\n🎯 WHAT WE DEMONSTRATED:');
        console.log('✨ Enhanced tools include detailed descriptions');
        console.log('✨ AI receives these descriptions in the prompt');
        console.log('✨ AI makes intelligent tool selection decisions');
        console.log('✨ Legacy tools work seamlessly alongside enhanced ones');
        console.log('✨ Better tool usage leads to more accurate results');

    } catch (error) {
        console.log('❌ Error:', error.message);
    }
}

// Build and run
console.log('Building project...\n');
const { execSync } = require('child_process');

try {
    execSync('npm run build:types', { stdio: 'inherit' });
    runDemo().catch(console.error);
} catch (error) {
    console.error('Build failed:', error.message);
} 