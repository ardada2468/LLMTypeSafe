/**
 * Test script for structured signatures with enhanced tools
 */

const { RespAct, configure } = require('./dist/index.js');

// Mock LM that understands structured output requirements
class StructuredMockLM {
    async generate(prompt) {
        console.log('\n🤖 AI PROCESSING STRUCTURED REQUEST...\n');

        // Show tool descriptions
        const lines = prompt.split('\n');
        const toolStart = lines.findIndex(line => line.includes('You have access to the following tools:'));
        if (toolStart !== -1) {
            console.log('📋 TOOLS AVAILABLE:');
            for (let i = toolStart + 1; i < lines.length; i++) {
                if (lines[i].startsWith('- ')) {
                    console.log('  ' + lines[i]);
                } else if (lines[i].trim() === '' || lines[i].startsWith('Question:')) {
                    break;
                }
            }
        }

        // Show output format requirements
        const formatStart = lines.findIndex(line => line.includes('IMPORTANT: When providing your Final Answer'));
        if (formatStart !== -1) {
            console.log('\n📋 REQUIRED OUTPUT FORMAT:');
            for (let i = formatStart; i < lines.length; i++) {
                if (lines[i].includes(':') && !lines[i].includes('Final Answer:')) {
                    console.log('  ' + lines[i].trim());
                } else if (lines[i].includes('Begin!')) {
                    break;
                }
            }
        }

        console.log('='.repeat(50));

        // Simulate intelligent tool usage
        if (prompt.includes('symbol') && prompt.includes('AAPL') && !prompt.includes('Observation:')) {
            console.log('🧠 Decision: I need to analyze the stock to get all required information');
            return 'I need to analyze AAPL stock. Action: analyzeStock\nAction Input: AAPL';
        } else if (prompt.includes('Observation:') && prompt.includes('symbol: AAPL')) {
            console.log('🧠 Decision: I have all the information, providing structured final answer');
            return 'Final Answer:\nsymbol: AAPL\nprice: 150.25\nrecommendation: Buy';
        } else {
            console.log('🧠 Decision: Providing fallback response');
            return 'Final Answer:\nsymbol: UNKNOWN\nprice: 0\nrecommendation: Hold';
        }
    }

    async chat(messages) { return this.generate(messages.map(m => m.content).join('\n')); }
    getUsage() { return { promptTokens: 0, completionTokens: 0, totalTokens: 0, totalCost: 0 }; }
    resetUsage() { }
}

async function testStructuredSignature() {
    console.log('🎯 Testing Structured Signatures with Enhanced Tools');
    console.log('====================================================\n');

    // Configure mock LM
    configure({ lm: new StructuredMockLM() });

    // Create a simple signature class for testing
    class SimpleStockAnalysis {
        static getOutputFields() {
            return {
                symbol: { type: 'string' },
                price: { type: 'number' },
                recommendation: { type: 'string' }
            };
        }

        static description = "Analyze a stock and provide symbol, price, and recommendation";
    }

    // Create agent with enhanced tool
    const analyst = new RespAct(SimpleStockAnalysis, {
        tools: {
            analyzeStock: {
                description: "Analyzes a stock symbol and returns comprehensive data including current price and investment recommendation. Input: stock symbol (e.g., 'AAPL'). Output: structured analysis data.",
                function: (symbol) => {
                    console.log(`\n🔧 STOCK ANALYSIS TOOL:`);
                    console.log(`   Input: "${symbol}"`);

                    const mockData = {
                        'AAPL': { price: 150.25, recommendation: 'Buy' },
                        'TSLA': { price: 245.80, recommendation: 'Hold' },
                        'MSFT': { price: 320.15, recommendation: 'Buy' }
                    };

                    const data = mockData[symbol.toUpperCase()] || { price: 100, recommendation: 'Hold' };

                    console.log(`   Processing: ${symbol.toUpperCase()}`);
                    console.log(`   Price: $${data.price}`);
                    console.log(`   Recommendation: ${data.recommendation}`);

                    // Return in format that can be parsed
                    const result = `symbol: ${symbol.toUpperCase()}\nprice: ${data.price}\nrecommendation: ${data.recommendation}`;
                    console.log(`   📤 Output: ${result.replace(/\n/g, ', ')}`);

                    return result;
                }
            }
        },
        maxSteps: 3
    });

    console.log('🧪 Test Case: Analyzing AAPL stock\n');

    try {
        const result = await analyst.forward({
            symbol: 'AAPL'
        });

        console.log('\n✅ RESULTS:');
        console.log('==================');
        console.log('Symbol:', result.symbol);
        console.log('Price:', result.price);
        console.log('Recommendation:', result.recommendation);
        console.log('Steps:', result.steps);

        console.log('\n🔍 Raw Result Object:');
        console.log(JSON.stringify(result, null, 2));

        // Validate results
        const hasSymbol = result.symbol && result.symbol !== 'Not provided';
        const hasPrice = result.price && !isNaN(result.price);
        const hasRecommendation = result.recommendation && result.recommendation !== 'Not provided';

        console.log('\n📊 VALIDATION:');
        console.log('Symbol field:', hasSymbol ? '✅ Present' : '❌ Missing');
        console.log('Price field:', hasPrice ? '✅ Present' : '❌ Missing');
        console.log('Recommendation field:', hasRecommendation ? '✅ Present' : '❌ Missing');

        if (hasSymbol && hasPrice && hasRecommendation) {
            console.log('\n🎉 SUCCESS: Structured signature with enhanced tools working perfectly!');
        } else {
            console.log('\n⚠️  PARTIAL SUCCESS: Some fields missing, but enhanced tools are working');
        }

    } catch (error) {
        console.log('❌ ERROR:', error.message);
    }
}

// Build and run
console.log('Building project...\n');
const { execSync } = require('child_process');

try {
    execSync('npm run build:types', { stdio: 'inherit' });
    testStructuredSignature().catch(console.error);
} catch (error) {
    console.error('Build failed:', error.message);
} 