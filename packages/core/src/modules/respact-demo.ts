/**
 * Demo script to showcase the enhanced tool descriptions feature
 * This runs without requiring API keys by using mock components
 */

import { RespAct } from './respact';

// Mock LM for demonstration
class MockLM {
    async generate(prompt: string): Promise<string> {
        console.log('\n🤖 AI PROMPT RECEIVED:');
        console.log('='.repeat(50));
        console.log(prompt);
        console.log('='.repeat(50));

        // Simulate AI response that uses tools intelligently based on descriptions
        if (prompt.includes('stock price')) {
            return "I need to get the current stock price. Action: fetchStockPrice\nAction Input: AAPL";
        } else if (prompt.includes('calculate')) {
            return "I need to perform a calculation. Action: calculate\nAction Input: 150 * 150.25";
        } else if (prompt.includes('convert')) {
            return "I need to convert currency. Action: convertCurrency\nAction Input: 22537.5, USD, EUR";
        } else {
            return "Final Answer: Based on the available tools and their descriptions, I can help with calculations, stock prices, and currency conversions.";
        }
    }
}

// Mock module for demonstration
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

// Temporarily override the Module import for demo
const OriginalModule = require('../core/module').Module;
require('../core/module').Module = MockModule;

async function demonstrateToolDescriptions() {
    console.log('🎯 Enhanced Tool Descriptions Demo');
    console.log('====================================\n');

    console.log('📝 Creating RespAct agent with enhanced tool descriptions...\n');

    // Create agent with enhanced tool descriptions
    const agent = new RespAct('question -> answer', {
        tools: {
            // Enhanced tools with detailed descriptions
            calculate: {
                description: "Performs mathematical calculations including arithmetic operations (addition, subtraction, multiplication, division), exponents, and basic mathematical functions. Use this when you need to compute numerical results or evaluate mathematical expressions.",
                function: (expression: string) => {
                    console.log(`🔧 CALCULATE TOOL: Evaluating "${expression}"`);
                    try {
                        const result = eval(expression);
                        console.log(`   ✅ Result: ${result}`);
                        return result;
                    } catch (error) {
                        console.log(`   ❌ Error: ${error}`);
                        return `Error: ${error}`;
                    }
                }
            },

            fetchStockPrice: {
                description: "Retrieves the current stock price for a given ticker symbol (e.g., AAPL, GOOGL, TSLA, MSFT). Returns the price in USD. Use this when you need current stock market data or stock prices for financial calculations.",
                function: (symbol: string) => {
                    console.log(`🔧 STOCK PRICE TOOL: Fetching price for "${symbol}"`);
                    const mockPrices: Record<string, number> = {
                        'AAPL': 150.25,
                        'GOOGL': 2800.50,
                        'TSLA': 245.80,
                        'MSFT': 320.15
                    };
                    const price = mockPrices[symbol.toUpperCase()] || 100.00;
                    console.log(`   ✅ ${symbol} price: $${price}`);
                    return price;
                }
            },

            convertCurrency: {
                description: "Converts an amount from one currency to another using current exchange rates. Takes three parameters: amount (number), from currency code (string, e.g., 'USD'), and to currency code (string, e.g., 'EUR'). Use this for currency conversions in financial calculations.",
                function: (amountStr: string, from: string, to: string) => {
                    const amount = parseFloat(amountStr);
                    console.log(`🔧 CURRENCY TOOL: Converting ${amount} ${from} to ${to}`);

                    const rates: Record<string, number> = {
                        'USD_EUR': 0.85,
                        'USD_GBP': 0.73,
                        'EUR_USD': 1.18,
                        'GBP_USD': 1.37
                    };

                    const rateKey = `${from.toUpperCase()}_${to.toUpperCase()}`;
                    const rate = rates[rateKey] || 1.0;
                    const converted = amount * rate;

                    console.log(`   ✅ ${amount} ${from} = ${converted.toFixed(2)} ${to} (rate: ${rate})`);
                    return converted.toFixed(2);
                }
            },

            // Legacy tool without description for comparison
            getRandomTip: () => {
                console.log('🔧 LEGACY TOOL: Getting random tip (no description)');
                const tips = [
                    "Diversify your portfolio",
                    "Dollar-cost averaging reduces risk",
                    "Research before investing"
                ];
                const tip = tips[Math.floor(Math.random() * tips.length)];
                console.log(`   ✅ Tip: ${tip}`);
                return tip;
            }
        },
        maxSteps: 4
    });

    // Show the enhanced prompt that includes tool descriptions
    console.log('📋 ENHANCED PROMPT WITH TOOL DESCRIPTIONS:');
    console.log('-'.repeat(50));
    const samplePrompt = (agent as any).buildInitialPrompt({ question: 'Sample question' });
    const toolSection = samplePrompt.split('\n\nQuestion:')[0];
    console.log(toolSection);
    console.log('-'.repeat(50) + '\n');

    // Test the agent with a complex query
    console.log('🧪 Testing with complex financial question...');
    const question = "I want to buy 150 shares of AAPL. Calculate the total cost in USD and then convert that amount to EUR.";
    console.log(`Question: "${question}"\n`);

    try {
        // This will trigger the mock LM to show how descriptions help tool selection
        const result = await agent.forward({ question });
        console.log('✅ Final Result:', result.answer);
        console.log('📊 Steps taken:', result.steps);
    } catch (error: any) {
        console.log('❌ Error:', error.message);
    }

    console.log('\n🎯 Key Improvements with Tool Descriptions:');
    console.log('• AI understands WHAT each tool does');
    console.log('• AI knows WHEN to use each tool');
    console.log('• AI understands HOW to format parameters');
    console.log('• Reduced tool misuse and errors');
    console.log('• Better reasoning and tool selection');
    console.log('• Backward compatible with legacy tools');
}

// Restore original module
function cleanup() {
    require('../core/module').Module = OriginalModule;
}

// Run the demo
if (require.main === module) {
    demonstrateToolDescriptions()
        .then(() => {
            console.log('\n✨ Demo completed successfully!');
            cleanup();
        })
        .catch((error) => {
            console.error('❌ Demo failed:', error);
            cleanup();
        });
}

export { demonstrateToolDescriptions }; 