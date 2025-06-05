import { configure, RespAct } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

// Configure with your API key
configure({
    lm: new OpenAILM({
        apiKey: process.env.OPENAI_API_KEY || 'OPENAPI KEY'
    })
});

async function demonstrateToolUsage() {
    console.log('=== Tool Usage Demonstration ===\n');

    // Create an agent with multiple tools
    const agent = new RespAct("question -> answer", {
        tools: {
            calculate: (expression: string) => {
                console.log('🔧 TOOL CALLED: calculate(' + expression + ')');
                try {
                    const result = eval(expression);
                    console.log('   ✅ Calculation result:', result);
                    return result;
                } catch (error) {
                    console.log('   ❌ Calculation error:', error);
                    return `Error: ${error}`;
                }
            },

            fetchStockPrice: (symbol: string) => {
                console.log('🔧 TOOL CALLED: fetchStockPrice(' + symbol + ')');
                // Mock stock prices
                const prices: Record<string, number> = {
                    'AAPL': 150.25,
                    'GOOGL': 2800.50,
                    'TSLA': 245.80,
                    'MSFT': 320.15
                };
                const price = prices[symbol.toUpperCase()] || 100;
                console.log(`   ✅ ${symbol} price: $${price}`);
                return price;
            },

            convertCurrency: (amount: number, from: string, to: string) => {
                console.log(`🔧 TOOL CALLED: convertCurrency(${amount}, ${from}, ${to})`);
                // Mock exchange rates
                const rates: Record<string, number> = {
                    'USD_EUR': 0.85,
                    'USD_GBP': 0.73
                };
                const rate = rates[`${from}_${to}`] || 1.0;
                const converted = amount * rate;
                console.log(`   ✅ ${amount} ${from} = ${converted.toFixed(2)} ${to}`);
                return converted.toFixed(2);
            }
        },
        maxSteps: 6
    });

    console.log('🤖 How the LLM knows what tools are available:');
    console.log('1. Tool names are listed in the prompt: calculate, fetchStockPrice, convertCurrency');
    console.log('2. The LLM is instructed to use this format:');
    console.log('   Action: [tool name]');
    console.log('   Action Input: [parameters]');
    console.log('3. RespAct extracts these patterns and calls the actual functions');
    console.log('4. The tool results are fed back to the LLM as "Observation:"\n');

    // Example 1: Simple calculation
    console.log('--- Example 1: Simple Calculation ---');
    const question1 = "What is 15 * 8 + 32?";
    console.log('Question:', question1);

    try {
        const result1 = await agent.forward({ question: question1 });
        console.log('Final Answer:', result1.answer);
        console.log('Steps taken:', result1.steps);
    } catch (error: any) {
        console.log('Error:', error.message);
    }

    console.log('\n--- Example 2: Multi-step with tools ---');
    const question2 = "Get the current price of AAPL stock, then calculate what 100 shares would cost in EUR.";
    console.log('Question:', question2);

    try {
        const result2 = await agent.forward({ question: question2 });
        console.log('Final Answer:', result2.answer);
        console.log('Steps taken:', result2.steps);
    } catch (error: any) {
        console.log('Error:', error.message);
    }
}

// Run the demonstration
if (require.main === module) {
    demonstrateToolUsage().catch(console.error);
} 