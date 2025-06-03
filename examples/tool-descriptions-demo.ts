import { configure, RespAct } from '../packages/core/src/index';
import { OpenAILM } from '../packages/openai/src/index';

// Configure with your API key
configure({
    lm: new OpenAILM({
        apiKey: process.env.OPENAI_API_KEY || 'OPENAPI KEY'
    })
});

async function demonstrateToolDescriptions() {
    console.log('=== Enhanced Tool Descriptions Demonstration ===\n');

    // Create an agent with tools that have detailed descriptions
    const agent = new RespAct("question -> answer", {
        tools: {
            // New format: tools with descriptions
            calculate: {
                description: "Performs mathematical calculations including arithmetic operations (addition, subtraction, multiplication, division), exponents, and basic mathematical functions. Use this when you need to compute numerical results.",
                function: (expression: string) => {
                    console.log('🔧 TOOL CALLED: calculate(' + expression + ')');
                    try {
                        const result = eval(expression);
                        console.log('   ✅ Calculation result:', result);
                        return result;
                    } catch (error) {
                        console.log('   ❌ Calculation error:', error);
                        return `Error: ${error}`;
                    }
                }
            },

            fetchStockPrice: {
                description: "Retrieves the current stock price for a given ticker symbol (e.g., AAPL, GOOGL, TSLA). Returns the price in USD. Use this when you need current stock market data.",
                function: (symbol: string) => {
                    console.log('🔧 TOOL CALLED: fetchStockPrice(' + symbol + ')');
                    const prices: Record<string, number> = {
                        'AAPL': 150.25,
                        'GOOGL': 2800.50,
                        'TSLA': 245.80,
                        'MSFT': 320.15,
                        'AMZN': 3200.75
                    };
                    const price = prices[symbol.toUpperCase()] || Math.random() * 1000 + 50;
                    console.log(`   ✅ ${symbol} price: $${price}`);
                    return price;
                }
            },

            convertCurrency: {
                description: "Converts an amount from one currency to another using current exchange rates. Accepts amount (number), from currency code (e.g., USD), and to currency code (e.g., EUR). Use this for currency conversions.",
                function: (amount: number, from: string, to: string) => {
                    console.log(`🔧 TOOL CALLED: convertCurrency(${amount}, ${from}, ${to})`);
                    const rates: Record<string, number> = {
                        'USD_EUR': 0.85,
                        'USD_GBP': 0.73,
                        'EUR_USD': 1.18,
                        'GBP_USD': 1.37
                    };
                    const rateKey = `${from.toUpperCase()}_${to.toUpperCase()}`;
                    const rate = rates[rateKey] || 1.0;
                    const converted = amount * rate;
                    console.log(`   ✅ ${amount} ${from} = ${converted.toFixed(2)} ${to}`);
                    return converted.toFixed(2);
                }
            },

            searchNews: {
                description: "Searches for recent news articles related to a given topic or company. Returns a summary of relevant news. Use this when you need current events or news information about stocks, companies, or market conditions.",
                function: (query: string) => {
                    console.log(`🔧 TOOL CALLED: searchNews('${query}')`);
                    const mockNews = [
                        `${query} reported strong quarterly earnings, beating analyst expectations.`,
                        `Market analysts predict continued growth for ${query} in the coming quarter.`,
                        `Recent developments in ${query} sector show positive trends.`
                    ];
                    const news = mockNews[Math.floor(Math.random() * mockNews.length)];
                    console.log(`   ✅ News: ${news}`);
                    return news;
                }
            },

            // Legacy format: function without description (still supported)
            getRandomFact: () => {
                console.log('🔧 TOOL CALLED: getRandomFact() [Legacy format]');
                const facts = [
                    "The first stock exchange was established in Amsterdam in 1602.",
                    "Warren Buffett made his first investment at age 11.",
                    "The term 'bull market' comes from how bulls attack upward with their horns."
                ];
                const fact = facts[Math.floor(Math.random() * facts.length)];
                console.log(`   ✅ Random fact: ${fact}`);
                return fact;
            }
        },
        maxSteps: 6
    });

    console.log('🤖 Enhanced Tool Descriptions:');
    console.log('✨ NEW: Tools now include detailed descriptions that help the AI understand:');
    console.log('   • What each tool does');
    console.log('   • When to use each tool');
    console.log('   • What parameters to provide');
    console.log('   • What kind of output to expect');
    console.log('🔄 LEGACY: Old tool format (functions only) still works for backward compatibility\n');

    // Example 1: Complex calculation with contextual tool selection
    console.log('--- Example 1: Complex Financial Calculation ---');
    const question1 = "I want to buy 150 shares of AAPL. Calculate the total cost in USD and then convert that to EUR.";
    console.log('Question:', question1);
    console.log('💡 Expected: AI should use fetchStockPrice, calculate, and convertCurrency tools based on descriptions');

    try {
        const result1 = await agent.forward({ question: question1 });
        console.log('Final Answer:', result1.answer);
        console.log('Steps taken:', result1.steps);
    } catch (error: any) {
        console.log('Error:', error.message);
    }

    console.log('\n--- Example 2: News-Informed Investment Decision ---');
    const question2 = "Get the latest news about Tesla, then tell me the current TSLA stock price and whether it's a good time to invest based on the news.";
    console.log('Question:', question2);
    console.log('💡 Expected: AI should use searchNews and fetchStockPrice tools intelligently');

    try {
        const result2 = await agent.forward({ question: question2 });
        console.log('Final Answer:', result2.answer);
        console.log('Steps taken:', result2.steps);
    } catch (error: any) {
        console.log('Error:', error.message);
    }

    console.log('\n--- Example 3: Legacy Tool Usage ---');
    const question3 = "Tell me an interesting fact about investing.";
    console.log('Question:', question3);
    console.log('💡 Expected: AI should use the legacy getRandomFact tool despite no description');

    try {
        const result3 = await agent.forward({ question: question3 });
        console.log('Final Answer:', result3.answer);
        console.log('Steps taken:', result3.steps);
    } catch (error: any) {
        console.log('Error:', error.message);
    }

    console.log('\n🎯 Key Benefits of Tool Descriptions:');
    console.log('• More intelligent tool selection by the AI');
    console.log('• Better parameter formatting and usage');
    console.log('• Reduced tool misuse and errors');
    console.log('• Self-documenting code for developers');
    console.log('• Backward compatibility with existing code');
}

// Comparison function to show the difference
function showPromptComparison() {
    console.log('\n=== Prompt Comparison: Before vs After ===\n');

    // Before: Legacy format
    const legacyAgent = new RespAct("question -> answer", {
        tools: {
            calculate: (expr: string) => eval(expr),
            fetchStockPrice: (symbol: string) => 150.25
        }
    });

    // After: With descriptions
    const enhancedAgent = new RespAct("question -> answer", {
        tools: {
            calculate: {
                description: "Performs mathematical calculations including arithmetic operations",
                function: (expr: string) => eval(expr)
            },
            fetchStockPrice: {
                description: "Retrieves current stock price for a ticker symbol in USD",
                function: (symbol: string) => 150.25
            }
        }
    });

    console.log('BEFORE (Legacy):');
    const legacyPrompt = (legacyAgent as any).buildInitialPrompt({ question: 'Test' });
    const toolSection = legacyPrompt.split('\n')[0];
    console.log(toolSection);

    console.log('\nAFTER (Enhanced):');
    const enhancedPrompt = (enhancedAgent as any).buildInitialPrompt({ question: 'Test' });
    const enhancedToolSection = enhancedPrompt.split('\n').slice(0, 3).join('\n');
    console.log(enhancedToolSection);
}

// Run the demonstration
if (require.main === module) {
    showPromptComparison();
    demonstrateToolDescriptions().catch(console.error);
} 