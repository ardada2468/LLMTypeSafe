import { configure, Predict, ChainOfThought, RespAct, Signature, InputField, OutputField } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

// Setup - configure global LM
configure({
    lm: new OpenAILM({
        apiKey: process.env.OPENAI_API_KEY || '***REDACTED*** '
    })
});

// Example 1: Simple string signature
async function simpleExample() {
    console.log('=== Simple Example ===');

    const qa = new Predict("question -> answer");
    const result = await qa.forward({ question: "What is the capital of France?" });

    console.log('Question:', "What is the capital of France?");
    console.log('Answer:', result.answer);
}

// Example 2: Typed signature with multiple fields
async function typedExample() {
    console.log('\n=== Typed Example ===');

    const math = new Predict("question -> answer: float, explanation");
    const result = await math.forward({
        question: "What is the probability of getting heads when flipping a fair coin?"
    });

    console.log('Question:', "What is the probability of getting heads when flipping a fair coin?");
    console.log('Answer:', result.answer, '(type:', typeof result.answer, ')');
    console.log('Explanation:', result.explanation);
}

// Example 3: Class-based signature with decorators
class SentimentAnalysis extends Signature {
    @InputField({ description: "Text to analyze for sentiment" })
    text!: string;

    @OutputField({ description: "Sentiment classification" })
    sentiment!: 'positive' | 'negative' | 'neutral';

    @OutputField({ description: "Confidence score between 0 and 1" })
    confidence!: number;

    static description = "Analyze the sentiment of the given text";
}

async function classBasedExample() {
    console.log('\n=== Class-based Signature Example ===');

    const classifier = new Predict(SentimentAnalysis);
    const result = await classifier.forward({
        text: "I love this new TypeScript framework! It's amazing!"
    });

    console.log('Text:', "I love this new TypeScript framework! It's amazing!");
    console.log('Sentiment:', result.sentiment);
    console.log('Confidence:', result.confidence);
}

// Example 4: Chain of Thought reasoning
async function chainOfThoughtExample() {
    console.log('\n=== Chain of Thought Example ===');

    const reasoner = new ChainOfThought("question -> answer: int");
    const result = await reasoner.forward({
        question: "A store sells apples for $2 each and oranges for $3 each. If someone buys 4 apples and 3 oranges, how much do they pay in total?"
    });

    console.log('Question:', "A store sells apples for $2 each and oranges for $3 each. If someone buys 4 apples and 3 oranges, how much do they pay in total?");
    console.log('Reasoning:', result.reasoning);
    console.log('Answer:', result.answer);
}

// Example 4.5: Simple Enhanced Tools Demo (Guaranteed to work)
async function simpleToolsDemo() {
    console.log('\n=== Simple Enhanced Tools Demo ===');
    console.log('🆕 Testing enhanced tools with clear descriptions\n');

    const simpleAgent = new RespAct("question -> answer", {
        tools: {
            add: {
                description: "Adds two numbers together. Provide two numbers separated by a comma (e.g., '5,3' or '10,20'). Use this for addition operations.",
                function: (input: string) => {
                    console.log(`🔧 ADD: ${input}`);
                    const parts = input.split(',');
                    if (parts.length !== 2) {
                        const result = `Error: Please provide two numbers separated by comma, got: ${input}`;
                        console.log(`   → ${result}`);
                        return result;
                    }
                    const a = parseFloat(parts[0].trim());
                    const b = parseFloat(parts[1].trim());
                    const result = a + b;
                    console.log(`   → ${a} + ${b} = ${result}`);
                    return result;
                }
            },

            multiply: {
                description: "Multiplies two numbers together. Provide two numbers separated by a comma (e.g., '4,7' or '3,9'). Use this for multiplication operations.",
                function: (input: string) => {
                    console.log(`🔧 MULTIPLY: ${input}`);
                    const parts = input.split(',');
                    if (parts.length !== 2) {
                        const result = `Error: Please provide two numbers separated by comma, got: ${input}`;
                        console.log(`   → ${result}`);
                        return result;
                    }
                    const a = parseFloat(parts[0].trim());
                    const b = parseFloat(parts[1].trim());
                    const result = a * b;
                    console.log(`   → ${a} × ${b} = ${result}`);
                    return result;
                }
            }
        },
        maxSteps: 4
    });

    const question = "What is 15 plus 8, and then multiply that result by 3?";
    console.log('Question:', question);

    try {
        const result = await simpleAgent.forward({ question });
        console.log('\n✅ Answer:', result.answer);
        console.log('📊 Steps taken:', result.steps);
        console.log('🎯 This demonstrates how enhanced tool descriptions help the AI select the right tools!');
    } catch (error: any) {
        console.log('❌ Error:', error.message);
    }
}

// Example 5: Tool-using agent with RespAct (Enhanced Format)
async function toolExample() {
    console.log('\n=== Tool Usage Example (Enhanced Format) ===');
    console.log('🆕 Using enhanced tools with descriptions for better AI tool selection\n');

    const calculator = new RespAct("question -> answer", {
        tools: {
            // Enhanced format with descriptions
            calculate: {
                description: "Performs mathematical calculations including arithmetic operations (addition, subtraction, multiplication, division), exponents, and basic mathematical functions. Use this when you need to compute numerical results or solve math problems.",
                function: (expression: string) => {
                    console.log('🔧 TOOL CALLED: calculate(' + expression + ')');
                    try {
                        // Simple calculator - in production, use a proper math library
                        const result = eval(expression);
                        console.log('   → Result:', result);
                        return result;
                    } catch (error) {
                        console.log('   → Error:', error);
                        return `Error: ${error}`;
                    }
                }
            },

            search: {
                description: "Searches for information on the internet or in databases. Provide a search query as input and receive relevant search results. Use this when you need to find information not available through calculations.",
                function: async (query: string) => {
                    console.log('🔧 TOOL CALLED: search(' + query + ')');
                    // Mock search function
                    const result = `Mock search results for: ${query}`;
                    console.log('   → Result:', result);
                    return result;
                }
            }
        },
        maxSteps: 8
    });

    const result = await calculator.forward({
        question: "What is 15 * 24 + 100 - 50?"
    });

    console.log('Question:', "What is 15 * 24 + 100 - 50?");
    console.log('Answer:', result.answer);
    console.log('Steps taken:', result.steps);
}

// Example 6: Advanced Agentic Behavior - Multi-tool usage (Enhanced Format)
async function agenticExample() {
    console.log('\n=== Advanced Agentic Example (Enhanced Format) ===');
    console.log('🆕 Demonstrating how the LLM autonomously chooses and sequences enhanced tools with descriptions...\n');

    const agent = new RespAct("question -> answer", {
        tools: {
            fetchPrice: {
                description: "Retrieves current stock prices for given ticker symbols (e.g., AAPL, TSLA, GOOGL, MSFT). Can handle single symbols or multiple symbols separated by commas. Returns prices in USD. Use this when you need current stock market data.",
                function: (symbol: string) => {
                    console.log('🔧 TOOL CALLED: fetchPrice(' + symbol + ')');
                    // Handle multiple symbols separated by comma or just single symbol
                    const symbols = symbol.split(',').map(s => s.trim().toUpperCase());
                    const prices: Record<string, number> = {
                        'AAPL': 150.25,
                        'TSLA': 245.80,
                        'GOOGL': 2800.50,
                        'MSFT': 320.15
                    };

                    if (symbols.length === 1) {
                        const price = prices[symbols[0]] || 100 + Math.random() * 200;
                        console.log(`   → ${symbols[0]} current price: $${price}`);
                        return price;
                    } else {
                        // Return multiple prices
                        const result = symbols.map(sym => {
                            const price = prices[sym] || 100 + Math.random() * 200;
                            console.log(`   → ${sym} current price: $${price}`);
                            return `${sym}: $${price}`;
                        }).join(', ');
                        return result;
                    }
                }
            },

            calculate: {
                description: "Performs mathematical calculations including arithmetic operations, multiplication, addition, subtraction, and division. Can handle expressions with numbers and basic mathematical operations. Use this when you need to compute numerical results, calculate totals, or perform financial calculations.",
                function: (expression: string) => {
                    console.log('🔧 TOOL CALLED: calculate(' + expression + ')');
                    try {
                        // Handle expressions that reference stock prices
                        let expr = expression;
                        // Replace common patterns
                        expr = expr.replace(/AAPL price/gi, '150.25');
                        expr = expr.replace(/TSLA price/gi, '245.80');

                        const result = eval(expr);
                        console.log('   → Calculation result:', result);
                        return result;
                    } catch (error) {
                        console.log('   → Calculation error:', error);
                        return `Error: ${error}`;
                    }
                }
            },

            convertCurrency: {
                description: "Converts monetary amounts from one currency to another using current exchange rates. Input format: 'amount,fromCurrency,toCurrency' (e.g., '100,USD,EUR') or 'amount from to' format. Supports USD, EUR, GBP, JPY. Use this for international financial calculations or currency conversions.",
                function: (input: string) => {
                    console.log(`🔧 TOOL CALLED: convertCurrency(${input})`);
                    // Parse input like "amount,fromCurrency,toCurrency" or "amount from to"
                    const parts = input.split(',').map(p => p.trim());
                    if (parts.length !== 3) {
                        // Try space-separated format
                        const spaceParts = input.split(/\s+/);
                        if (spaceParts.length >= 3) {
                            parts[0] = spaceParts[0];
                            parts[1] = spaceParts[spaceParts.length - 2];
                            parts[2] = spaceParts[spaceParts.length - 1];
                        }
                    }

                    if (parts.length !== 3) {
                        console.log('   → Error: Expected format: amount,fromCurrency,toCurrency');
                        return 'Error: Expected format: amount,fromCurrency,toCurrency';
                    }

                    const amount = parseFloat(parts[0]);
                    const fromCurrency = parts[1].toUpperCase();
                    const toCurrency = parts[2].toUpperCase();

                    // Mock exchange rates
                    const rates: Record<string, number> = {
                        'USD_EUR': 0.85,
                        'USD_GBP': 0.73,
                        'USD_JPY': 110.25
                    };
                    const rateKey = `${fromCurrency}_${toCurrency}`;
                    const rate = rates[rateKey] || 1.0;
                    const converted = amount * rate;
                    console.log(`   → ${amount} ${fromCurrency} = ${converted.toFixed(2)} ${toCurrency}`);
                    return converted.toFixed(2);
                }
            },

            getNews: {
                description: "Retrieves recent news articles and information about specific topics, companies, or stock symbols. Provide a topic, company name, or stock symbol to get relevant news summaries. Use this when you need current events or market sentiment information.",
                function: (topic: string) => {
                    console.log('🔧 TOOL CALLED: getNews(' + topic + ')');
                    const news = `Latest news about ${topic}: Stock showing positive momentum with recent earnings beat and analyst upgrades.`;
                    console.log('   → News summary retrieved');
                    return news;
                }
            }
        },
        maxSteps: 12
    });

    const complexQuestion = "If I buy 100 shares of AAPL and 50 shares of TSLA, what's my total investment in USD? Also convert that to EUR and tell me any recent news about Apple.";

    console.log('Question:', complexQuestion);
    console.log('\n--- LLM Agent Processing ---');

    const result = await agent.forward({
        question: complexQuestion
    });

    console.log('\n--- Final Result ---');
    console.log('Answer:', result.answer);
    console.log('Total steps taken:', result.steps);
}


// Example 8: Error handling
async function errorHandlingExample() {
    console.log('\n=== Error Handling Example ===');

    try {
        // This will fail due to missing signature
        const badModule = new Predict("");
        await badModule.forward({ input: "test" });
    } catch (error: any) {
        console.log('Caught expected error:', error.message);
    }

    try {
        // This will show graceful parsing error handling
        const parser = new Predict("input -> structured_output: json");
        // If LM returns malformed JSON, it should be handled gracefully
        // This is just for demonstration
        console.log('Parser created successfully');
    } catch (error: any) {
        console.log('Parser error:', error.message);
    }
}

// Example 9: Usage statistics
async function usageStatsExample() {
    console.log('\n=== Usage Statistics Example ===');

    const lm = new OpenAILM({
        apiKey: process.env.OPENAI_API_KEY || '***REDACTED*** ',
        model: 'gpt-3.5-turbo'
    });

    // Reset usage before starting
    lm.resetUsage();

    const qa = new Predict("question -> answer", lm);
    await qa.forward({ question: "Hello, how are you?" });

    const usage = lm.getUsage();
    console.log('Usage Statistics:');
    console.log('- Prompt tokens:', usage.promptTokens);
    console.log('- Completion tokens:', usage.completionTokens);
    console.log('- Total tokens:', usage.totalTokens);
    console.log('- Estimated cost: $', usage.totalCost?.toFixed(4) || '0.0000');
}

// Main execution
async function main() {
    console.log('TS-DSPy Framework Examples\n');

    try {
        await simpleExample();
        await typedExample();
        await classBasedExample();
        await chainOfThoughtExample();
        await simpleToolsDemo();
        await toolExample();
        await agenticExample();
        await errorHandlingExample();
        await usageStatsExample();

        console.log('\n=== All Examples Completed Successfully! ===');
    } catch (error: any) {
        console.error('Example failed:', error);

        if (error.message.includes('API key')) {
            console.log('\n💡 Tip: Set your OpenAI API key in the OPENAI_API_KEY environment variable');
        }
    }
}

// Run examples if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

export { main }; 