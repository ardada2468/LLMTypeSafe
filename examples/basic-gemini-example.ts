import { configure, Predict, ChainOfThought, RespAct, Signature, InputField, OutputField } from '@ts-dspy/core';
import { GeminiLM } from '@ts-dspy/gemini';

// Setup - configure global LM
configure({
    lm: new GeminiLM({
        apiKey: process.env.GEMINI_API_KEY || '***REDACTED*** ',
        model: 'gemini-2.0-flash',
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

// Example 5: Tool-using agent with RespAct (Enhanced Format)
async function toolExample() {
    console.log('\n=== Tool Usage Example (Enhanced Format) ===');
    console.log('🆕 Using enhanced tools with descriptions for better AI tool selection\n');

    const calculator = new RespAct("question -> answer", {
        tools: {
            calculate: {
                description: "Performs mathematical calculations including arithmetic operations. Use this when you need to compute numerical results.",
                function: (expression: string) => {
                    console.log('🔧 TOOL CALLED: calculate(' + expression + ')');
                    try {
                        // Using a safer method than eval for simple expressions
                        const result = new Function('return ' + expression)();
                        console.log('   → Result:', result);
                        return result;
                    } catch (error) {
                        console.log('   → Error:', error);
                        return `Error: ${error}`;
                    }
                }
            },
        },
        maxSteps: 4
    });

    const result = await calculator.forward({
        question: "What is 15 * 24 + 100 - 50?"
    });

    console.log('Question:', "What is 15 * 24 + 100 - 50?");
    console.log('Answer:', result.answer);
    console.log('Steps taken:', result.steps);
}

// Example 6: Error handling
async function errorHandlingExample() {
    console.log('\n=== Error Handling Example ===');

    try {
        // This will fail due to missing signature
        const badModule = new Predict("");
        await badModule.forward({ input: "test" });
    } catch (error: any) {
        console.log('Caught expected error:', error.message);
    }
}

// Example 7: Usage statistics
async function usageStatsExample() {
    console.log('\n=== Usage Statistics Example ===');

    const lm = new GeminiLM({
        apiKey: process.env.GEMINI_API_KEY || '***REDACTED*** ',
        model: 'gemini-2.0-flash',
    });



    lm.resetUsage();

    const qa = new Predict("question -> answer", lm);
    await qa.forward({ question: "Hello, how are you?" });

    const usage = lm.getUsage();
    console.log('Usage Statistics (Gemini):');
    console.log('Note: The Gemini API currently does not provide token usage information via this SDK implementation.');
    console.log('- Prompt tokens:', usage.promptTokens);
    console.log('- Completion tokens:', usage.completionTokens);
    console.log('- Total tokens:', usage.totalTokens);
    console.log('- Estimated cost: $', usage.totalCost?.toFixed(4) || '0.0000');
}

// Main execution
async function main() {
    console.log('TS-DSPy Gemini Framework Examples\n');

    try {
        await simpleExample();
        await typedExample();
        await classBasedExample();
        await chainOfThoughtExample();
        await toolExample();
        await errorHandlingExample();
        await usageStatsExample();

        console.log('\n=== All Gemini Examples Completed Successfully! ===');
    } catch (error: any) {
        console.error('An example failed:', error.message);

        if (error.message.includes('API key')) {
            console.log('\n💡 Tip: Set your Gemini API key in the GEMINI_API_KEY environment variable');
        } else if (error.message.includes('API Error')) {
            console.log('\n💡 Tip: This may be due to a blocked prompt. Check the Gemini API safety settings.');
        }
    }
}

// Run examples if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

export { main };