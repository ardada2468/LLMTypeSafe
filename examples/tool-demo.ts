/**
 * RespAct: reason-and-act loops with tools.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npx tsx examples/tool-demo.ts
 */
import { Signature, InputField, OutputField, RespAct, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { evaluateArithmetic, requireEnv, section } from './utils';

class ResearchQuestion extends Signature {
    static description = 'Answer a question using the available tools.';

    @InputField({ description: 'the question to answer' })
    question!: string;

    @OutputField({ description: 'the final answer' })
    answer!: string;

    @OutputField({ description: 'confidence between 0 and 1', type: 'number' })
    confidence!: number;
}

// A small stand-in for a real data source.
const POPULATION: Record<string, number> = {
    tokyo: 37_000_000,
    delhi: 34_000_000,
    shanghai: 30_000_000,
    paris: 11_000_000,
};

async function main(): Promise<void> {
    configure({ lm: new OpenAILM({ apiKey: requireEnv('OPENAI_API_KEY') }) });

    // Tool descriptions are what the model uses to decide when to call each one,
    // so they are worth writing carefully.
    const tools = {
        calculator: {
            description:
                'Evaluate an arithmetic expression. Input: an expression using + - * / % ^ and parentheses, e.g. "37000000 / 11000000". Returns the numeric result.',
            // Never eval() model output — this is a bounded arithmetic parser.
            function: (expression: string) => evaluateArithmetic(expression),
        },
        population: {
            description:
                'Look up a city\'s metro-area population. Input: a city name, e.g. "Tokyo". Returns the population as a number, or an error if the city is unknown.',
            function: (city: string) => {
                const value = POPULATION[city.trim().toLowerCase()];
                if (value === undefined) {
                    throw new Error(
                        `Unknown city. Known cities: ${Object.keys(POPULATION).join(', ')}`
                    );
                }
                return value;
            },
        },
    };

    section('RespAct with tools');

    const agent = new RespAct(ResearchQuestion, {
        tools,
        maxSteps: 8,
        // Replaces the console logging the loop used to do internally.
        onEvent: (event) => {
            switch (event.type) {
                case 'tool_call':
                    console.log(`  [step ${event.step}] ${event.tool}(${event.input})`);
                    break;
                case 'tool_result':
                    console.log(`  [step ${event.step}] -> ${event.output}`);
                    break;
                case 'tool_error':
                    console.log(`  [step ${event.step}] tool failed: ${event.error}`);
                    break;
                case 'repeated_tool_call':
                    console.log(`  [step ${event.step}] skipped repeated call`);
                    break;
                default:
                    break;
            }
        },
    });

    const result = await agent.forward({
        question: 'How many times larger is the population of Tokyo than that of Paris?',
    });

    console.log(`\nanswer:     ${result.answer}`);
    console.log(`confidence: ${result.confidence}`);
    console.log(`steps:      ${result.steps}`);

    // Bare functions still work; they just get a generated description.
    section('Tools as plain functions');

    const simple = new RespAct('question -> answer', {
        tools: { calculator: (expression: string) => evaluateArithmetic(expression) },
        maxSteps: 5,
    });

    const sum = await simple.forward({ question: 'What is 1234 * 5678?' });
    console.log(`answer: ${sum.answer}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
