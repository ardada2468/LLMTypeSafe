/**
 * Running ts-dspy programs on a Vercel AI SDK model.
 *
 * `AISDKLM` wraps any AI SDK `LanguageModel` — every provider package, the AI
 * Gateway, and models wrapped in AI SDK middleware — as a ts-dspy model. This
 * example uses the AI Gateway, which ships with `ai` itself, so it needs no
 * extra provider package:
 *
 *   export AI_GATEWAY_API_KEY="..."
 *   npm run example:ai-sdk
 *
 * With a provider package instead (`npm install @ai-sdk/openai`):
 *
 *   import { openai } from '@ai-sdk/openai';
 *   new AISDKLM({ model: openai('gpt-5.2') });
 */
import { gateway } from 'ai';
import { z } from 'zod';
import { ChainOfThought, Predict, configure, signature } from '@ts-dspy/core';
import { AISDKLM } from '@ts-dspy/ai-sdk';
import { requireEnv, section } from './utils';

const Triage = signature({
    description: 'Triage an incoming bug report.',
    input: z.object({ report: z.string() }),
    output: z.object({
        severity: z.enum(['low', 'medium', 'high', 'critical']),
        component: z.string(),
        summary: z.string().describe('one sentence'),
    }),
});

async function main(): Promise<void> {
    requireEnv('AI_GATEWAY_API_KEY');

    const lm = new AISDKLM({
        model: gateway(process.env.AI_GATEWAY_MODEL ?? 'openai/gpt-5-mini'),
    });
    configure({ lm });

    section('Structured output through the AI SDK');
    const triage = await new Predict(Triage).forward({
        report: 'Since this morning every checkout fails with a 500 and customers cannot pay.',
    });
    console.log(`severity:  ${triage.severity}`);
    console.log(`component: ${triage.component}`);
    console.log(`summary:   ${triage.summary}`);

    section('Streaming');
    const cot = new ChainOfThought('question -> answer');
    let shown = 0;
    for await (const partial of cot.stream({
        question: 'Why is the sky blue? One sentence.',
    })) {
        const answer = typeof partial.answer === 'string' ? partial.answer : '';
        process.stdout.write(answer.slice(shown));
        shown = answer.length;
    }
    console.log();

    section('Usage');
    const usage = lm.getUsage();
    console.log(`requests: ${usage.requestCount}, tokens: ${usage.totalTokens}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
