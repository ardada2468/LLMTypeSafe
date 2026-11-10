/**
 * A production-shaped model stack, runnable offline.
 *
 * Wraps a model in the layers a deployment usually needs — redaction of
 * personal data, a spending cap, a concurrency limit, retries, and a fallback
 * behind a circuit breaker — and records every call to an audit trail.
 *
 * It runs against `MockLM`, so there is no API key and no network: the point
 * is the wiring. Swap `primary` and `secondary` for real providers.
 *
 *   npm run example:production
 */
import {
    Predict,
    LMError,
    auditLogger,
    budget,
    circuitBreaker,
    concurrencyLimit,
    configure,
    fallback,
    redact,
    retry,
    wrapLM,
    type AuditRecord,
    type ChatMessage,
} from '@ts-dspy/core';
import { MockLM } from '@ts-dspy/core/testing';
import { section } from './utils';

const lastMessage = (messages: ChatMessage[]) => String(messages.at(-1)?.content ?? '');
const reply = (answer: string) => `[[ ## answer ## ]]\n${answer}\n\n[[ ## completed ## ]]`;

// What each provider actually receives, so the redaction is visible.
const seenByProvider: string[] = [];

// A primary that is down, and a secondary that answers.
const primary = new MockLM({
    respond: () => {
        throw new LMError('primary', 'Service unavailable', { status: 503 });
    },
});
const secondary = new MockLM({
    respond: (messages) => {
        seenByProvider.push(lastMessage(messages));
        const email = /\[EMAIL_\d+\]/.exec(lastMessage(messages))?.[0] ?? 'the customer';
        return reply(`I have opened a ticket and will reply to ${email}.`);
    },
});

const lm = wrapLM(
    primary,
    // Outermost: everything beneath only ever sees placeholders.
    redact({ onRedact: (counts) => console.log('redacted:', counts) }),
    // A hard stop for a runaway loop.
    budget({ maxRequests: 100 }),
    concurrencyLimit({ maxConcurrent: 8 }),
    fallback([secondary]),
    // Inside the fallback: once the primary has failed twice, stop calling it
    // and go straight to the secondary.
    circuitBreaker({
        failureThreshold: 2,
        resetTimeoutMs: 30_000,
        onStateChange: (state, previous) => console.log(`circuit: ${previous} -> ${state}`),
    }),
    retry({ maxRetries: 1, initialDelayMs: 10 })
);

const audit: AuditRecord[] = [];
configure({
    lm,
    callbacks: [auditLogger({ sink: (record) => void audit.push(record), events: ['lm'] })],
});

async function main() {
    const support = new Predict('message -> answer');

    section('Three requests against a failing primary');
    for (const requestId of ['r1', 'r2', 'r3']) {
        const result = await support.forward(
            {
                message:
                    'My order is late. Reach me at jane.doe@example.com or +1 415-555-0132.',
            },
            { metadata: { requestId, userId: 'u_42' } }
        );
        console.log(`${requestId}: ${result.answer}`);
    }

    section('What the provider saw');
    console.log(seenByProvider[0]?.split('\n').find((line) => line.includes('[EMAIL_')));

    section('Calls reaching each model');
    console.log('primary attempts:', primary.calls.length, '(the breaker stopped the rest)');
    console.log('secondary calls: ', secondary.calls.length);

    section('Audit trail');
    for (const record of audit) {
        console.log(JSON.stringify(record));
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
