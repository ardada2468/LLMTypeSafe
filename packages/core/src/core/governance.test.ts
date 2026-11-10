import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { auditLogger, type AuditRecord } from './audit';
import { buildCacheKey, FileCache, MemoryCache } from './cache';
import { context } from './config';
import { AuthError } from './errors';
import { wrapLM } from './middleware';
import { redact } from './redact';
import { Tool } from './tool';
import { Predict } from '../modules/predict';
import { ReAct } from '../modules/react';
import { JSONAdapter } from '../adapters/json';
import { MockLM } from '../test-utils';
import { BaseLM } from './base-lm';
import type { ChatMessage, ChatResult, LLMCallOptions } from '../types/language-model';

const text = (messages: ChatMessage[]) => messages.map((m) => String(m.content)).join('\n');

describe('redact', () => {
    it('replaces personal data before the provider sees it and restores it in the reply', async () => {
        const counts: Array<Record<string, number>> = [];
        const lm = new MockLM({
            respond: (messages) =>
                /\[EMAIL_1\]/.test(text(messages)) ? 'I will write to [EMAIL_1].' : 'leak',
        });
        const wrapped = wrapLM(lm, redact({ onRedact: (c) => counts.push(c) }));

        const reply = await wrapped.chat([
            {
                role: 'user',
                content:
                    'Mail jane.doe@example.com (again: jane.doe@example.com), call +1 415-555-0132, ' +
                    'SSN 123-45-6789, card 4111 1111 1111 1111, host 10.0.0.12.',
            },
        ]);

        const sent = text(lm.calls[0]!.messages);
        expect(sent).toBe(
            'Mail [EMAIL_1] (again: [EMAIL_1]), call [PHONE_1], SSN [SSN_1], card [CARD_1], host [IP_1].'
        );
        expect(reply).toBe('I will write to jane.doe@example.com.');
        expect(counts).toEqual([{ EMAIL: 1, SSN: 1, CARD: 1, IP: 1, PHONE: 1 }]);
    });

    it('leaves ordinary numbers and invalid card numbers alone', async () => {
        const lm = new MockLM({ respond: () => 'ok' });
        const wrapped = wrapLM(lm, redact());
        const content =
            'Order 1234 5678 9012 3456 shipped 2026-11-10, total 1,234.50, version 1.2.3.';
        await wrapped.chat([{ role: 'user', content }]);
        expect(text(lm.calls[0]!.messages)).toBe(content);
    });

    it('supports custom patterns, chosen entities and roles, and text parts', async () => {
        const lm = new MockLM({ respond: () => 'ok' });
        const wrapped = wrapLM(
            lm,
            redact({
                entities: ['email'],
                patterns: [{ name: 'account', pattern: /ACC-\d+/ }],
                roles: ['user'],
            })
        );
        await wrapped.chat([
            { role: 'system', content: 'Escalate to ops@example.com.' },
            {
                role: 'user',
                content: [{ type: 'text', text: 'ACC-991 and ACC-992, a@b.co, 123-45-6789' }],
            },
        ]);
        const [system, user] = lm.calls[0]!.messages;
        expect(system!.content).toBe('Escalate to ops@example.com.');
        expect(user!.content).toEqual([
            { type: 'text', text: '[ACCOUNT_1] and [ACCOUNT_2], [EMAIL_1], 123-45-6789' },
        ]);
    });

    it('restores values inside structured results and tool-call arguments', async () => {
        const structured = new MockLM({
            capabilities: { supportsStructuredOutput: true },
            structuredResponses: [{ contact: { email: '[EMAIL_1]' }, note: 'none' }],
        });
        await expect(
            wrapLM(structured, redact()).generateStructured('Find a@b.co', { type: 'object' })
        ).resolves.toEqual({ contact: { email: 'a@b.co' }, note: 'none' });
        expect(structured.structuredCalls[0]!.prompt).toBe('Find [EMAIL_1]');

        class ToolLM extends BaseLM {
            constructor() {
                super('fake', 'm');
            }
            async chat() {
                return '';
            }
            async chatWithTools(_m: ChatMessage[], _o?: LLMCallOptions): Promise<ChatResult> {
                return {
                    content: '',
                    finishReason: 'tool_calls',
                    toolCalls: [{ name: 'send', arguments: { to: '[EMAIL_1]' } }],
                };
            }
            getCapabilities() {
                return new MockLM().getCapabilities();
            }
        }
        const result = await wrapLM(new ToolLM(), redact()).chatWithTools!([
            { role: 'user', content: 'Email a@b.co' },
        ]);
        expect(result.toolCalls![0]!.arguments).toEqual({ to: 'a@b.co' });
    });

    it('can leave placeholders in the reply', async () => {
        const lm = new MockLM({ respond: () => 'Noted [EMAIL_1].' });
        const reply = await wrapLM(lm, redact({ restore: false })).chat([
            { role: 'user', content: 'a@b.co' },
        ]);
        expect(reply).toBe('Noted [EMAIL_1].');
    });
});

describe('auditLogger', () => {
    const answer = '[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]';

    it('records module and LM calls without content, attributed by metadata', async () => {
        const records: AuditRecord[] = [];
        const lm = new MockLM({ responses: [answer] });
        await context(
            {
                lm,
                callbacks: [
                    auditLogger({
                        sink: (r) => void records.push(r),
                        base: { service: 'api' },
                    }),
                ],
            },
            () =>
                new Predict('question -> answer').forward(
                    { question: 'What is 2 + 2? My SSN is 123-45-6789.' },
                    { metadata: { userId: 'u_42' } }
                )
        );

        const lmRecord = records.find((r) => r.event === 'lm')!;
        const moduleRecord = records.find((r) => r.event === 'module')!;
        expect(lmRecord).toMatchObject({
            service: 'api',
            status: 'ok',
            method: 'chat',
            models: ['mock/mock-model'],
            metadata: { userId: 'u_42' },
            parentCallId: moduleRecord.callId,
        });
        expect(lmRecord.totalTokens).toBeGreaterThan(0);
        expect(moduleRecord.moduleId).toMatch(/^Predict#/);
        expect(Number.isNaN(Date.parse(moduleRecord.timestamp))).toBe(false);
        expect(JSON.stringify(records)).not.toContain('123-45-6789');
    });

    it('records failures by type, and content only when asked', async () => {
        const lines: string[] = [];
        const lm = new MockLM();
        vi.spyOn(lm, 'chat').mockRejectedValue(
            new AuthError('mock', 'bad key sk-secret', { status: 401 })
        );
        const run = (includeContent: boolean) =>
            context(
                {
                    lm,
                    callbacks: [
                        auditLogger({ sink: { write: (l) => lines.push(l) }, includeContent }),
                    ],
                },
                () => new Predict('question -> answer').forward({ question: 'q' })
            ).catch(() => undefined);

        await run(false);
        const quiet = lines.map((line) => JSON.parse(line) as AuditRecord);
        expect(quiet.find((r) => r.event === 'lm')).toMatchObject({
            status: 'error',
            errorType: 'AuthError',
            errorStatus: 401,
            models: ['mock-model'],
        });
        expect(lines.join('')).not.toContain('sk-secret');
        expect(lines.every((line) => line.endsWith('\n'))).toBe(true);

        lines.length = 0;
        await run(true);
        const loud = lines.map((line) => JSON.parse(line) as AuditRecord);
        expect(loud.find((r) => r.event === 'module')).toMatchObject({
            inputs: { question: 'q' },
            errorMessage: expect.stringContaining('bad key'),
        });
    });

    it('records tool calls and survives a failing sink', async () => {
        const records: AuditRecord[] = [];
        const tool = Tool.fromFunction(() => 'ok', { name: 'lookup' });
        await context(
            {
                callbacks: [
                    auditLogger({ sink: (r) => void records.push(r), events: ['tool'] }),
                    auditLogger({
                        sink: async () => {
                            throw new Error('disk full');
                        },
                    }),
                ],
            },
            () => tool.call({})
        );
        expect(records).toHaveLength(1);
        expect(records[0]).toMatchObject({ event: 'tool', toolName: 'lookup', status: 'ok' });
    });
});

describe('MemoryCache ttl', () => {
    afterEach(() => vi.useRealTimers());

    it('expires entries after ttlMs', () => {
        vi.useFakeTimers();
        const cache = new MemoryCache({ ttlMs: 1000 });
        cache.set('k', 'v');
        vi.advanceTimersByTime(999);
        expect(cache.get('k')).toBe('v');
        vi.advanceTimersByTime(1);
        expect(cache.get('k')).toBeUndefined();
        expect(cache.size).toBe(0);
        expect(() => new MemoryCache({ ttlMs: 0 })).toThrow(RangeError);
    });
});

describe('FileCache', () => {
    let directory: string;
    beforeEach(() => {
        directory = mkdtempSync(join(tmpdir(), 'tsdspy-cache-'));
    });
    afterEach(() => {
        vi.useRealTimers();
        rmSync(directory, { recursive: true, force: true });
    });

    it('persists replies across instances and serves a model call from disk', async () => {
        const nested = join(directory, 'a', 'b');
        const lm = new MockLM({ responses: ['first'] });
        const ask = () =>
            context({ cache: new FileCache({ directory: nested }) }, () =>
                lm.chat([{ role: 'user', content: 'hi' }])
            );

        await expect(ask()).resolves.toBe('first');
        await expect(ask()).resolves.toBe('first');
        expect(lm.calls).toHaveLength(1);
        expect(readdirSync(nested).filter((name) => name.endsWith('.tmp'))).toEqual([]);
    });

    it('expires entries, treats corrupt files as a miss, and never leaves the directory', async () => {
        const cache = new FileCache({ directory, ttlMs: 50 });
        const key = buildCacheKey({ provider: 'p', model: 'm', operation: 'chat' });
        await cache.set(key, { a: 1 });
        await expect(cache.get(key)).resolves.toEqual({ a: 1 });
        await new Promise((resolve) => setTimeout(resolve, 80));
        await expect(cache.get(key)).resolves.toBeUndefined();

        writeFileSync(join(directory, `${key}.json`), '{ truncated');
        await expect(cache.get(key)).resolves.toBeUndefined();

        await cache.set('../../escape', 'x');
        await expect(cache.get('../../escape')).resolves.toBe('x');
        expect(readdirSync(directory).every((name) => /^[a-f0-9]+\.json$/.test(name))).toBe(
            true
        );

        writeFileSync(join(directory, 'README.txt'), 'keep me');
        await cache.clear();
        expect(readdirSync(directory)).toEqual(['README.txt']);
        expect(() => new FileCache({ directory: '' })).toThrow(TypeError);
    });
});

describe('ReAct toolTimeoutMs', () => {
    it('reports a tool that overruns as a failed call and carries on', async () => {
        const step = (tool: string) =>
            JSON.stringify({
                next_thought: 't',
                next_tool_name: tool,
                next_tool_args: '{"input":"x"}',
            });
        const lm = new MockLM({
            responses: [
                step('slow'),
                step('finish'),
                JSON.stringify({ reasoning: 'r', answer: 'a' }),
            ],
        });
        const agent = new ReAct('question -> answer', {
            tools: {
                slow: { description: 'slow', function: () => new Promise<string>(() => {}) },
            },
            lm,
            adapter: new JSONAdapter(),
            toolTimeoutMs: 30,
        });

        const result = await agent.forward({ question: 'Q' });
        expect(result.trajectory.observation_0).toContain('timed out after 30ms');
        expect(result.answer).toBe('a');
        expect(() => new ReAct('q -> a', { tools: {}, toolTimeoutMs: 0 })).toThrow(RangeError);
        expect(() => new ReAct('q -> a', { tools: {}, maxSteps: 0 })).toThrow(RangeError);
    });
});
