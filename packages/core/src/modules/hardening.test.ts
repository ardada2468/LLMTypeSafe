/** Regression tests for parsing and agent-loop failures found in a production-readiness review. */
import { z } from 'zod';
import { ChatAdapter } from '../adapters/chat';
import { JSONAdapter } from '../adapters/json';
import { XMLAdapter } from '../adapters/xml';
import { ContextLengthError, ValidationError } from '../core/errors';
import { signature } from '../core/signature';
import { BaseLM } from '../core/base-lm';
import type { ChatMessage, ChatResult } from '../types/language-model';
import { MockLM } from '../test-utils';
import { parsePartialJson } from '../utils/partial-json';
import { extractLabelledFields } from '../utils/parsing';
import { Predict } from './predict';
import { runSandboxed } from './program-of-thought';
import { ReAct } from './react';

const fencedCode = 'Use:\n```js\nconsole.log(1)\n```\ndone';

describe('JSON replies', () => {
    it('keeps a code fence that sits inside a string value', async () => {
        const raw = JSON.stringify({ code: fencedCode });
        expect(new JSONAdapter().parseOutputs('q -> code', raw)).toEqual({ code: fencedCode });

        const lm = new MockLM({
            capabilities: { supportsStructuredOutput: true },
            structuredResponses: [{ code: fencedCode }],
        });
        const result = await new Predict('q -> code', lm).forward({ q: 'x' });
        expect(result.code).toBe(fencedCode);
    });

    it('still unwraps a reply that is fenced as a whole', () => {
        const raw = '```json\n{"answer": "4"}\n```';
        expect(new JSONAdapter().parseOutputs('q -> answer', raw)).toEqual({ answer: '4' });
        const chatty = 'Sure!\n```json\n{"answer": "4"}\n```\nHope that helps.';
        expect(new JSONAdapter().parseOutputs('q -> answer', chatty)).toEqual({ answer: '4' });
    });

    it('never lets a model-chosen key reach a prototype', () => {
        const zodSig = signature({
            input: z.object({ q: z.string() }),
            output: z.object({ answer: z.string() }),
        });
        const adapter = new JSONAdapter();
        expect(adapter.parseOutputs(zodSig, '{"answer":"x","constructor":null}')).toMatchObject(
            {
                answer: 'x',
            }
        );
        expect(() =>
            adapter.parseOutputs('q -> answer', '{"__proto__":{"answer":"x"}}')
        ).toThrow(ValidationError);
    });

    it('survives pathological nesting in a partial reply', () => {
        expect(() => parsePartialJson('['.repeat(200_000))).not.toThrow();
    });
});

describe('text coercion', () => {
    const parse = (sig: string, value: string) =>
        new ChatAdapter().parseOutputs(sig, `[[ ## v ## ]]\n${value}\n\n[[ ## completed ## ]]`);

    it('reads thousands separators but not decimal commas or hex', () => {
        expect(parse('q -> v: number', '1,234.5')).toEqual({ v: 1234.5 });
        expect(parse('q -> v: number', '12,345,678')).toEqual({ v: 12345678 });
        expect(parse('q -> v: number', '85%')).toEqual({ v: 85 });
        expect(parse('q -> v: number', '-3.5e2')).toEqual({ v: -350 });
        expect(() => parse('q -> v: number', '1,5')).toThrow(ValidationError);
        expect(() => parse('q -> v: int', '0x10')).toThrow(ValidationError);
    });

    it('reads a fenced or annotated JSON array instead of splitting its syntax', () => {
        expect(parse('q -> v: string[]', '```json\n["a, b", "c"]\n```')).toEqual({
            v: ['a, b', 'c'],
        });
        expect(parse('q -> v: string[]', '["a, b", "c"] (two tags)')).toEqual({
            v: ['a, b', 'c'],
        });
        expect(parse('q -> v: string[]', 'red, green; blue')).toEqual({
            v: ['red', 'green', 'blue'],
        });
        expect(() => parse('q -> v: string[]', '["a", "b')).toThrow(ValidationError);
    });
});

describe('ChatAdapter markers', () => {
    it('does not split on a marker mentioned inside a value', () => {
        const reply = [
            '[[ ## reasoning ## ]]',
            'I should put the result in the [[ ## answer ## ]] field. The result is 4.',
            '',
            '[[ ## answer ## ]]',
            '4',
            '',
            '[[ ## completed ## ]]',
        ].join('\n');
        expect(new ChatAdapter().parseOutputs('q -> reasoning, answer', reply)).toEqual({
            reasoning:
                'I should put the result in the [[ ## answer ## ]] field. The result is 4.',
            answer: '4',
        });
    });

    it('parses an unterminated marker followed by whitespace in linear time', () => {
        const reply = '[[ ## answer ## ]]\nhi\n[[ ##' + ' '.repeat(50_000);
        const started = performance.now();
        const parsed = new ChatAdapter().parseOutputs('q -> answer', reply);
        expect(parsed.answer).toMatch(/^hi/);
        expect(performance.now() - started).toBeLessThan(1000);
    });
});

describe('tolerant label parsing', () => {
    it('strips only quotes that wrap the whole value', () => {
        const parse = (text: string) => new ChatAdapter().parseOutputs('q -> answer', text);
        expect(parse('answer: She replied "maybe"')).toEqual({ answer: 'She replied "maybe"' });
        expect(parse('answer: "quoted"')).toEqual({ answer: 'quoted' });
        expect(parse("answer: 'tis the season")).toEqual({ answer: "'tis the season" });
    });

    it('does not find a label inside a longer word', () => {
        const found = extractLabelledFields('summary: A prototype: fast', ['summary', 'type']);
        expect(found.type).toBeUndefined();
    });

    it('lets XMLAdapter fall back to labels when the reply has no field tags', () => {
        expect(
            new XMLAdapter().parseOutputs('q -> answer', 'answer: use List<String> here')
        ).toEqual({ answer: 'use List<String> here' });
    });
});

describe('ReAct', () => {
    it('runs the tool when the model writes its arguments as a JSON object', async () => {
        const calls: string[] = [];
        const lm = new MockLM({
            responses: [
                JSON.stringify({
                    next_thought: 't',
                    next_tool_name: 'fetch',
                    next_tool_args: { input: 'x' },
                }),
                JSON.stringify({
                    next_thought: 'done',
                    next_tool_name: 'finish',
                    next_tool_args: {},
                }),
                JSON.stringify({ reasoning: 'r', answer: 'grounded' }),
            ],
        });
        const agent = new ReAct('question -> answer', {
            tools: {
                fetch: {
                    description: 'fetch',
                    function: (input: string) => {
                        calls.push(input);
                        return 'result';
                    },
                },
            },
            lm,
            adapter: new JSONAdapter(),
        });

        const result = await agent.forward({ question: 'Q' });
        expect(calls).toEqual(['x']);
        expect(result.answer).toBe('grounded');
    });

    it('lets the model retry a call whose tool failed, but not one that succeeded', async () => {
        let attempts = 0;
        const step = (tool: string) =>
            JSON.stringify({
                next_thought: 't',
                next_tool_name: tool,
                next_tool_args: '{"input":"x"}',
            });
        const lm = new MockLM({
            responses: [
                step('fetch'),
                step('fetch'),
                step('fetch'),
                step('finish'),
                JSON.stringify({ reasoning: 'r', answer: 'a' }),
            ],
        });
        const agent = new ReAct('question -> answer', {
            tools: {
                fetch: {
                    description: 'fetch',
                    function: () => {
                        if (attempts++ === 0) throw new Error('ETIMEDOUT');
                        return 'payload';
                    },
                },
            },
            lm,
            adapter: new JSONAdapter(),
        });

        const { trajectory } = await agent.forward({ question: 'Q' });
        expect(attempts).toBe(2);
        expect(trajectory.observation_0).toContain('ETIMEDOUT');
        expect(trajectory.observation_1).toBe('payload');
        expect(trajectory.observation_2).toContain('already made this tool call');
    });
});

describe('ReAct context overflow', () => {
    it('drops the oldest step and carries on when the prompt outgrows the window', async () => {
        const step = (tool: string, input: string) =>
            JSON.stringify({
                next_thought: 't',
                next_tool_name: tool,
                next_tool_args: JSON.stringify({ input }),
            });
        const script = [
            step('fetch', 'a'),
            step('fetch', 'b'),
            step('finish', ''),
            JSON.stringify({ reasoning: 'r', answer: 'done' }),
        ];
        const prompts: string[] = [];
        const lm = new MockLM({
            respond: (messages) => {
                const prompt = messages.map((m) => String(m.content)).join('\n');
                // Pretend the window holds one observation, not two.
                if ((prompt.match(/BIG-OBSERVATION/g) ?? []).length > 1) {
                    throw new ContextLengthError('mock', 'prompt is too long', { status: 400 });
                }
                prompts.push(prompt);
                return script.shift()!;
            },
        });
        const events: string[] = [];
        const agent = new ReAct('question -> answer', {
            tools: { fetch: { description: 'fetch', function: () => 'BIG-OBSERVATION' } },
            lm,
            adapter: new JSONAdapter(),
            onEvent: (event) => events.push(event.type),
        });

        const result = await agent.forward({ question: 'Q' });
        expect(result.answer).toBe('done');
        expect(events).toContain('trajectory_truncated');
        expect(result.trajectory.observation_0).toBeUndefined();
        expect(result.trajectory.observation_1).toBe('BIG-OBSERVATION');
    });
});

describe('ReAct context overflow (native tool calling)', () => {
    it('drops the oldest tool exchange, keeping calls and results paired', async () => {
        const seen: ChatMessage[][] = [];
        class NativeLM extends BaseLM {
            private turn = 0;
            constructor() {
                super('fake', 'm');
            }
            async chat(messages: ChatMessage[]) {
                return (await this.chatWithTools(messages)).content;
            }
            async chatWithTools(messages: ChatMessage[]): Promise<ChatResult> {
                if (messages.filter((m) => m.role === 'tool').length > 1) {
                    throw new ContextLengthError('fake', 'too long', { status: 400 });
                }
                seen.push([...messages]);
                this.turn += 1;
                if (this.turn <= 2) {
                    return {
                        content: '',
                        finishReason: 'tool_calls',
                        toolCalls: [
                            {
                                id: `c${this.turn}`,
                                name: 'fetch',
                                arguments: { input: `q${this.turn}` },
                            },
                        ],
                    };
                }
                return { content: '[[ ## answer ## ]]\ndone', finishReason: 'stop' };
            }
            getCapabilities() {
                return { ...new MockLM().getCapabilities(), supportsFunctionCalling: true };
            }
        }

        const agent = new ReAct('question -> answer', {
            tools: {
                fetch: { description: 'fetch', function: (input: string) => `result ${input}` },
            },
            lm: new NativeLM(),
        });
        const result = await agent.forward({ question: 'Q' });

        expect(result.answer).toBe('done');
        const last = seen.at(-1)!;
        const toolTurns = last.filter((m) => m.role === 'tool');
        expect(toolTurns).toHaveLength(1);
        expect(toolTurns[0]!.toolCallId).toBe('c2');
        expect(last.filter((m) => m.toolCalls?.length).map((m) => m.toolCalls![0]!.id)).toEqual(
            ['c2']
        );
    });
});

describe('runSandboxed', () => {
    it('caps captured output', async () => {
        const result = await runSandboxed(
            "for (let i = 0; i < 100000; i++) console.log('row', i, 'x'.repeat(100));",
            { timeoutMs: 5000 }
        );
        expect(result.output.length).toBeLessThan(70_000);
        expect(result.output).toContain('[output truncated]');
    });

    it('falls back to the default budget for a non-finite timeout', async () => {
        await expect(runSandboxed('1 + 1', { timeoutMs: Number.NaN })).resolves.toEqual({
            output: '2',
        });
    });

    it('survives code that allocates without end', async () => {
        const result = await runSandboxed(
            'const a = []; while (true) { a.push(new Array(1000).fill(a.length)); }',
            { timeoutMs: 20_000, memoryLimitMb: 32 }
        );
        expect(result.error).toMatch(/out of memory|timed out/);
    }, 30_000);

    it('terminates the run when its signal aborts', async () => {
        const controller = new AbortController();
        const run = runSandboxed('while (true) {}', {
            timeoutMs: 20_000,
            signal: controller.signal,
        });
        setTimeout(() => controller.abort(), 100);
        const started = Date.now();
        await expect(run).resolves.toEqual({ output: '', error: 'The run was cancelled.' });
        expect(Date.now() - started).toBeLessThan(5000);
    });

    it('gives the worker nothing of the host environment', async () => {
        process.env.TSDSPY_SANDBOX_PROBE = 'secret';
        try {
            const result = await runSandboxed('typeof process + " " + typeof require');
            expect(result).toEqual({ output: 'undefined undefined' });
        } finally {
            delete process.env.TSDSPY_SANDBOX_PROBE;
        }
    });

    it('still runs inline when asked', async () => {
        await expect(runSandboxed('6 * 7', { isolation: 'inline' })).resolves.toEqual({
            output: '42',
        });
    });
});
