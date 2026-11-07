import { ProgramOfThought, extractCode, runSandboxed } from './program-of-thought';
import { TsDspyError } from '../core/errors';
import { MockLM } from '../test-utils';
import { contentToText } from '../utils/content';
import type { ChatMessage } from '../types/language-model';

const code = (source: string) =>
    `[[ ## reasoning ## ]]\nCompute it.\n\n[[ ## generated_code ## ]]\n${source}\n\n[[ ## completed ## ]]`;
const final = (value: string) =>
    `[[ ## reasoning ## ]]\nThe code printed it.\n\n[[ ## answer ## ]]\n${value}\n\n[[ ## completed ## ]]`;
const text = (messages: ChatMessage[]) =>
    messages.map((message) => contentToText(message.content)).join('\n');

describe('ProgramOfThought', () => {
    it('generates code, runs it, and answers from its output', async () => {
        const lm = new MockLM({
            responses: [code('```js\nconsole.log(6 * 7);\n```'), final('42')],
        });

        const result = await new ProgramOfThought('question -> answer: int', { lm }).forward({
            question: 'What is 6 times 7?',
        });

        expect(result.answer).toBe(42);
        expect(lm.calls).toHaveLength(2);
        const answerPrompt = text(lm.calls[1].messages);
        expect(answerPrompt).toContain('[[ ## code_output ## ]]\n42');
        expect(answerPrompt).toContain('console.log(6 * 7);');
    });

    it('feeds an error back for regeneration', async () => {
        const lm = new MockLM({
            responses: [code('undefinedFunction()'), code('1 + 1'), final('2')],
        });

        const result = await new ProgramOfThought('question -> answer', { lm }).forward({
            question: 'Q',
        });

        expect(result.answer).toBe('2');
        const regenerate = text(lm.calls[1].messages);
        expect(regenerate).toContain('[[ ## previous_code ## ]]\nundefinedFunction()');
        expect(regenerate).toMatch(/ReferenceError: undefinedFunction is not defined/);
    });

    it('throws after maxIters failed generations', async () => {
        const lm = new MockLM({ respond: () => code('throw new Error("nope")') });

        await expect(
            new ProgramOfThought('question -> answer', { lm, maxIters: 2 }).forward({
                question: 'Q',
            })
        ).rejects.toThrow(TsDspyError);
        expect(lm.calls).toHaveLength(2);
    });

    it('exposes its three predictors', () => {
        const pot = new ProgramOfThought('question -> answer');
        expect(pot.namedPredictors().map(([path]) => path)).toEqual([
            'codeGenerate',
            'codeRegenerate',
            'generateOutput',
        ]);
    });
});

describe('runSandboxed', () => {
    it('captures logs and the completion value', async () => {
        expect(await runSandboxed('console.log("a"); console.log({ b: 1 }); 40 + 2')).toEqual({
            output: 'a\n{"b":1}\n42',
        });
    });

    it('has no require, process, or string evaluation', async () => {
        expect((await runSandboxed('require("fs")')).error).toMatch(/require is not defined/);
        expect((await runSandboxed('process.exit(1)')).error).toMatch(/process is not defined/);
        expect((await runSandboxed('eval("1")')).error).toMatch(/EvalError|disallowed/i);
    });

    it('cuts off code that runs too long', async () => {
        const result = await runSandboxed('while (true) {}', { timeoutMs: 50 });
        expect(result.error).toMatch(/timed out/i);

        const logged = await runSandboxed('console.log("before"); while (true) {}', {
            timeoutMs: 50,
        });
        expect(logged).toEqual({
            output: 'before',
            // vm reports the remaining budget, which can dip below 50ms on a slow runner.
            error: expect.stringMatching(/^Error: Script execution timed out after \d+ms$/),
        });
    });

    it('awaits a promise completion value within the budget', async () => {
        expect(await runSandboxed('Promise.resolve(7)')).toEqual({ output: '7' });
        const pending = await runSandboxed('new Promise(() => {})', { timeoutMs: 30 });
        expect(pending.error).toMatch(/timed out/i);
    });

    it('cuts off a runaway microtask instead of hanging the host', async () => {
        const result = await runSandboxed(
            'Promise.resolve().then(() => { while (true) {} }); 1',
            { timeoutMs: 50 }
        );
        expect(result.error).toMatch(/timed out/i);
    });

    it('reports async results and rejections', async () => {
        expect(
            await runSandboxed('(async () => { await null; console.log("x"); return 42; })()')
        ).toEqual({ output: 'x\n42' });
        expect((await runSandboxed('(async () => { throw new Error("boom"); })()')).error).toBe(
            'Error: boom'
        );
    });

    it('does not expose host functions through console', async () => {
        const result = await runSandboxed('typeof console.log.constructor("return process")()');
        expect(result.error).toMatch(/disallowed/i);
    });

    it('runs a hostile then() getter inside the timeout', async () => {
        const result = await runSandboxed('({ get then() { while (true) {} } })', {
            timeoutMs: 50,
        });
        expect(result.error).toMatch(/timed out/i);
    });

    it('reports code that produced nothing, or no code at all', async () => {
        expect((await runSandboxed('const x = 1;')).error).toMatch(/no output/);
        expect((await runSandboxed('   ')).error).toMatch(/No code/);
    });
});

describe('extractCode', () => {
    it('takes the first fenced block, or the whole reply', () => {
        expect(extractCode('Here:\n```javascript\nconst a = 1;\na\n```\nDone')).toBe(
            'const a = 1;\na'
        );
        expect(extractCode('  1 + 1  ')).toBe('1 + 1');
    });
});
