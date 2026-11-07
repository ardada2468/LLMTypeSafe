import { Parallel, ParallelError } from './parallel';
import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { context } from '../core/config';
import { Predict } from './predict';
import { MockLM } from '../test-utils';
import type { LLMCallOptions } from '../types/language-model';

/** Echoes its input after `delay` ms, or throws when the input says to. */
class Echo extends Module {
    started = 0;

    async forward(inputs: Record<string, any>, _options?: LLMCallOptions): Promise<Prediction> {
        this.started += 1;
        await new Promise((resolve) => setTimeout(resolve, inputs.delay ?? 0));
        if (inputs.fail) throw new Error(`failed ${inputs.value}`);
        return new Prediction({ value: inputs.value });
    }
}

describe('Parallel', () => {
    it('returns results in input order, whatever order they finish in', async () => {
        const echo = new Echo();
        const results = await new Parallel({ concurrency: 3 }).run([
            [echo, { value: 'a', delay: 15 }],
            [echo, { value: 'b', delay: 1 }],
            [echo, { value: 'c', delay: 5 }],
        ]);

        expect(results.map((result) => result?.value)).toEqual(['a', 'b', 'c']);
    });

    it('runs different modules in one batch, with per-pair options', async () => {
        const lm = new MockLM({
            respond: (_messages, options) =>
                `[[ ## answer ## ]]\nt=${options?.temperature}\n\n[[ ## completed ## ]]`,
        });
        const results = await new Parallel().run([
            [new Predict('question -> answer', lm), { question: 'x' }, { temperature: 0.1 }],
            [new Echo(), { value: 'echo' }],
        ]);

        expect(results[0]?.answer).toBe('t=0.1');
        expect(results[1]?.value).toBe('echo');
    });

    it('leaves undefined in the slot of a failed pair', async () => {
        const echo = new Echo();
        const results = await new Parallel().run([
            [echo, { value: 'a' }],
            [echo, { value: 'b', fail: true }],
        ]);

        expect(results).toHaveLength(2);
        expect(results[0]?.value).toBe('a');
        expect(results[1]).toBeUndefined();
    });

    it('reports failures with their inputs when returnFailed is set', async () => {
        const echo = new Echo();
        const { results, failed } = await new Parallel({ returnFailed: true }).run([
            [echo, { value: 'a', fail: true, delay: 5 }],
            [echo, { value: 'b' }],
            [echo, { value: 'c', fail: true }],
        ]);

        expect(results.map((result) => result?.value)).toEqual([undefined, 'b', undefined]);
        expect(failed.map((failure) => failure.index)).toEqual([0, 2]);
        expect(failed[0]!.inputs.value).toBe('a');
        expect(String(failed[1]!.error)).toContain('failed c');
    });

    it('stops starting work and rejects once maxErrors pairs have failed', async () => {
        const echo = new Echo();
        const pairs = Array.from(
            { length: 10 },
            (_, index) => [echo, { value: index, fail: true }] as const
        );

        const run = new Parallel({ concurrency: 1, maxErrors: 2 }).run(pairs);

        await expect(run).rejects.toBeInstanceOf(ParallelError);
        await run.catch((error: ParallelError) => {
            expect(error.failed).toHaveLength(2);
            expect(String((error.cause as Error).message)).toBe('failed 1');
        });
        expect(echo.started).toBe(2);
    });

    it('stops on the caller’s abort signal', async () => {
        const controller = new AbortController();
        controller.abort(new Error('cancelled'));

        await expect(
            new Parallel({ signal: controller.signal }).run([[new Echo(), { value: 1 }]])
        ).rejects.toThrow('cancelled');
    });

    it('runs every pair in the caller’s settings scope', async () => {
        const lm = new MockLM({
            respond: () => '[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]',
        });
        const predict = new Predict('question -> answer');

        const results = await context({ lm }, () =>
            new Parallel().run([
                [predict, { question: '1' }],
                [predict, { question: '2' }],
            ])
        );

        expect(results.map((result) => result?.answer)).toEqual(['ok', 'ok']);
        expect(lm.calls).toHaveLength(2);
    });

    it('validates its options', () => {
        expect(() => new Parallel({ concurrency: 0 })).toThrow(RangeError);
        expect(() => new Parallel({ maxErrors: 0 })).toThrow(RangeError);
    });
});
