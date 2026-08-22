import { Predict } from '../modules/predict';
import { Signature, InputField, OutputField } from './signature';
import { ValidationError } from './errors';
import { MockLM } from '../test-utils';

class QA extends Signature {
    static description = 'Answer a question';

    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;

    @OutputField({ description: 'confidence 0-1', type: 'number' })
    confidence!: number;
}

const questions = (count: number) =>
    Array.from({ length: count }, (_, index) => ({ question: `q${index}` }));

const replies = (count: number) =>
    Array.from({ length: count }, (_, index) => `answer: a${index}\nconfidence: 0.9`);

/** A reply whose `confidence` cannot be coerced, so the input fails validation. */
const badReply = 'answer: a1\nconfidence: very high';

describe('Module.batch', () => {
    it('returns one result per input, in input order', async () => {
        const lm = new MockLM({ responses: replies(5) });
        const predict = new Predict(QA, lm);

        const results = await predict.batch(questions(5), { concurrency: 3 });

        expect(results).toHaveLength(5);
        expect(
            results.map((result) => result.status === 'fulfilled' && result.value.answer)
        ).toEqual(['a0', 'a1', 'a2', 'a3', 'a4']);
    });

    it('holds input order even when later calls finish first', async () => {
        const lm = new MockLM({ responses: replies(3) });
        // Reverse the latency so the last call settles first.
        const delays = [30, 15, 0];
        const chat = lm.chat.bind(lm);
        let issued = 0;
        vi.spyOn(lm, 'chat').mockImplementation(async (messages, options) => {
            const delay = delays[issued++] ?? 0;
            const reply = await chat(messages, options);
            await new Promise((resolve) => setTimeout(resolve, delay));
            return reply;
        });

        const results = await new Predict(QA, lm).batch(questions(3), { concurrency: 3 });

        expect(
            results.map((result) => result.status === 'fulfilled' && result.value.answer)
        ).toEqual(['a0', 'a1', 'a2']);
    });

    it('captures a failing input instead of destroying the batch', async () => {
        // The second reply carries an uncoercible `confidence`, so only that input fails.
        const lm = new MockLM({
            responses: ['answer: a0\nconfidence: 0.9', badReply, 'answer: a2\nconfidence: 0.9'],
        });

        const results = await new Predict(QA, lm).batch(questions(3));

        expect(results[0]!.status).toBe('fulfilled');
        expect(results[2]!.status).toBe('fulfilled');
        expect(results[1]!.status).toBe('rejected');
        expect(results[1]!.status === 'rejected' && results[1]!.reason).toBeInstanceOf(
            ValidationError
        );
    });

    it('rejects the whole batch on the first failure when stopOnError is set', async () => {
        const lm = new MockLM({ responses: [badReply, ...replies(4)] });

        await expect(
            new Predict(QA, lm).batch(questions(5), { concurrency: 1, stopOnError: true })
        ).rejects.toThrow(ValidationError);
    });

    it('reports progress as inputs settle', async () => {
        const lm = new MockLM({ responses: replies(4) });
        const progress: Array<[number, number]> = [];

        await new Predict(QA, lm).batch(questions(4), {
            concurrency: 2,
            onProgress: (done, total) => progress.push([done, total]),
        });

        expect(progress).toEqual([
            [1, 4],
            [2, 4],
            [3, 4],
            [4, 4],
        ]);
    });

    it('caps the calls in flight at the requested concurrency', async () => {
        const lm = new MockLM({ responses: replies(12) });
        let inFlight = 0;
        let peak = 0;
        const chat = lm.chat.bind(lm);
        vi.spyOn(lm, 'chat').mockImplementation(async (messages, options) => {
            inFlight += 1;
            peak = Math.max(peak, inFlight);
            const reply = await chat(messages, options);
            await new Promise((resolve) => setTimeout(resolve, 2));
            inFlight -= 1;
            return reply;
        });

        await new Predict(QA, lm).batch(questions(12), { concurrency: 3 });

        expect(peak).toBe(3);
        expect(inFlight).toBe(0);
    });

    it('passes the remaining call options through to every call', async () => {
        const lm = new MockLM({ responses: replies(2) });

        await new Predict(QA, lm).batch(questions(2), {
            concurrency: 2,
            temperature: 0,
            timeout: 5_000,
        });

        expect(lm.calls).toHaveLength(2);
        for (const call of lm.calls) {
            expect(call.options).toEqual({ temperature: 0, timeout: 5_000 });
        }
    });

    it('stops issuing calls once the signal aborts', async () => {
        const lm = new MockLM({ responses: replies(20) });
        const controller = new AbortController();
        const chat = lm.chat.bind(lm);
        vi.spyOn(lm, 'chat').mockImplementation(async (messages, options) => {
            if (lm.calls.length >= 2) controller.abort();
            return chat(messages, options);
        });

        const run = new Predict(QA, lm).batch(questions(20), {
            concurrency: 2,
            signal: controller.signal,
        });

        await expect(run).rejects.toThrow();
        expect(lm.calls.length).toBeLessThan(20);
    });

    it('throws on a non-positive concurrency rather than hanging', async () => {
        const lm = new MockLM({ responses: replies(2) });

        await expect(
            new Predict(QA, lm).batch(questions(2), { concurrency: 0 })
        ).rejects.toThrow(RangeError);

        expect(lm.calls).toHaveLength(0);
    });

    it('resolves to an empty array for no inputs', async () => {
        const lm = new MockLM({ responses: [] });

        expect(await new Predict(QA, lm).batch([])).toEqual([]);
    });
});
