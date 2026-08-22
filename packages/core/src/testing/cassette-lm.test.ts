import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CassetteLM } from './cassette-lm';
import { hashRequest } from './cassette';
import { MockLM } from './mock-lm';
import { LMError } from '../core/errors';
import { Predict } from '../modules/predict';
import { Signature, InputField, OutputField } from '../core/signature';
import type { ChatMessage } from '../types/language-model';

class QA extends Signature {
    static description = 'Answer a question';

    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;

    @OutputField({ description: 'confidence 0-1', type: 'number' })
    confidence!: number;
}

let directory: string;
let cassette: string;

beforeEach(() => {
    directory = mkdtempSync(join(tmpdir(), 'ts-dspy-cassette-'));
    cassette = join(directory, 'nested', 'qa.json');
});

afterEach(() => {
    rmSync(directory, { recursive: true, force: true });
});

describe('CassetteLM', () => {
    describe('record', () => {
        it("captures the wrapped model's replies to disk", async () => {
            const live = new MockLM({ responses: ['answer: Paris\nconfidence: 0.9'] });
            const lm = CassetteLM.record(cassette, live);

            const reply = await lm.generate('Capital of France?');

            expect(reply).toBe('answer: Paris\nconfidence: 0.9');
            expect(existsSync(cassette)).toBe(true);
            expect(lm.entries).toHaveLength(1);
        });

        it('writes a diffable array of keyed request/response entries', async () => {
            const live = new MockLM({ responses: ['hello'] });
            await CassetteLM.record(cassette, live).generate('hi', { temperature: 0 });

            const written = JSON.parse(readFileSync(cassette, 'utf8'));

            expect(Array.isArray(written)).toBe(true);
            expect(written[0].request).toEqual({
                kind: 'chat',
                model: 'mock-model',
                messages: [{ role: 'user', content: 'hi' }],
                options: { temperature: 0 },
            });
            expect(written[0].response).toBe('hello');
            expect(written[0].key).toBe(hashRequest(written[0].request));
        });

        it('rewrites the file rather than appending to an old take', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['one'] })).generate('q');
            await CassetteLM.record(cassette, new MockLM({ responses: ['two'] })).generate('q');

            const written = JSON.parse(readFileSync(cassette, 'utf8'));

            expect(written).toHaveLength(1);
            expect(written[0].response).toBe('two');
        });

        it('leaves the file alone when autoSave is off until save() is called', async () => {
            const lm = new CassetteLM({
                path: cassette,
                mode: 'record',
                lm: new MockLM({ responses: ['one'] }),
                autoSave: false,
            });

            await lm.generate('q');
            expect(existsSync(cassette)).toBe(false);

            lm.save();
            expect(existsSync(cassette)).toBe(true);
        });

        it('refuses to record without a live model', () => {
            expect(() => new CassetteLM({ path: cassette, mode: 'record' })).toThrow(LMError);
        });
    });

    describe('replay', () => {
        it('returns recorded replies without a live model', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['recorded'] })).generate(
                'q'
            );

            const replayed = await CassetteLM.replay(cassette).generate('q');

            expect(replayed).toBe('recorded');
        });

        it('drives a module identically on the recorded and replayed run', async () => {
            const live = new MockLM({ responses: ['answer: Paris\nconfidence: 0.9'] });
            const recorded = await new Predict(QA, CassetteLM.record(cassette, live)).forward({
                question: 'Capital of France?',
            });

            const replayed = await new Predict(QA, CassetteLM.replay(cassette)).forward({
                question: 'Capital of France?',
            });

            expect(replayed.answer).toBe(recorded.answer);
            expect(replayed.confidence).toBe(0.9);
        });

        it('reports an unrecorded request with the key and the prompt', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['recorded'] })).generate(
                'q'
            );
            const lm = CassetteLM.replay(cassette);

            await expect(lm.generate('a different question')).rejects.toThrow(
                /No recorded chat response \(key [0-9a-f]{16}\).*a different question/s
            );
        });

        it('reports a missing cassette file', () => {
            expect(() => CassetteLM.replay(join(directory, 'absent.json'))).toThrow(
                /No cassette at/
            );
        });

        it('replays repeated identical requests in recorded order', async () => {
            const live = new MockLM({ responses: ['first', 'second'] });
            const recorder = CassetteLM.record(cassette, live);
            await recorder.generate('q');
            await recorder.generate('q');

            const lm = CassetteLM.replay(cassette);

            expect(await lm.generate('q')).toBe('first');
            expect(await lm.generate('q')).toBe('second');
            expect(await lm.generate('q')).toBe('second');
        });

        it('counts replayed calls in the usage stats', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['recorded'] })).generate(
                'q'
            );
            const lm = CassetteLM.replay(cassette);

            await lm.generate('q');

            expect(lm.getUsage().requestCount).toBe(1);
        });

        it('takes its model name from the cassette', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['x'] })).generate('q');

            expect(CassetteLM.replay(cassette).getModelName()).toBe('mock-model');
        });

        it('picks up a hand-edited request without a rewritten key', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['recorded'] })).generate(
                'q'
            );
            const written = JSON.parse(readFileSync(cassette, 'utf8'));
            written[0].request.messages[0].content = 'edited';
            written[0].key = 'staleeeeeeeeeeee';
            writeFileSync(cassette, JSON.stringify(written, null, 2));

            expect(await CassetteLM.replay(cassette).generate('edited')).toBe('recorded');
        });

        it('hands back a fresh copy of a structured reply each time', async () => {
            const live = new MockLM({
                structuredResponses: [{ answer: 'Paris', confidence: 0.9 }],
                capabilities: { supportsStructuredOutput: true },
            });
            await CassetteLM.record(cassette, live).generateStructured('Capital?', {});
            const lm = CassetteLM.replay(cassette);

            const first = await lm.generateStructured<Record<string, unknown>>('Capital?', {});
            first.answer = 'Berlin';
            const second = await lm.generateStructured<Record<string, unknown>>('Capital?', {});

            expect(second.answer).toBe('Paris');
        });

        it('rejects a cassette that is not an array of entries', () => {
            const malformed = join(directory, 'malformed.json');
            writeFileSync(malformed, '{"entries":[]}');

            expect(() => CassetteLM.replay(malformed)).toThrow(/must contain an array/);
        });
    });

    describe('auto', () => {
        it('replays what it has and records the rest', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['known'] })).generate(
                'q'
            );

            const live = new MockLM({ responses: ['fresh'] });
            const lm = new CassetteLM({ path: cassette, mode: 'auto', lm: live });

            expect(await lm.generate('q')).toBe('known');
            expect(await lm.generate('new question')).toBe('fresh');
            expect(await lm.generate('new question')).toBe('fresh');

            expect(live.calls).toHaveLength(1);
            expect(JSON.parse(readFileSync(cassette, 'utf8'))).toHaveLength(2);
        });

        it('starts a cassette that does not exist yet', async () => {
            const lm = new CassetteLM({
                path: cassette,
                mode: 'auto',
                lm: new MockLM({ responses: ['fresh'] }),
            });

            expect(await lm.generate('q')).toBe('fresh');
            expect(existsSync(cassette)).toBe(true);
        });
    });

    describe('request keys', () => {
        it('separates requests that differ in sampling options', async () => {
            const live = new MockLM({ responses: ['cold', 'hot'] });
            const recorder = CassetteLM.record(cassette, live);
            await recorder.generate('q', { temperature: 0 });
            await recorder.generate('q', { temperature: 1 });

            const lm = CassetteLM.replay(cassette);

            expect(await lm.generate('q', { temperature: 1 })).toBe('hot');
            expect(await lm.generate('q', { temperature: 0 })).toBe('cold');
        });

        it('keeps a recorded turn intact when the caller grows the message list', async () => {
            const live = new MockLM({ responses: ['first', 'second'] });
            const recorder = CassetteLM.record(cassette, live);
            const messages: ChatMessage[] = [{ role: 'user', content: 'one' }];

            await recorder.chat(messages);
            messages.push({ role: 'assistant', content: 'first' });
            messages.push({ role: 'user', content: 'two' });
            await recorder.chat(messages);

            const lm = CassetteLM.replay(cassette);

            expect(await lm.chat([{ role: 'user', content: 'one' }])).toBe('first');
            expect(await lm.chat(messages)).toBe('second');
        });

        it('ignores transport options that cannot change a reply', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['recorded'] })).generate(
                'q',
                { timeout: 1000 }
            );

            const replayed = await CassetteLM.replay(cassette).generate('q', { retries: 5 });

            expect(replayed).toBe('recorded');
        });
    });

    describe('structured output', () => {
        it('records and replays a structured call', async () => {
            const live = new MockLM({
                structuredResponses: [{ answer: 'Paris', confidence: 0.9 }],
                capabilities: { supportsStructuredOutput: true },
            });
            const recorder = CassetteLM.record(cassette, live);
            expect(recorder.getCapabilities().supportsStructuredOutput).toBe(true);

            const predicted = await new Predict(QA, recorder).forward({ question: 'Capital?' });
            const lm = CassetteLM.replay(cassette);
            const replayed = await new Predict(QA, lm).forward({ question: 'Capital?' });

            expect(predicted.answer).toBe('Paris');
            expect(replayed.answer).toBe('Paris');
            expect(lm.getCapabilities().supportsStructuredOutput).toBe(true);
            expect(lm.entries[0].request.kind).toBe('structured');
        });
    });

    describe('capabilities', () => {
        it('never claims to stream', () => {
            const live = new MockLM({ capabilities: { supportsStreaming: true } });

            expect(CassetteLM.record(cassette, live).getCapabilities().supportsStreaming).toBe(
                false
            );
        });

        it('accepts an explicit override', async () => {
            await CassetteLM.record(cassette, new MockLM({ responses: ['x'] })).generate('q');

            const lm = CassetteLM.replay(cassette, { capabilities: { maxContextLength: 42 } });

            expect(lm.getCapabilities().maxContextLength).toBe(42);
        });
    });
});
