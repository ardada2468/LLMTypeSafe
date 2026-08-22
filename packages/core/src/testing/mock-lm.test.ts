import { MockLM } from './mock-lm';

describe('MockLM', () => {
    it('returns scripted replies in order and records every call', async () => {
        const lm = new MockLM({ responses: ['first', 'second'] });

        expect(await lm.generate('a')).toBe('first');
        expect(await lm.chat([{ role: 'user', content: 'b' }])).toBe('second');

        expect(lm.calls).toHaveLength(2);
        expect(lm.lastPrompt()).toBe('b');
    });

    it('records each turn as it was sent, not as the list ends up', async () => {
        const lm = new MockLM({ responses: ['one', 'two'] });
        const messages = [{ role: 'user' as const, content: 'first' }];

        await lm.chat(messages);
        messages.push({ role: 'user' as const, content: 'second' });
        await lm.chat(messages);

        expect(lm.calls[0].messages).toEqual([{ role: 'user', content: 'first' }]);
        expect(lm.calls[1].messages).toHaveLength(2);
    });

    it('throws once the script runs out', async () => {
        const lm = new MockLM({ responses: [] });

        await expect(lm.generate('a')).rejects.toThrow('no more scripted responses');
    });

    describe('streaming', () => {
        it('reports streaming support', () => {
            expect(new MockLM().getCapabilities().supportsStreaming).toBe(true);
        });

        it('streams the scripted reply in word-sized pieces', async () => {
            const lm = new MockLM({ responses: ['answer: Paris now'] });

            const chunks = [];
            for await (const chunk of lm.chatStream([{ role: 'user', content: 'q' }])) {
                chunks.push(chunk);
            }

            expect(chunks.map((chunk) => chunk.content).join('')).toBe('answer: Paris now');
            expect(chunks.length).toBeGreaterThan(1);
            expect(chunks.at(-1)).toEqual({
                content: '',
                done: true,
                usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
            });
            expect(chunks.slice(0, -1).every((chunk) => chunk.done === false)).toBe(true);
        });

        it('honours an explicit chunk size', async () => {
            const lm = new MockLM({ responses: ['abcde'], chunkSize: 2 });

            const contents = [];
            for await (const chunk of lm.generateStream('q')) {
                contents.push(chunk.content);
            }

            expect(contents).toEqual(['ab', 'cd', 'e', '']);
        });

        it('records a streamed call like any other', async () => {
            const lm = new MockLM({ responses: ['hi'] });

            for await (const _chunk of lm.generateStream('what?', { temperature: 0.1 })) {
                // drain
            }

            expect(lm.lastPrompt()).toBe('what?');
            expect(lm.calls[0].options).toEqual({ temperature: 0.1 });
            expect(lm.getUsage().totalTokens).toBe(15);
        });

        it('emits only the terminating chunk for an empty reply', async () => {
            const lm = new MockLM({ responses: [''] });

            const contents = [];
            for await (const chunk of lm.generateStream('q')) {
                contents.push(chunk.content);
            }

            expect(contents).toEqual(['']);
        });
    });
});
