import { z } from 'zod';
import { Tool } from './tool';
import { TsDspyError } from './errors';
import { cosineSimilarity, type Embedder } from './embedder';
import { RespAct } from '../modules/respact';
import { MockLM } from '../test-utils';

describe('Tool', () => {
    it('wraps a named function with a default description and schema', async () => {
        async function lookup(args: { id: string }) {
            return `item ${args.id}`;
        }
        const tool = Tool.fromFunction(lookup);

        expect(tool.name).toBe('lookup');
        expect(tool.description).toBe('Tool: lookup');
        expect(tool.toSpec()).toEqual({
            name: 'lookup',
            description: 'Tool: lookup',
            parameters: { type: 'object', properties: {} },
        });
        expect(await tool.call({ id: '7' })).toBe('item 7');
    });

    it('derives a JSON Schema from Zod and validates arguments', async () => {
        const tool = Tool.fromFunction(({ a, b }: { a: number; b: number }) => a + b, {
            name: 'add',
            description: 'Add two numbers',
            parameters: z.object({ a: z.number(), b: z.number() }),
        });

        expect(tool.parameters).toMatchObject({
            type: 'object',
            properties: { a: { type: 'number' }, b: { type: 'number' } },
        });
        expect(await tool.call({ a: 2, b: 3 })).toBe(5);
        await expect(tool.call({ a: 'two', b: 3 })).rejects.toThrow(TsDspyError);
        await expect(tool.call({ a: 'two', b: 3 })).rejects.toThrow(/Invalid arguments: a:/);
    });

    it('passes a JSON Schema through untouched, without validating', async () => {
        const parameters = { type: 'object', properties: { q: { type: 'string' } } };
        const tool = Tool.fromFunction((args: { q: string }) => args, {
            name: 'echo',
            parameters,
        });

        expect(tool.parameters).toBe(parameters);
        expect(await tool.call({ q: 1 })).toEqual({ q: 1 });
    });

    it('requires a name for an anonymous function', () => {
        expect(() => Tool.fromFunction(() => 1)).toThrow(/needs a name/);
    });

    it('is accepted by RespAct as a record value or in an array', async () => {
        const add = Tool.fromFunction(({ a, b }: { a: number; b: number }) => a + b, {
            name: 'add',
            parameters: z.object({ a: z.number(), b: z.number() }),
        });
        const lm = new MockLM({
            responses: [
                'Action: add\nAction Input: {"a": 2, "b": 3}',
                'Final Answer: answer: 5',
                'Action: plus\nAction Input: {"a": 1, "b": 1}',
                'Final Answer: answer: 2',
            ],
        });

        const viaArray = await new RespAct('question -> answer', { lm, tools: [add] }).forward({
            question: '2+3?',
        });
        expect(viaArray.answer).toBe('5');
        expect(lm.calls[1].messages[0].content).toContain('Observation: 5');

        // A record key renames the tool.
        const viaRecord = await new RespAct('question -> answer', {
            lm,
            tools: { plus: add },
        }).forward({ question: '1+1?' });
        expect(viaRecord.answer).toBe('2');
        expect(lm.calls[3].messages[0].content).toContain('Observation: 2');
        expect(add.name).toBe('add');
    });
});

describe('embedder contract', () => {
    it('computes cosine similarity', () => {
        expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
        expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
        expect(cosineSimilarity([1, 2], [-1, -2])).toBeCloseTo(-1);
        expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
        expect(() => cosineSimilarity([1], [1, 2])).toThrow(RangeError);
    });

    it('is implementable in a few lines', async () => {
        const embedder: Embedder = {
            dimensions: 2,
            embed: async (texts) => texts.map((text) => [text.length, 1]),
        };
        const [a, b] = await embedder.embed(['hi', 'hello']);
        expect(cosineSimilarity(a, b)).toBeGreaterThan(0.9);
    });
});
