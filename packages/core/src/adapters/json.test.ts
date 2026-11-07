import { JSONAdapter, flattenMessages } from './json';
import { Predict } from '../modules/predict';
import { Signature, InputField, OutputField, ImageField } from '../core/signature';
import { Example } from '../core/example';
import { ValidationError } from '../core/errors';
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

const adapter = new JSONAdapter();

describe('JSONAdapter.format', () => {
    it('puts the JSON schema in the system message and the inputs in the user turn', () => {
        const [system, user] = adapter.format({ signature: QA, inputs: { question: 'Q?' } });

        expect(system.content).toContain('2. `confidence` (number): confidence 0-1');
        expect(system.content).toContain('Respond with JSON matching this schema');
        expect(system.content).toContain('"required": [\n    "answer",\n    "confidence"\n  ]');
        expect(system.content).toContain('your objective is:\nAnswer a question');
        expect(user.content).toBe(
            '[[ ## question ## ]]\nQ?\n\n' +
                'Respond with a JSON object in the following order of fields: `answer`, then `confidence` (must be a number).'
        );
    });

    it('renders demo replies as JSON objects', () => {
        const messages = adapter.format({
            signature: QA,
            inputs: { question: 'Q?' },
            demos: [new Example({ question: 'D?', answer: 'A', confidence: 1 })],
        });

        expect(messages[2]).toEqual({
            role: 'assistant',
            content: '{"answer":"A","confidence":1}',
        });
    });
});

describe('JSONAdapter.parse', () => {
    it('reads a JSON object, tolerating fences and prose', () => {
        expect(adapter.parse(QA, 'Here:\n```json\n{"answer": "Paris"}\n```')).toEqual({
            answer: 'Paris',
        });
    });

    it('reports a reply that is not a JSON object as a validation failure', () => {
        expect(() => adapter.parse(QA, 'Paris')).toThrow(ValidationError);
        expect(() => adapter.parse(QA, '[1, 2]')).toThrow(ValidationError);
    });

    it('treats nulls as absent optional fields when validating', () => {
        class Opt extends Signature {
            @OutputField({ description: 'answer' })
            answer!: string;

            @OutputField({ description: 'note', required: false })
            note?: string;
        }

        expect(adapter.parseOutputs(Opt, '{"answer": "a", "note": null}')).toEqual({
            answer: 'a',
        });
    });

    it('reads a truncated object while streaming', () => {
        expect(adapter.parsePartial(QA, '{"answer": "Pa')).toEqual({ answer: 'Pa' });
        expect(adapter.parsePartial(QA, 'not json yet')).toEqual({});
    });
});

describe('JSONAdapter.call', () => {
    it('uses native structured output when the model has it', async () => {
        const lm = new MockLM({
            structuredResponses: [{ answer: 'Paris', confidence: 0.9 }],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await adapter.call(lm, { signature: QA, inputs: { question: 'Q' } });

        expect(result).toEqual({ answer: 'Paris', confidence: 0.9 });
        expect(lm.structuredCalls).toHaveLength(1);
        expect(lm.structuredCalls[0].prompt).toContain('[[ ## question ## ]]\nQ');
    });

    it('asks for JSON in a chat call when the model has no structured output', async () => {
        const lm = new MockLM({ responses: ['{"answer": "Paris", "confidence": "0.9"}'] });

        const result = await adapter.call(lm, { signature: QA, inputs: { question: 'Q' } });

        expect(result).toEqual({ answer: 'Paris', confidence: 0.9 });
        expect(lm.structuredCalls).toHaveLength(0);
    });

    it('falls back to chat for image prompts, which structured calls cannot carry', async () => {
        class Caption extends Signature {
            @ImageField()
            photo!: string;

            @OutputField()
            caption!: string;
        }
        const lm = new MockLM({
            responses: ['{"caption": "a cat"}'],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await adapter.call(lm, {
            signature: Caption,
            inputs: { photo: 'https://example.com/cat.png' },
        });

        expect(result.caption).toBe('a cat');
        expect(lm.structuredCalls).toHaveLength(0);
        expect(Array.isArray(lm.calls[0].messages.at(-1)?.content)).toBe(true);
    });

    it('is selected by default for a model with structured output', async () => {
        const lm = new MockLM({
            structuredResponses: [{ answer: 'Paris', confidence: 0.9 }],
            capabilities: { supportsStructuredOutput: true },
        });

        await new Predict(QA, lm).forward({ question: 'Q' });

        expect(lm.structuredCalls).toHaveLength(1);
    });
});

describe('flattenMessages', () => {
    it('keeps roles when the conversation has more than one turn', () => {
        expect(
            flattenMessages([
                { role: 'system', content: 'S' },
                { role: 'user', content: 'U1' },
                { role: 'assistant', content: 'A1' },
                { role: 'user', content: 'U2' },
            ])
        ).toBe('S\n\nUser:\nU1\n\nAssistant:\nA1\n\nUser:\nU2');
        expect(
            flattenMessages([
                { role: 'system', content: 'S' },
                { role: 'user', content: 'U' },
            ])
        ).toBe('S\n\nU');
    });
});
