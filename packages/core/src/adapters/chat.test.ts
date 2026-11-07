import { z } from 'zod';
import { ChatAdapter } from './chat';
import { Predict } from '../modules/predict';
import { Signature, InputField, OutputField, ImageField, signature } from '../core/signature';
import { Example } from '../core/example';
import { ValidationError } from '../core/errors';
import { contentToText } from '../utils/content';
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

const adapter = new ChatAdapter();

describe('ChatAdapter.format', () => {
    it('writes the field listing, structure and objective into the system message', () => {
        const [system] = adapter.format({ signature: QA, inputs: { question: 'Q' } });

        expect(system.role).toBe('system');
        expect(system.content).toBe(
            [
                'Your input fields are:',
                '1. `question` (string): the question',
                'Your output fields are:',
                '1. `answer` (string): the answer',
                '2. `confidence` (number): confidence 0-1',
                '',
                'All interactions will be structured in the following way, with the appropriate values filled in.',
                '',
                '[[ ## question ## ]]',
                '{question}',
                '',
                '[[ ## answer ## ]]',
                '{answer}',
                '',
                '[[ ## confidence ## ]]',
                '{confidence}        # note: the value you produce must be a number',
                '',
                '[[ ## completed ## ]]',
                '',
                'In adhering to this structure, your objective is:',
                'Answer a question',
            ].join('\n')
        );
    });

    it('writes the inputs and the response order into the user message', () => {
        const messages = adapter.format({ signature: QA, inputs: { question: 'Capital?' } });

        expect(messages).toHaveLength(2);
        expect(messages[1]).toEqual({
            role: 'user',
            content:
                '[[ ## question ## ]]\nCapital?\n\n' +
                'Respond with the corresponding output fields, starting with the field ' +
                '`[[ ## answer ## ]]`, then `[[ ## confidence ## ]]` (must be a number), ' +
                'and then ending with the marker for `[[ ## completed ## ]]`.',
        });
    });

    it('names enum members, and JSON for structured types', () => {
        const sig =
            'text -> label: enum(spam|ham), tags: string[], meta: object, ok: boolean, n: int';
        const [system] = adapter.format({ signature: sig, inputs: { text: 't' } });

        expect(system.content).toContain('must be one of: spam; ham');
        expect(system.content).toContain('must be a JSON array of strings');
        expect(system.content).toContain('must be a JSON object');
        expect(system.content).toContain('must be true or false');
        expect(system.content).toContain('must be an integer');
    });

    it('uses zod constraints as hints', () => {
        const sig = signature({
            input: z.object({ review: z.string() }),
            output: z.object({
                sentiment: z.enum(['positive', 'negative']),
                rating: z.number().int().min(1).max(5),
            }),
        });
        const [system] = adapter.format({ signature: sig, inputs: { review: 'r' } });

        expect(system.content).toContain('must be one of: positive, negative');
        expect(system.content).toContain('must be of type int, between 1 and 5');
    });

    it('defaults the objective for a string signature', () => {
        const [system] = adapter.format({ signature: 'a, b -> c', inputs: {} });

        expect(system.content).toContain('Given the fields `a`, `b`, produce the fields `c`.');
    });

    it('replaces the objective with an instructions override', () => {
        const [system] = adapter.format({
            signature: QA,
            inputs: {},
            instructions: 'Be terse.',
        });

        expect(system.content).toContain('your objective is:\nBe terse.');
        expect(system.content).not.toContain('Answer a question');
    });

    it('renders demos as user/assistant pairs before the real turn', () => {
        const messages = adapter.format({
            signature: QA,
            inputs: { question: 'Real?' },
            demos: [
                new Example({ question: 'Demo?', answer: 'Yes', confidence: 1 }).withInputs(
                    'question'
                ),
            ],
        });

        expect(messages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user']);
        expect(messages[1].content).toBe('[[ ## question ## ]]\nDemo?');
        expect(messages[2].content).toBe(
            '[[ ## answer ## ]]\nYes\n\n[[ ## confidence ## ]]\n1\n\n[[ ## completed ## ]]'
        );
    });

    it('sends image inputs as content parts after their marker', () => {
        class Describe extends Signature {
            @ImageField({ description: 'the picture' })
            picture!: string;

            @InputField({ description: 'the language' })
            language!: string;

            @OutputField({ description: 'what it shows' })
            caption!: string;
        }

        const messages = adapter.format({
            signature: Describe,
            inputs: { picture: 'https://example.com/a.png', language: 'French' },
        });

        const content = messages[1].content;
        expect(Array.isArray(content)).toBe(true);
        expect((content as any[])[0]).toEqual({ type: 'text', text: '[[ ## picture ## ]]\n' });
        expect((content as any[])[1]).toEqual({
            type: 'image',
            source: { kind: 'url', url: 'https://example.com/a.png' },
        });
        expect(contentToText(content)).toContain('[[ ## language ## ]]\nFrench');
    });

    it('skips inputs that were not supplied', () => {
        const messages = adapter.format({ signature: 'a, b -> c', inputs: { b: 'x' } });

        expect(messages[1].content).not.toContain('[[ ## a ## ]]');
        expect(messages[1].content).toContain('[[ ## b ## ]]\nx');
    });
});

describe('ChatAdapter.parse', () => {
    it('splits a reply on its markers', () => {
        const parsed = adapter.parse(
            QA,
            'Sure.\n\n[[ ## answer ## ]]\nParis\nis the capital\n\n[[ ## confidence ## ]]\n0.9\n\n[[ ## completed ## ]]'
        );

        expect(parsed).toEqual({ answer: 'Paris\nis the capital', confidence: '0.9' });
    });

    it('ignores unknown markers and keeps the first occurrence of a field', () => {
        const parsed = adapter.parse(
            QA,
            '[[ ## notes ## ]]\nx\n[[ ## answer ## ]]\nfirst\n[[ ## answer ## ]]\nsecond'
        );

        expect(parsed).toEqual({ answer: 'first' });
    });

    it('tolerates field: value lines when the model drops the markers', () => {
        expect(adapter.parse(QA, 'answer: Paris\nconfidence: 0.9')).toEqual({
            answer: 'Paris',
            confidence: '0.9',
        });
    });

    it('tolerates a JSON object reply', () => {
        expect(adapter.parse(QA, '{"answer": "Paris", "confidence": 0.9}')).toEqual({
            answer: 'Paris',
            confidence: 0.9,
        });
    });

    it('returns nothing for a reply without any field', () => {
        expect(adapter.parse(QA, 'I am not sure what you mean.')).toEqual({});
    });

    it('validates through parseOutputs, coercing declared types', () => {
        expect(
            adapter.parseOutputs(QA, '[[ ## answer ## ]]\nParis\n[[ ## confidence ## ]]\n0.5')
        ).toEqual({
            answer: 'Paris',
            confidence: 0.5,
        });
        expect(() => adapter.parseOutputs(QA, '[[ ## answer ## ]]\nParis')).toThrow(
            ValidationError
        );
    });

    it('reads partial sections while streaming, hiding a half-written marker', () => {
        expect(adapter.parsePartial(QA, '[[ ## ans')).toEqual({});
        expect(adapter.parsePartial(QA, '[[ ## answer ## ]]\nPar')).toEqual({ answer: 'Par' });
        expect(adapter.parsePartial(QA, '[[ ## answer ## ]]\nParis\n\n[[ ## conf')).toEqual({
            answer: 'Paris',
        });
    });
});

describe('ChatAdapter JSON fallback', () => {
    it('retries with JSONAdapter when the reply holds no output field', async () => {
        const lm = new MockLM({
            responses: [
                'I refuse to follow formats.',
                '{"answer": "Paris", "confidence": 0.9}',
            ],
        });

        const result = await new Predict(QA, lm).forward({ question: 'Q' });

        expect(result.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(2);
        expect(String(lm.calls[1].messages[0].content)).toContain(
            'Respond with JSON matching this schema'
        );
    });

    it('does not fall back when the reply has fields that fail validation', async () => {
        const lm = new MockLM({ responses: ['answer: Paris\nconfidence: very high'] });

        await expect(new Predict(QA, lm).forward({ question: 'Q' })).rejects.toThrow(
            ValidationError
        );
        expect(lm.calls).toHaveLength(1);
    });

    it('reports the original failure when the fallback fails too', async () => {
        const lm = new MockLM({ responses: ['no fields here', 'still no JSON'] });

        const error = await new Predict(QA, lm).forward({ question: 'Q' }).catch((e) => e);

        expect(error).toBeInstanceOf(ValidationError);
        expect((error as ValidationError).rawOutput).toBe('no fields here');
    });

    it('never retries a language-model error', async () => {
        const lm = new MockLM({ responses: [] });

        await expect(new Predict(QA, lm).forward({ question: 'Q' })).rejects.toThrow(
            'no more scripted responses'
        );
        expect(lm.calls).toHaveLength(1);
    });

    it('can be switched off', async () => {
        const lm = new MockLM({ responses: ['nothing', '{"answer": "x", "confidence": 1}'] });
        const predict = new Predict(QA, {
            lm,
            adapter: new ChatAdapter({ jsonFallback: false }),
        });

        await expect(predict.forward({ question: 'Q' })).rejects.toThrow(ValidationError);
        expect(lm.calls).toHaveLength(1);
    });
});
