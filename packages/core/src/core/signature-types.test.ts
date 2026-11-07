import { z } from 'zod';
import {
    Signature,
    InputField,
    OutputField,
    extendSignature,
    history,
    image,
    isZodSignature,
    signature,
    toSignatureClass,
    zodSpecialFieldType,
} from './signature';
import { History } from './history';
import { Predict } from '../modules/predict';
import { ChatAdapter } from '../adapters/chat';
import { JSONAdapter } from '../adapters/json';
import { getInputFieldConfigs, getOutputFieldConfigs } from '../utils/schema';
import { MockLM } from '../test-utils';

describe('History', () => {
    class Chat extends Signature {
        @InputField({ description: 'the question' })
        question!: string;

        @InputField({ type: 'history' })
        history!: History;

        @OutputField({ description: 'the answer' })
        answer!: string;
    }

    const past = new History([
        { question: 'Capital of France?', answer: 'Paris' },
        { question: 'And of Spain?', answer: 'Madrid' },
    ]);

    it('renders earlier turns as user/assistant pairs, not as a field', () => {
        const messages = new ChatAdapter().format({
            signature: Chat,
            inputs: { question: 'And of Italy?', history: past },
        });

        expect(messages.map((m) => m.role)).toEqual([
            'system',
            'user',
            'assistant',
            'user',
            'assistant',
            'user',
        ]);
        expect(messages[1].content).toBe('[[ ## question ## ]]\nCapital of France?');
        expect(messages[2].content).toBe('[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]');
        expect(messages[4].content).toContain('Madrid');
        // The history field itself appears nowhere in the field listing or the turn.
        expect(messages[0].content).not.toContain('`history`');
        expect(messages[5].content).not.toContain('history');
    });

    it('comes after demos and works on every declaration form', () => {
        const zodChat = signature({
            input: z.object({ question: z.string(), history: history() }),
            output: z.object({ answer: z.string() }),
        });
        for (const sig of [Chat, zodChat, 'question, history: history -> answer'] as const) {
            const messages = new JSONAdapter().format({
                signature: sig,
                inputs: { question: 'Q', history: past },
            });
            expect(messages).toHaveLength(6);
            expect(messages[2].content).toBe('{"answer":"Paris"}');
        }
    });

    it('accepts a plain { messages } object and an explicit history argument', () => {
        const messages = new ChatAdapter().format({
            signature: 'question -> answer',
            inputs: { question: 'Q' },
            history: {
                messages: [{ question: 'Earlier?', answer: 'Yes' }],
            } as unknown as History,
        });

        expect(messages).toHaveLength(4);
    });

    it('drives a multi-turn conversation through Predict', async () => {
        const lm = new MockLM({ responses: ['answer: Rome'] });

        const result = await new Predict(Chat, lm).forward({
            question: 'And of Italy?',
            history: past,
        });

        expect(result.answer).toBe('Rome');
        expect(lm.calls[0].messages).toHaveLength(6);
        expect(past.with({ question: 'And of Italy?', answer: 'Rome' }).messages).toHaveLength(
            3
        );
        expect(past.messages).toHaveLength(2);
    });
});

describe('zod image() and history()', () => {
    it('marks fields that survive describe() and optional()', () => {
        expect(zodSpecialFieldType(image().describe('photo').optional())).toBe('image');
        expect(zodSpecialFieldType(history())).toBe('history');
        expect(zodSpecialFieldType(z.string())).toBeUndefined();
    });

    it('lets a zod signature send an image as a content part', async () => {
        const Caption = signature({
            input: z.object({ photo: image().describe('the photo') }),
            output: z.object({ caption: z.string() }),
        });
        const lm = new MockLM({ responses: ['caption: a cat'] });

        const result = await new Predict(Caption, lm).forward({
            photo: 'data:image/png;base64,iVBORw0KGgo=',
        });

        expect(result.caption).toBe('a cat');
        expect(getInputFieldConfigs(Caption).photo.type).toBe('image');
        expect(lm.calls[0].messages.at(-1)?.content).toContainEqual({
            type: 'image',
            source: { kind: 'base64', mediaType: 'image/png', data: 'iVBORw0KGgo=' },
        });
    });
});

describe('optional outputs', () => {
    class Opt extends Signature {
        @OutputField({ description: 'answer' })
        answer!: string;

        @OutputField({ description: 'note', required: false })
        note?: string;
    }
    const zodOpt = signature({
        input: z.object({ q: z.string() }),
        output: z.object({
            answer: z.string(),
            note: z.string().optional(),
            tone: z.string().default('neutral'),
        }),
    });

    it.each([
        ['a class signature', Opt],
        ['a zod signature', zodOpt],
        ['a string signature', 'q -> answer, note?: string'],
    ])(
        'resolves a missing optional output to undefined or its default on %s',
        async (_, sig) => {
            const lm = new MockLM({
                responses: ['[[ ## answer ## ]]\nyes\n\n[[ ## completed ## ]]'],
            });

            const result = await new Predict(sig as any, lm).forward({ q: 'Q' });

            expect(result.answer).toBe('yes');
            expect(result.note).toBeUndefined();
            if (sig === zodOpt) expect(result.tone).toBe('neutral');
        }
    );

    it('treats an empty section as absent', async () => {
        const lm = new MockLM({
            responses: ['[[ ## answer ## ]]\nyes\n\n[[ ## note ## ]]\n\n'],
        });

        const result = await new Predict(Opt, lm).forward({});

        expect(result.note).toBeUndefined();
    });
});

describe('string signature descriptions and instructions', () => {
    it('parses quoted descriptions and optional markers', () => {
        const parsed = Signature.parseStringSignature(
            'question: string "the user question, verbatim", context -> answer: string "short answer", note?: string'
        );

        expect(parsed.inputs).toEqual(['question', 'context']);
        expect(parsed.outputs).toEqual(['answer', 'note']);
        expect(parsed.descriptions).toEqual({
            question: 'the user question, verbatim',
            answer: 'short answer',
        });
        expect(parsed.optional).toEqual(['note']);
        expect(parsed.types.question).toBe('string');
    });

    it('keeps an inline enum intact and allows arrows inside descriptions', () => {
        const parsed = Signature.parseStringSignature(
            'text "input -> text" -> label: enum(a|b) "the label"'
        );

        expect(parsed.descriptions.text).toBe('input -> text');
        expect(parsed.types.label).toBe('enum(a|b)');
    });

    it('puts descriptions in the prompt', () => {
        const [system] = new ChatAdapter().format({
            signature: 'question "the user question" -> answer: int "the result"',
            inputs: {},
        });

        expect(system.content).toContain('1. `question` (string): the user question');
        expect(system.content).toContain('1. `answer` (int): the result');
        expect(getOutputFieldConfigs('q -> a "x"').a.description).toBe('x');
    });

    it('attaches instructions with signature(shorthand, instructions)', async () => {
        const Sig = signature('question -> answer: int', 'Answer the maths question.');
        const lm = new MockLM({ responses: ['answer: 4'] });

        const result = await new Predict(Sig, lm).forward({ question: '2+2' });

        expect(result.answer).toBe(4);
        expect(String(lm.calls[0].messages[0].content)).toContain('Answer the maths question.');
    });

    it('accepts instructions on Predict for a string signature', async () => {
        const lm = new MockLM({ responses: ['answer: 4'] });

        await new Predict('question -> answer', { lm, instructions: 'Be brief.' }).forward({
            question: 'Q',
        });

        expect(String(lm.calls[0].messages[0].content)).toContain('objective is:\nBe brief.');
    });
});

describe('extendSignature', () => {
    class QA extends Signature {
        static description = 'Answer it.';

        @InputField({ description: 'the question' })
        question!: string;

        @OutputField({ description: 'the answer', type: 'number' })
        answer!: number;
    }

    it('prepends and appends fields on a class signature without touching it', () => {
        const extended = extendSignature(QA, {
            prependOutputs: { reasoning: { description: 'think' } },
            appendInputs: { context: { description: 'background' } },
        });

        expect(Object.keys(extended.getInputFields())).toEqual(['question', 'context']);
        expect(Object.keys(extended.getOutputFields())).toEqual(['reasoning', 'answer']);
        expect(extended.getOutputFields().answer.type).toBe('number');
        expect(extended.description).toBe('Answer it.');
        expect(Object.keys(QA.getOutputFields())).toEqual(['answer']);
    });

    it('turns a string signature into a class, keeping types and descriptions', () => {
        const extended = extendSignature('question "q" -> answer: int', {
            prependOutputs: { reasoning: {} },
        });

        expect(Object.keys(extended.getOutputFields())).toEqual(['reasoning', 'answer']);
        expect(extended.getOutputFields().answer.type).toBe('int');
        expect(extended.getInputFields().question.description).toBe('q');
    });

    it('keeps a zod signature zod, with its schemas intact', () => {
        const base = signature({
            description: 'Rate it.',
            input: z.object({ review: z.string() }),
            output: z.object({ rating: z.number().int().min(1).max(5) }),
        });

        const extended = extendSignature(base, {
            prependOutputs: { reasoning: { description: 'think' } },
            appendOutputs: { label: { type: 'enum', values: ['a', 'b'] } },
        });

        expect(isZodSignature(extended)).toBe(true);
        expect(Object.keys(extended.output.shape)).toEqual(['reasoning', 'rating', 'label']);
        expect(extended.output.shape.rating.safeParse(9).success).toBe(false);
        expect(extended.output.shape.label.safeParse('a').success).toBe(true);
        expect(extended.description).toBe('Rate it.');
    });

    it('removes fields and replaces instructions', () => {
        const extended = extendSignature(QA, {
            removeOutputs: ['answer'],
            appendOutputs: { summary: z.string().describe('a summary') },
            instructions: 'Summarise instead.',
        });

        expect(Object.keys(extended.getOutputFields())).toEqual(['summary']);
        expect(extended.getOutputFields().summary.description).toBe('a summary');
        expect(extended.description).toBe('Summarise instead.');
    });

    it('converts a string signature with toSignatureClass', () => {
        const Sig = toSignatureClass('a, b?: int -> c "the c"', 'Do it.');

        expect(Sig.description).toBe('Do it.');
        expect(Sig.getInputFields().b.required).toBe(false);
        expect(Sig.getOutputFields().c.description).toBe('the c');
    });
});
