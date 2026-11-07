import { XMLAdapter } from './xml';
import { TwoStepAdapter } from './two-step';
import { JSONAdapter } from './json';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { Signature, InputField, OutputField } from '../core/signature';
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

describe('XMLAdapter', () => {
    const adapter = new XMLAdapter();

    it('wraps every field in tags', () => {
        const [system, user] = adapter.format({ signature: QA, inputs: { question: 'Q?' } });

        expect(system.content).toContain('<question>\n{question}\n</question>');
        expect(system.content).toContain(
            '<confidence>\n{confidence} (the value you produce must be a number)\n</confidence>'
        );
        expect(system.content).not.toContain('[[ ##');
        expect(user.content).toContain('<question>\nQ?\n</question>');
        expect(user.content).toContain('starting with `<answer>`, then `<confidence>`');
    });

    it('parses tagged fields and validates them', async () => {
        const lm = new MockLM({
            responses: ['<answer>\nParis\n</answer>\n<confidence>0.8</confidence>'],
        });

        const result = await new Predict(QA, { lm, adapter }).forward({ question: 'Q' });

        expect(result).toMatchObject({ answer: 'Paris', confidence: 0.8 });
    });

    it('reads an unclosed tag while streaming', () => {
        expect(adapter.parsePartial(QA, '<answer>Par')).toEqual({ answer: 'Par' });
        expect(adapter.parsePartial(QA, '<answer>Paris</ans')).toEqual({ answer: 'Paris' });
    });

    it('rejects a reply missing a required tag', () => {
        expect(() => adapter.parseOutputs(QA, '<answer>Paris</answer>')).toThrow(
            ValidationError
        );
    });
});

describe('TwoStepAdapter', () => {
    it('answers freely with the main model, then extracts with the second', async () => {
        const main = new MockLM({
            responses: ['Paris, obviously — I am about 90% sure of it.'],
        });
        const extractor = new MockLM({
            responses: ['[[ ## answer ## ]]\nParis\n\n[[ ## confidence ## ]]\n0.9'],
        });
        const adapter = new TwoStepAdapter({ extractionLM: extractor });

        const result = await new Predict(QA, { lm: main, adapter }).forward({
            question: 'Capital of France?',
        });

        expect(result).toMatchObject({ answer: 'Paris', confidence: 0.9 });

        // The main prompt is plain: no structure template.
        const mainPrompt = main.lastPrompt();
        expect(mainPrompt).toContain('question: Capital of France?');
        expect(mainPrompt).toContain('Your outputs must contain:');
        expect(mainPrompt).not.toContain('[[ ##');

        // The extraction call reads the main model's text.
        expect(extractor.lastPrompt()).toContain(
            '[[ ## text ## ]]\nParis, obviously — I am about 90% sure of it.'
        );
    });

    it('uses JSONAdapter for an extraction model with structured output', async () => {
        const main = new MockLM({ responses: ['It is Paris.'] });
        const extractor = new MockLM({
            structuredResponses: [{ answer: 'Paris', confidence: 1 }],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await new Predict(QA, {
            lm: main,
            adapter: new TwoStepAdapter({ extractionLM: extractor }),
        }).forward({ question: 'Q' });

        expect(result.answer).toBe('Paris');
        expect(extractor.structuredCalls).toHaveLength(1);
    });

    it('honours an explicit extraction adapter and works under ChainOfThought', async () => {
        const main = new MockLM({ responses: ['Step one... so Paris.'] });
        const extractor = new MockLM({
            responses: ['{"reasoning": "Step one", "answer": "Paris", "confidence": 1}'],
        });

        const result = await new ChainOfThought(QA, {
            lm: main,
            adapter: new TwoStepAdapter({
                extractionLM: extractor,
                extractionAdapter: new JSONAdapter(),
            }),
        }).forward({ question: 'Q' });

        expect(result.reasoning).toBe('Step one');
        expect(result.answer).toBe('Paris');
    });
});
