import { LabeledFewShot } from './labeled-few-shot';
import { Example } from '../core/example';
import { Predict } from '../modules/predict';
import { Signature, InputField, OutputField } from '../core/signature';
import { MockLM } from '../test-utils';

class QA extends Signature {
    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;
}

function trainset(size: number): Example[] {
    return Array.from({ length: size }, (_, i) =>
        new Example({ question: `q${i}`, answer: `a${i}` }).withInputs('question')
    );
}

function answers(examples: Example[]): string[] {
    return examples.map((example) => example.get('answer'));
}

describe('LabeledFewShot', () => {
    it('selects k demos from the trainset', () => {
        const selected = new LabeledFewShot({ k: 3 }).select(trainset(10));

        expect(selected).toHaveLength(3);
    });

    it('selects the same demos for the same seed', () => {
        const data = trainset(10);

        const first = new LabeledFewShot({ k: 3, seed: 42 }).select(data);
        const second = new LabeledFewShot({ k: 3, seed: 42 }).select(data);

        expect(answers(first)).toEqual(answers(second));
    });

    it('selects a different order for a different seed', () => {
        const data = trainset(20);

        const first = new LabeledFewShot({ k: 5, seed: 1 }).select(data);
        const second = new LabeledFewShot({ k: 5, seed: 2 }).select(data);

        expect(answers(first)).not.toEqual(answers(second));
    });

    it('returns the whole trainset when k exceeds it', () => {
        const selected = new LabeledFewShot({ k: 10 }).select(trainset(3));

        expect(selected).toHaveLength(3);
    });

    it('selects nothing for k of zero', () => {
        expect(new LabeledFewShot({ k: 0 }).select(trainset(5))).toEqual([]);
    });

    it('leaves the trainset it was given in its original order', () => {
        const data = trainset(10);

        new LabeledFewShot({ k: 4, seed: 3 }).select(data);

        expect(answers(data)).toEqual(answers(trainset(10)));
    });

    it('compiles a module that renders the selected demos', async () => {
        const lm = new MockLM({ responses: ['answer: compiled'] });
        const compiled = new LabeledFewShot({ k: 2, seed: 7 }).compile(new Predict(QA, lm), {
            trainset: trainset(6),
        });

        await compiled.forward({ question: 'live question' });

        const prompt = lm.lastPrompt();
        expect(prompt).toContain('Here are 2 worked examples of this task:');
        for (const demo of compiled.getDemos()) {
            expect(prompt).toContain(`answer: ${demo.get('answer')}`);
        }
    });

    it('leaves the student it compiled untouched', () => {
        const student = new Predict(QA, new MockLM());

        const compiled = new LabeledFewShot({ k: 2 }).compile(student, {
            trainset: trainset(5),
        });

        expect(student.getDemos()).toHaveLength(0);
        expect(compiled.getDemos()).toHaveLength(2);
    });
});
