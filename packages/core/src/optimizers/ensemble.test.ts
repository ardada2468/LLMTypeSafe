import { describe, expect, it } from 'vitest';
import { Ensemble, EnsembleProgram, majority } from './ensemble';
import { Predict } from '../modules/predict';
import { Prediction } from '../core/prediction';
import { MockLM } from '../testing/mock-lm';
import { reply } from './test-support';

const fixed = (answer: string) =>
    new Predict('question -> answer', new MockLM({ respond: () => reply({ answer }) }));

const failing = () =>
    new Predict(
        'question -> answer',
        new MockLM({
            respond: () => {
                throw new Error('down');
            },
        })
    );

describe('majority', () => {
    it('returns the first prediction carrying the most common answer', () => {
        const a = new Prediction({ answer: 'Paris', reasoning: 'one' });
        const b = new Prediction({ answer: 'Lyon', reasoning: 'two' });
        const c = new Prediction({ answer: ' paris ', reasoning: 'three' });
        expect(majority([b, a, c])).toBe(a);
    });

    it('breaks ties by first appearance and can vote on one field', () => {
        const a = new Prediction({ answer: 'x', note: '1' });
        const b = new Prediction({ answer: 'y', note: '1' });
        expect(majority([a, b])).toBe(a);
        expect(majority([a, b], { field: 'note' })).toBe(a);
        expect(
            majority([new Prediction({ answer: 'A' }), new Prediction({ answer: 'a' })], {
                normalize: false,
            }).get('answer')
        ).toBe('A');
        expect(() => majority([])).toThrow();
    });
});

describe('Ensemble', () => {
    it('runs every member and returns the majority answer', async () => {
        const members = [fixed('Paris'), fixed('Lyon'), fixed('Paris')];
        const ensemble = new Ensemble().compile(members);

        expect(ensemble).toBeInstanceOf(EnsembleProgram);
        const prediction = await ensemble.forward({ question: 'Capital of France?' });
        expect(prediction.get('answer')).toBe('Paris');
        expect(ensemble.namedPredictors().map(([path]) => path)).toEqual([
            'programs[0]',
            'programs[1]',
            'programs[2]',
        ]);
    });

    it('accepts a custom reducer', async () => {
        const ensemble = new Ensemble({
            reduceFn: (predictions) =>
                new Prediction({ answer: predictions.map((p) => p.get('answer')).join('+') }),
        }).compile([fixed('a'), fixed('b')]);
        expect((await ensemble.forward({ question: 'q' })).get('answer')).toBe('a+b');
    });

    it('samples `size` members per call with the seeded RNG', async () => {
        const lms = ['a', 'b', 'c', 'd'].map(
            (answer) => new MockLM({ respond: () => reply({ answer }) })
        );
        const members = lms.map((lm) => new Predict('question -> answer', lm));
        const ensemble = new Ensemble({ size: 2, seed: 3 }).compile(members);
        await ensemble.forward({ question: 'q' });
        expect(lms.reduce((sum, lm) => sum + lm.calls.length, 0)).toBe(2);

        const again = new Ensemble({ size: 2, seed: 3 }).compile(
            ['a', 'b', 'c', 'd'].map((answer) => fixed(answer))
        );
        const first = await new Ensemble({ size: 2, seed: 3 })
            .compile(['a', 'b', 'c', 'd'].map((answer) => fixed(answer)))
            .forward({ question: 'q' });
        expect((await again.forward({ question: 'q' })).get('answer')).toBe(
            first.get('answer')
        );
    });

    it('leaves failing members out of the vote, and fails only when all fail', async () => {
        const ensemble = new Ensemble().compile([failing(), fixed('ok')]);
        expect((await ensemble.forward({ question: 'q' })).get('answer')).toBe('ok');
        await expect(
            new Ensemble().compile([failing()]).forward({ question: 'q' })
        ).rejects.toThrow('down');
    });

    it('copies its members and validates its options', () => {
        const member = fixed('a');
        const ensemble = new Ensemble().compile([member]);
        expect(ensemble.programs[0]).not.toBe(member);
        expect(() => new Ensemble().compile([])).toThrow();
        expect(() => new Ensemble({ size: 0 })).toThrow(RangeError);
    });
});
