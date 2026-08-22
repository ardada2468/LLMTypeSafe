import { Example } from '../core/example';
import { Prediction } from '../core/prediction';
import {
    exactMatch,
    fieldAccuracy,
    matchMetric,
    normalizedMatch,
    numericMatch,
    tokenF1,
    tokenF1Metric,
} from './metrics';
import { type Metric } from './types';

/** One-input/one-output example plus a prediction, the shape metrics grade. */
const grade = async (
    metric: Metric,
    expected: Record<string, any>,
    predicted: Record<string, any>
) => {
    const example = new Example({ question: 'q', ...expected }).withInputs('question');
    return metric(example, new Prediction(predicted));
};

describe('exactMatch', () => {
    it.each([
        ['Paris', 'Paris', true],
        ['Paris', 'paris', false],
        ['Paris', ' Paris ', false],
        ['Paris', 'London', false],
    ])('grades expected %s against %s', async (expected, predicted, pass) => {
        expect(await grade(exactMatch, { answer: expected }, { answer: predicted })).toBe(pass);
    });

    it('matches a number against its string rendering', async () => {
        expect(await grade(exactMatch, { count: 42 }, { count: '42' })).toBe(true);
    });

    it('requires every expected field to match', async () => {
        const score = await grade(
            exactMatch,
            { answer: 'Paris', country: 'France' },
            { answer: 'Paris', country: 'Spain' }
        );

        expect(score).toBe(false);
    });

    it('compares arrays element by element', async () => {
        expect(await grade(exactMatch, { tags: ['a', 'b'] }, { tags: ['a', 'b'] })).toBe(true);
        expect(await grade(exactMatch, { tags: ['a', 'b'] }, { tags: ['b', 'a'] })).toBe(false);
    });

    it('compares nested objects by key', async () => {
        expect(await grade(exactMatch, { meta: { a: 1 } }, { meta: { a: 1 } })).toBe(true);
        expect(await grade(exactMatch, { meta: { a: 1 } }, { meta: { a: 1, b: 2 } })).toBe(
            false
        );
    });

    it('scores zero when the example expects nothing, rather than passing vacuously', async () => {
        const example = new Example({ question: 'q' }).withInputs('question');

        expect(exactMatch(example, new Prediction({ answer: 'anything' }))).toBe(0);
    });

    it('fails a field the prediction omits entirely', async () => {
        expect(await grade(exactMatch, { answer: 'Paris' }, {})).toBe(false);
    });

    it('fails a field neither side has, so a typo cannot score a perfect run', async () => {
        const metric = matchMetric({ fields: ['answr'] });

        expect(await grade(metric, { answer: 'Paris' }, { answer: 'London' })).toBe(false);
    });
});

describe('normalizedMatch', () => {
    it.each([
        ['Paris', '  paris  ', true],
        ['New  York', 'new york', true],
        ['Paris', 'Paris, France', false],
    ])('grades expected %s against %s', async (expected, predicted, pass) => {
        expect(await grade(normalizedMatch, { answer: expected }, { answer: predicted })).toBe(
            pass
        );
    });
});

describe('numericMatch', () => {
    it.each([
        [0.5, 0.5001, 0.001, true],
        [0.5, 0.52, 0.001, false],
        [100, '99.5', 1, true],
    ])('grades %s against %s within %s', async (expected, predicted, tolerance, pass) => {
        const metric = numericMatch(tolerance);

        expect(await grade(metric, { score: expected }, { score: predicted })).toBe(pass);
    });

    it('falls back to a normalized text comparison for non-numeric fields', async () => {
        const metric = numericMatch(0.01);

        expect(await grade(metric, { label: 'Bug' }, { label: 'bug' })).toBe(true);
    });
});

describe('matchMetric', () => {
    it('grades only the fields it is given', async () => {
        const metric = matchMetric({ fields: ['answer'] });

        const score = await grade(
            metric,
            { answer: 'Paris', country: 'France' },
            { answer: 'Paris', country: 'Spain' }
        );

        expect(score).toBe(true);
    });
});

describe('fieldAccuracy', () => {
    it('awards partial credit across a multi-output signature', async () => {
        const score = await grade(
            fieldAccuracy,
            { category: 'bug', urgency: 'high', owner: 'alice' },
            { category: 'BUG', urgency: 'low', owner: 'alice' }
        );

        expect(score).toBeCloseTo(2 / 3);
    });

    it('scores one when every field matches', async () => {
        expect(await grade(fieldAccuracy, { a: '1', b: '2' }, { a: '1', b: '2' })).toBe(1);
    });
});

describe('tokenF1', () => {
    it('scores one for an identical answer', async () => {
        expect(
            await grade(
                tokenF1,
                { answer: 'the capital is Paris' },
                { answer: 'The Capital is paris' }
            )
        ).toBe(1);
    });

    it('scores zero when no tokens overlap', async () => {
        expect(await grade(tokenF1, { answer: 'Paris' }, { answer: 'London' })).toBe(0);
    });

    it.each([
        ['東京', 'ロンドン'],
        ['Москва', 'Париж'],
        ['Αθήνα', 'Ρώμη'],
    ])('scores zero for two different non-Latin answers (%s vs %s)', async (a, b) => {
        expect(await grade(tokenF1, { answer: a }, { answer: b })).toBe(0);
    });

    it('scores one for the same non-Latin answer', async () => {
        expect(await grade(tokenF1, { answer: '東京' }, { answer: '東京' })).toBe(1);
    });

    it('keeps accented letters inside a token', async () => {
        expect(await grade(tokenF1, { answer: 'café' }, { answer: 'café' })).toBe(1);
        expect(await grade(tokenF1, { answer: 'café' }, { answer: 'cafe' })).toBe(0);
    });

    it('scores zero for two answers that differ but tokenize to nothing', async () => {
        expect(await grade(tokenF1, { answer: '!!!' }, { answer: '???' })).toBe(0);
    });

    it('rewards partial overlap between free-text answers', async () => {
        const score = (await grade(
            tokenF1,
            { answer: 'the capital of France is Paris' },
            { answer: 'Paris is the capital' }
        )) as number;

        expect(score).toBeGreaterThan(0.5);
        expect(score).toBeLessThan(1);
    });

    it('averages across the fields it is given', async () => {
        const metric = tokenF1Metric({ fields: ['a', 'b'] });

        const score = await grade(
            metric,
            { a: 'one two', b: 'three' },
            { a: 'one two', b: 'x' }
        );

        expect(score).toBe(0.5);
    });

    it('does not credit a repeated token more than it appears', async () => {
        const score = (await grade(
            tokenF1,
            { answer: 'red blue' },
            { answer: 'red red red red' }
        )) as number;

        expect(score).toBeCloseTo((2 * 0.25 * 0.5) / 0.75);
    });
});
