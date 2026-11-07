import { Example } from './example';
import { Prediction } from './prediction';
import { parseJsonResponse } from './base-lm';
import { LMError } from './errors';

describe('Example', () => {
    it('exposes data fields as properties', () => {
        const ex = new Example({ question: 'Q', answer: 'A' }) as Example & Record<string, any>;

        expect(ex.question).toBe('Q');
        expect(ex.answer).toBe('A');
        expect(ex).toBeInstanceOf(Example);
    });

    it('never lets a field shadow a method', () => {
        const ex = new Example({
            get: 'g',
            set: 's',
            toObject: 't',
            withInputs: 'w',
            question: 'Q',
        }).withInputs('question', 'get');

        expect(typeof ex.get).toBe('function');
        expect(typeof ex.set).toBe('function');
        expect(typeof ex.toObject).toBe('function');
        expect(ex.get('get')).toBe('g');
        expect(ex.get('toObject')).toBe('t');
        expect(ex.toObject()).toEqual({
            get: 'g',
            set: 's',
            toObject: 't',
            withInputs: 'w',
            question: 'Q',
        });
        expect(ex.getInputs()).toEqual({ question: 'Q', get: 'g' });
        expect(ex.getOutputs()).toEqual({ set: 's', toObject: 't', withInputs: 'w' });
    });

    it('keeps internal state out of reach of field names', () => {
        const ex = new Example({ _data: 'd', _inputKeys: 'k' }) as Example &
            Record<string, any>;

        expect(ex._data).toBe('d');
        expect(ex._inputKeys).toBe('k');
        expect(() => ex.getInputs()).toThrow('Input keys not specified');
    });

    it('reports data keys through Object.keys, spread and JSON', () => {
        const ex = new Example({ question: 'Q', answer: 'A' }).withInputs('question');

        expect(Object.keys(ex)).toEqual(['question', 'answer']);
        expect({ ...ex }).toEqual({ question: 'Q', answer: 'A' });
        expect(JSON.parse(JSON.stringify(ex))).toEqual({ question: 'Q', answer: 'A' });
        expect('question' in ex).toBe(true);
        expect('missing' in ex).toBe(false);
    });

    it('serialises shadowed fields too', () => {
        const ex = new Example({ get: 'g', question: 'Q' });

        expect(JSON.parse(JSON.stringify(ex))).toEqual({ get: 'g', question: 'Q' });
    });

    it('reads and writes fields through set() and property assignment', () => {
        const ex = new Example({ question: 'Q' }) as Example & Record<string, any>;

        ex.set('answer', 'A');
        ex.question = 'Q2';

        expect(ex.answer).toBe('A');
        expect(ex.get('question')).toBe('Q2');
        expect(ex.toObject()).toEqual({ question: 'Q2', answer: 'A' });
    });

    it('does not share data with the example it was copied from', () => {
        const original = new Example({ question: 'Q' });
        const copy = original.withInputs('question');

        copy.set('question', 'changed');

        expect(original.get('question')).toBe('Q');
        expect(copy).toBeInstanceOf(Example);
    });
});

describe('Prediction', () => {
    it('exposes data fields as properties', () => {
        const prediction = new Prediction({ answer: 'A' }) as Prediction & Record<string, any>;

        expect(prediction.answer).toBe('A');
        expect(prediction).toBeInstanceOf(Prediction);
    });

    it('keeps trace and methods when the data uses their names', () => {
        const trace = { moduleId: 'Predict#1' } as any;
        const prediction = new Prediction(
            { trace: 'field', get: 'g', toObject: 't', toString: 's', answer: 'A' },
            trace
        );

        expect(prediction.trace).toBe(trace);
        expect(typeof prediction.get).toBe('function');
        expect(typeof prediction.toObject).toBe('function');
        expect(prediction.get('trace')).toBe('field');
        expect(prediction.get('get')).toBe('g');
        expect(prediction.toObject()).toEqual({
            trace: 'field',
            get: 'g',
            toObject: 't',
            toString: 's',
            answer: 'A',
        });
        expect(prediction.toString()).toContain('"answer": "A"');
    });

    it('reports data keys through Object.keys, spread and JSON', () => {
        const prediction = new Prediction({ answer: 'A', score: 1 }, { moduleId: 'x' } as any);

        expect(Object.keys(prediction)).toEqual(['answer', 'score']);
        expect({ ...prediction }).toEqual({ answer: 'A', score: 1 });
        expect(JSON.parse(JSON.stringify(prediction))).toEqual({ answer: 'A', score: 1 });
    });

    it('keeps data fields read-only', () => {
        const prediction = new Prediction({ answer: 'A' }) as Prediction & Record<string, any>;

        expect(() => {
            prediction.answer = 'B';
        }).toThrow(TypeError);
        expect(prediction.answer).toBe('A');
    });
});

describe('parseJsonResponse', () => {
    it('reads a fenced JSON object', () => {
        expect(parseJsonResponse('```json\n{"a":1}\n```', 'p')).toEqual({ a: 1 });
    });

    it('recovers an object wrapped in prose', () => {
        expect(parseJsonResponse('Sure: {"a":1} hope that helps', 'p')).toEqual({ a: 1 });
    });

    it('throws an LMError for a provider name', () => {
        expect(() => parseJsonResponse('nope', 'p')).toThrow(LMError);
    });

    it('throws the caller-built error when given a factory', () => {
        const custom = new Error('custom');
        expect(() => parseJsonResponse('nope', () => custom)).toThrow(custom);
    });
});
