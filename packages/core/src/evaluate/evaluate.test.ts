import { Example } from '../core/example';
import { Prediction } from '../core/prediction';
import { Signature, InputField, OutputField } from '../core/signature';
import { Predict } from '../modules/predict';
import { MockLM } from '../test-utils';
import { evaluate } from './evaluate';
import { exactMatch, fieldAccuracy } from './metrics';
import { formatReport } from './report';
import { type EvaluationResult } from './types';

class QA extends Signature {
    static description = 'Answer a question';

    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;
}

const dataset = (pairs: Array<[string, string]>) =>
    pairs.map(([question, answer]) => new Example({ question, answer }).withInputs('question'));

/** Echoes the question back as the answer, and throws on a poisoned input. */
const echoProgram = async (inputs: Record<string, any>) => {
    if (inputs.question === 'boom') throw new Error('provider exploded');
    return new Prediction({ answer: inputs.question });
};

describe('evaluate', () => {
    it('reports the mean score across the dataset', async () => {
        const lm = new MockLM({
            responses: [
                'answer: Paris',
                'answer: Rome',
                'answer: Madrid',
                'answer: Lisbon',
                'answer: nope',
            ],
        });
        const data = dataset([
            ['France?', 'Paris'],
            ['Italy?', 'Rome'],
            ['Spain?', 'Madrid'],
            ['Portugal?', 'Lisbon'],
            ['Germany?', 'Berlin'],
        ]);

        const report = await evaluate(new Predict(QA, lm), data, exactMatch, {
            concurrency: 1,
        });

        expect(report.score).toBe(0.8);
        expect(report.count).toBe(5);
        expect(report.totalScore).toBe(4);
        expect(report.errorCount).toBe(0);
    });

    it('lines each result up with the input it came from', async () => {
        const data = dataset([
            ['alpha', 'alpha'],
            ['beta', 'beta'],
            ['gamma', 'gamma'],
        ]);

        const report = await evaluate(echoProgram, data, exactMatch, { concurrency: 3 });

        expect(report.results.map((result) => result.index)).toEqual([0, 1, 2]);
        expect(report.results.map((result) => result.inputs.question)).toEqual([
            'alpha',
            'beta',
            'gamma',
        ]);
        expect(report.results.map((result) => result.expected.answer)).toEqual([
            'alpha',
            'beta',
            'gamma',
        ]);
        expect(report.results[1].prediction?.get('answer')).toBe('beta');
    });

    it('records a failing example as a zero score with the error attached', async () => {
        const data = dataset([
            ['alpha', 'alpha'],
            ['boom', 'anything'],
            ['gamma', 'gamma'],
        ]);

        const report = await evaluate(echoProgram, data, exactMatch, { concurrency: 2 });

        expect(report.score).toBeCloseTo(2 / 3);
        expect(report.errorCount).toBe(1);
        expect(report.results[1].score).toBe(0);
        expect(report.results[1].error?.message).toBe('provider exploded');
        expect(report.results[1].prediction).toBeUndefined();
        expect(report.results[2].score).toBe(1);
    });

    it('records a throwing metric as a failed example rather than aborting', async () => {
        const data = dataset([
            ['alpha', 'alpha'],
            ['beta', 'beta'],
        ]);
        const metric = (example: Example) => {
            if (example.get('question') === 'beta') throw new Error('metric blew up');
            return true;
        };

        const report = await evaluate(echoProgram, data, metric, { concurrency: 1 });

        expect(report.score).toBe(0.5);
        expect(report.results[1].error?.message).toBe('metric blew up');
    });

    it('keeps the prediction on an example whose metric threw', async () => {
        const data = dataset([['alpha', 'alpha']]);
        const metric = () => {
            throw new Error('metric blew up');
        };

        const report = await evaluate(echoProgram, data, metric, { concurrency: 1 });

        expect(report.results[0].error).toBeDefined();
        expect(report.results[0].prediction?.get('answer')).toBe('alpha');
    });

    it('reads concurrency 0 as one in flight rather than the default', async () => {
        const data = dataset(
            Array.from({ length: 6 }, (_, i): [string, string] => [`q${i}`, `q${i}`])
        );
        let inFlight = 0;
        let peak = 0;

        const program = async (inputs: Record<string, any>) => {
            inFlight += 1;
            peak = Math.max(peak, inFlight);
            await new Promise((resolve) => setTimeout(resolve, 1));
            inFlight -= 1;
            return new Prediction({ answer: inputs.question });
        };

        await evaluate(program, data, exactMatch, { concurrency: 0 });

        expect(peak).toBe(1);
    });

    it('keeps no more than `concurrency` examples in flight', async () => {
        const data = dataset(
            Array.from({ length: 12 }, (_, i): [string, string] => [`q${i}`, `q${i}`])
        );
        let inFlight = 0;
        let peak = 0;

        const program = async (inputs: Record<string, any>) => {
            inFlight += 1;
            peak = Math.max(peak, inFlight);
            await new Promise((resolve) => setTimeout(resolve, 1));
            inFlight -= 1;
            return new Prediction({ answer: inputs.question });
        };

        const report = await evaluate(program, data, exactMatch, { concurrency: 3 });

        expect(peak).toBe(3);
        expect(report.score).toBe(1);
    });

    it('applies inputKeys to a dataset that carries none', async () => {
        const data = [new Example({ question: 'alpha', answer: 'alpha' })];

        const report = await evaluate(echoProgram, data, exactMatch, {
            inputKeys: ['question'],
        });

        expect(report.score).toBe(1);
        expect(report.results[0].inputs).toEqual({ question: 'alpha' });
    });

    it('fails only the affected example when its input keys are missing', async () => {
        const data = [new Example({ question: 'alpha', answer: 'alpha' })];

        const report = await evaluate(echoProgram, data, exactMatch);

        expect(report.score).toBe(0);
        expect(report.results[0].error?.message).toContain('Input keys not specified');
    });

    it('aggregates token usage and latency by diffing the model', async () => {
        const lm = new MockLM({ responses: ['answer: a', 'answer: b'] });
        const data = dataset([
            ['a', 'a'],
            ['b', 'b'],
        ]);

        const report = await evaluate(new Predict(QA, lm), data, exactMatch, {
            concurrency: 1,
        });

        expect(report.usage.requestCount).toBe(2);
        expect(report.usage.promptTokens).toBe(20);
        expect(report.usage.completionTokens).toBe(10);
        expect(report.usage.totalTokens).toBe(30);
        expect(report.usage.averageLatency).toBe(1);
        expect(report.usage.durationMs).toBeGreaterThanOrEqual(0);
    });

    it('counts only the requests made during the run', async () => {
        const lm = new MockLM({ responses: ['answer: warmup', 'answer: a'] });
        const predict = new Predict(QA, lm);
        await predict.forward({ question: 'warmup' });

        const report = await evaluate(predict, dataset([['a', 'a']]), exactMatch);

        expect(report.usage.requestCount).toBe(1);
        expect(report.usage.totalTokens).toBe(15);
    });

    it('reports zeroed usage when no model can be resolved', async () => {
        const report = await evaluate(echoProgram, dataset([['a', 'a']]), exactMatch);

        expect(report.usage.requestCount).toBe(0);
        expect(report.usage.totalTokens).toBe(0);
    });

    it('returns an empty report for an empty dataset', async () => {
        const report = await evaluate(echoProgram, [], exactMatch);

        expect(report.score).toBe(0);
        expect(report.count).toBe(0);
        expect(report.results).toEqual([]);
    });

    it('reads a boolean metric as one and zero', async () => {
        const data = dataset([
            ['a', 'a'],
            ['b', 'b'],
        ]);

        const report = await evaluate(
            echoProgram,
            data,
            (example) => example.get('answer') === 'a'
        );

        expect(report.results.map((result) => result.score)).toEqual([1, 0]);
    });

    it('scores a non-finite metric result as zero', async () => {
        const report = await evaluate(echoProgram, dataset([['a', 'a']]), () => Number.NaN);

        expect(report.results[0].score).toBe(0);
    });

    it('awards partial credit with a field-wise metric', async () => {
        const data = [
            new Example({ ticket: 't', category: 'bug', urgency: 'high' }).withInputs('ticket'),
        ];
        const program = async () => new Prediction({ category: 'bug', urgency: 'low' });

        const report = await evaluate(program, data, fieldAccuracy);

        expect(report.score).toBe(0.5);
    });

    it('calls onResult once per example', async () => {
        const seen: EvaluationResult[] = [];
        const data = dataset([
            ['a', 'a'],
            ['boom', 'x'],
        ]);

        await evaluate(echoProgram, data, exactMatch, {
            concurrency: 1,
            onResult: (result) => seen.push(result),
        });

        expect(seen).toHaveLength(2);
        expect(seen[1].error).toBeDefined();
    });

    it('ignores an onResult hook that throws', async () => {
        const report = await evaluate(echoProgram, dataset([['a', 'a']]), exactMatch, {
            onResult: () => {
                throw new Error('reporting is broken');
            },
        });

        expect(report.score).toBe(1);
        expect(report.results[0].error).toBeUndefined();
    });

    it('forwards call options to the program', async () => {
        const lm = new MockLM({ responses: ['answer: a'] });

        await evaluate(new Predict(QA, lm), dataset([['a', 'a']]), exactMatch, {
            callOptions: { temperature: 0 },
        });

        expect(lm.calls[0].options).toMatchObject({ temperature: 0 });
    });
});

describe('formatReport', () => {
    it('renders the aggregate, the usage, and one row per example', async () => {
        const data = dataset([
            ['alpha', 'alpha'],
            ['boom', 'x'],
        ]);
        const report = await evaluate(echoProgram, data, exactMatch, { concurrency: 1 });

        const text = formatReport(report);

        expect(text).toContain('score 0.500');
        expect(text).toContain('errors 1');
        expect(text).toContain('tokens 0');
        expect(text).toContain('provider exploded');
        expect(text.split('\n')).toHaveLength(6);
    });

    it('caps the rows it renders and says how many are left', async () => {
        const data = dataset(
            Array.from({ length: 5 }, (_, i): [string, string] => [`q${i}`, `q${i}`])
        );
        const report = await evaluate(echoProgram, data, exactMatch);

        const text = formatReport(report, { maxRows: 2 });

        expect(text).toContain('… 3 more');
    });

    it('still says how many rows exist when none are rendered', async () => {
        const data = dataset(
            Array.from({ length: 3 }, (_, i): [string, string] => [`q${i}`, `q${i}`])
        );
        const report = await evaluate(echoProgram, data, exactMatch);

        const text = formatReport(report, { maxRows: 0 });

        expect(text).toContain('… 3 more');
    });

    it('lists only the failures when passing rows are excluded', async () => {
        const data = dataset([
            ['alpha', 'alpha'],
            ['boom', 'x'],
        ]);
        const report = await evaluate(echoProgram, data, exactMatch);

        const text = formatReport(report, { includePassing: false });

        expect(text).toContain('provider exploded');
        expect(text).not.toContain('alpha');
    });
});
