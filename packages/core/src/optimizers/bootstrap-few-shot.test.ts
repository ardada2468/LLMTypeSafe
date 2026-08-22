import { BootstrapFewShot, type BootstrapProgressEvent } from './bootstrap-few-shot';
import { Example } from '../core/example';
import { Prediction } from '../core/prediction';
import { Predict } from '../modules/predict';
import { Signature, InputField, OutputField } from '../core/signature';
import type { ILanguageModel } from '../types/language-model';
import { MockLM } from '../test-utils';

class QA extends Signature {
    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;
}

/**
 * A module whose reply is a pure function of its input, so a test can decide
 * which trainset rows pass without depending on the order they are attempted in.
 */
class FakeModule {
    demos: Example[] = [];
    /** Shared with every copy on purpose, so a test can see what the copies did. */
    readonly runs: Array<{ inputs: Record<string, any>; model?: string }> = [];
    lm?: ILanguageModel;

    constructor(private readonly reply: (inputs: Record<string, any>) => Record<string, any>) {}

    async forward(inputs: Record<string, any>): Promise<Prediction> {
        this.runs.push({ inputs, model: this.lm?.getModelName() });
        return new Prediction(this.reply(inputs));
    }

    withDemos(demos: Example[]): this {
        return this.cloneWith({ demos: [...demos] });
    }

    withLM(lm: ILanguageModel): this {
        return this.cloneWith({ lm });
    }

    private cloneWith(patch: Record<string, any>): this {
        const clone = Object.create(Object.getPrototypeOf(this)) as this;
        Object.assign(clone, this, patch);
        return clone;
    }
}

const exactMatch = (example: Example, prediction: Prediction) =>
    example.get('answer') === prediction.get('answer');

function trainset(size: number): Example[] {
    return Array.from({ length: size }, (_, i) =>
        new Example({ question: `q${i}`, answer: `a${i}` }).withInputs('question')
    );
}

/** Answer `q<n>` correctly only for the listed indices. */
function replyCorrectlyFor(indices: number[]) {
    return (inputs: Record<string, any>) => {
        const index = Number(String(inputs.question).slice(1));
        return { answer: indices.includes(index) ? `a${index}` : 'wrong' };
    };
}

function demoQuestions(demos: Example[]): string[] {
    return demos.map((demo) => demo.get('question'));
}

describe('BootstrapFewShot', () => {
    it('promotes only the runs whose metric passes', async () => {
        const student = new FakeModule(replyCorrectlyFor([1, 3]));

        const compiled = await new BootstrapFewShot({ metric: exactMatch }).compile(student, {
            trainset: trainset(5),
        });

        expect(demoQuestions(compiled.demos).sort()).toEqual(['q1', 'q3']);
    });

    it('builds demos from what the model produced, paired with the example inputs', async () => {
        const student = new FakeModule(() => ({ answer: 'a0', note: 'extra field' }));

        const compiled = await new BootstrapFewShot({ metric: exactMatch }).compile(student, {
            trainset: [new Example({ question: 'q0', answer: 'a0' }).withInputs('question')],
        });

        expect(compiled.demos[0].getInputs()).toEqual({ question: 'q0' });
        expect(compiled.demos[0].getOutputs()).toEqual({ answer: 'a0', note: 'extra field' });
    });

    it('skips an example whose attempt throws instead of aborting the run', async () => {
        const student = new FakeModule((inputs) => {
            if (inputs.question === 'q2') {
                throw new Error('provider exploded');
            }
            return { answer: `a${String(inputs.question).slice(1)}` };
        });

        const compiled = await new BootstrapFewShot({ metric: exactMatch }).compile(student, {
            trainset: trainset(4),
        });

        expect(demoQuestions(compiled.demos).sort()).toEqual(['q0', 'q1', 'q3']);
    });

    it('skips an example whose metric throws', async () => {
        const student = new FakeModule((inputs) => ({
            answer: `a${String(inputs.question).slice(1)}`,
        }));
        const metric = (example: Example, prediction: Prediction) => {
            if (example.get('question') === 'q1') {
                throw new Error('metric exploded');
            }
            return exactMatch(example, prediction);
        };

        const compiled = await new BootstrapFewShot({ metric }).compile(student, {
            trainset: trainset(3),
        });

        expect(demoQuestions(compiled.demos).sort()).toEqual(['q0', 'q2']);
    });

    it('produces the same demos, in the same order, for the same seed', async () => {
        const data = trainset(12);
        const optimizer = () =>
            new BootstrapFewShot({ metric: exactMatch, seed: 99, maxBootstrappedDemos: 4 });

        const first = await optimizer().compile(
            new FakeModule(replyCorrectlyFor([0, 2, 4, 6, 8, 10])),
            {
                trainset: data,
            }
        );
        const second = await optimizer().compile(
            new FakeModule(replyCorrectlyFor([0, 2, 4, 6, 8, 10])),
            { trainset: data }
        );

        expect(demoQuestions(first.demos)).toEqual(demoQuestions(second.demos));
    });

    it('caps demos at maxBootstrappedDemos', async () => {
        const student = new FakeModule((inputs) => ({
            answer: `a${String(inputs.question).slice(1)}`,
        }));

        const compiled = await new BootstrapFewShot({
            metric: exactMatch,
            maxBootstrappedDemos: 2,
        }).compile(student, { trainset: trainset(8) });

        expect(compiled.demos).toHaveLength(2);
    });

    it('treats a numeric score below the threshold as a failure', async () => {
        const student = new FakeModule(() => ({ answer: 'anything' }));

        const compiled = await new BootstrapFewShot({
            metric: () => 0.4,
            threshold: 0.5,
        }).compile(student, { trainset: trainset(3) });

        expect(compiled.demos).toHaveLength(0);
    });

    it('treats a numeric score at the threshold as a pass', async () => {
        const student = new FakeModule(() => ({ answer: 'anything' }));

        const compiled = await new BootstrapFewShot({
            metric: () => 0.5,
            threshold: 0.5,
        }).compile(student, { trainset: trainset(3) });

        expect(compiled.demos).toHaveLength(3);
    });

    it('awaits an async metric', async () => {
        const student = new FakeModule(() => ({ answer: 'a0' }));

        const compiled = await new BootstrapFewShot({
            metric: async (example, prediction) => exactMatch(example, prediction),
        }).compile(student, {
            trainset: [new Example({ question: 'q0', answer: 'a0' }).withInputs('question')],
        });

        expect(compiled.demos).toHaveLength(1);
    });

    it('tops up with labelled examples when bootstrapping came up short', async () => {
        const student = new FakeModule(replyCorrectlyFor([0]));

        const compiled = await new BootstrapFewShot({
            metric: exactMatch,
            maxBootstrappedDemos: 3,
            maxLabeledDemos: 2,
        }).compile(student, { trainset: trainset(5) });

        expect(compiled.demos).toHaveLength(3);
        expect(demoQuestions(compiled.demos)).toContain('q0');
        // The example that produced the bootstrapped demo is not repeated.
        expect(demoQuestions(compiled.demos).filter((q) => q === 'q0')).toHaveLength(1);
    });

    it('adds no labelled demos by default', async () => {
        const student = new FakeModule(() => ({ answer: 'wrong' }));

        const compiled = await new BootstrapFewShot({ metric: exactMatch }).compile(student, {
            trainset: trainset(5),
        });

        expect(compiled.demos).toHaveLength(0);
    });

    it('reports the outcome of every trainset example', async () => {
        const student = new FakeModule(replyCorrectlyFor([0]));
        const events: BootstrapProgressEvent[] = [];

        await new BootstrapFewShot({
            metric: exactMatch,
            onProgress: (event) => events.push(event),
        }).compile(student, { trainset: trainset(3) });

        expect(events).toHaveLength(3);
        expect(events.filter((e) => e.status === 'passed')).toHaveLength(1);
        expect(events.filter((e) => e.status === 'failed')).toHaveLength(2);
        expect(events.every((e) => e.total === 3)).toBe(true);
    });

    it('reports a thrown attempt as an error event', async () => {
        const student = new FakeModule(() => {
            throw new Error('provider exploded');
        });
        const events: BootstrapProgressEvent[] = [];

        await new BootstrapFewShot({
            metric: exactMatch,
            onProgress: (event) => events.push(event),
        }).compile(student, { trainset: trainset(1) });

        expect(events[0].status).toBe('error');
        expect((events[0].error as Error).message).toBe('provider exploded');
    });

    it('generates demos with the teacher and returns the student carrying them', async () => {
        const teacher = new MockLM();
        const student = new FakeModule((inputs) => ({
            answer: `a${String(inputs.question).slice(1)}`,
        }));

        const compiled = await new BootstrapFewShot({ metric: exactMatch, teacher }).compile(
            student,
            { trainset: trainset(2) }
        );

        // Every trainset run went through the teacher's model.
        expect(student.runs.map((run) => run.model)).toEqual(['mock-model', 'mock-model']);
        // The compiled module is the student, still on the student's own model.
        expect(compiled.lm).toBeUndefined();
        expect(compiled.demos).toHaveLength(2);
    });

    it('rejects a teacher when the module cannot swap its model', async () => {
        const student: any = {
            forward: async () => new Prediction({}),
            withDemos: () => student,
        };

        await expect(
            new BootstrapFewShot({ metric: exactMatch, teacher: new MockLM() }).compile(
                student as any,
                { trainset: trainset(1) }
            )
        ).rejects.toThrow(/withLM/);
    });

    it('runs at most `concurrency` attempts at a time', async () => {
        let inFlight = 0;
        let peak = 0;
        const student = new FakeModule(() => ({ answer: 'x' })) as any;
        student.forward = async (inputs: Record<string, any>) => {
            inFlight += 1;
            peak = Math.max(peak, inFlight);
            await new Promise((resolve) => setTimeout(resolve, 1));
            inFlight -= 1;
            return new Prediction({ answer: `a${String(inputs.question).slice(1)}` });
        };

        await new BootstrapFewShot({ metric: exactMatch, concurrency: 2 }).compile(student, {
            trainset: trainset(8),
        });

        expect(peak).toBeLessThanOrEqual(2);
        expect(peak).toBeGreaterThan(1);
    });

    it('names the offending example when the trainset declares no input fields', async () => {
        const student = new FakeModule(() => ({ answer: 'a0' }));

        await expect(
            new BootstrapFewShot({ metric: exactMatch }).compile(student, {
                trainset: [new Example({ question: 'q0', answer: 'a0' })],
            })
        ).rejects.toThrow(/withInputs/);
    });

    it('accepts inputKeys instead of withInputs on every example', async () => {
        const student = new FakeModule(() => ({ answer: 'a0' }));

        const compiled = await new BootstrapFewShot({
            metric: exactMatch,
            inputKeys: ['question'],
        }).compile(student, { trainset: [new Example({ question: 'q0', answer: 'a0' })] });

        expect(compiled.demos).toHaveLength(1);
        expect(student.runs[0].inputs).toEqual({ question: 'q0' });
    });

    it('requires a metric', () => {
        expect(() => new BootstrapFewShot({} as any)).toThrow(/metric/);
    });

    it('names a field that inputKeys asks for but the example lacks', async () => {
        const student = new FakeModule(() => ({ answer: 'a0' }));

        await expect(
            new BootstrapFewShot({ metric: exactMatch, inputKeys: ['qeustion'] }).compile(
                student,
                { trainset: [new Example({ question: 'q0', answer: 'a0' })] }
            )
        ).rejects.toThrow(/qeustion/);
    });

    it('stops running the trainset once it has enough demos', async () => {
        const student = new FakeModule((inputs) => ({
            answer: `a${String(inputs.question).slice(1)}`,
        }));

        await new BootstrapFewShot({
            metric: exactMatch,
            maxBootstrappedDemos: 2,
            concurrency: 2,
        }).compile(student, { trainset: trainset(50) });

        // One batch of two suffices; the other 48 rows are never paid for.
        expect(student.runs).toHaveLength(2);
    });

    it('makes no model calls at all for a labels-only compile', async () => {
        const student = new FakeModule(() => ({ answer: 'a0' }));

        const compiled = await new BootstrapFewShot({
            metric: exactMatch,
            maxBootstrappedDemos: 0,
            maxLabeledDemos: 3,
        }).compile(student, { trainset: trainset(6) });

        expect(student.runs).toHaveLength(0);
        expect(compiled.demos).toHaveLength(3);
    });

    it('carries on when the progress callback throws', async () => {
        const student = new FakeModule((inputs) => ({
            answer: `a${String(inputs.question).slice(1)}`,
        }));

        const compiled = await new BootstrapFewShot({
            metric: exactMatch,
            onProgress: () => {
                throw new Error('logging exploded');
            },
        }).compile(student, { trainset: trainset(4) });

        expect(compiled.demos).toHaveLength(4);
    });

    it('compiles an empty trainset into a module with no demos', async () => {
        const student = new FakeModule(() => ({ answer: 'a0' }));

        const compiled = await new BootstrapFewShot({ metric: exactMatch }).compile(student, {
            trainset: [],
        });

        expect(compiled.demos).toEqual([]);
    });

    it('puts the bootstrapped demos into the compiled module prompt', async () => {
        const lm = new MockLM({
            responses: ['answer: Paris', 'answer: from the compiled run'],
        });
        const optimizer = new BootstrapFewShot({
            metric: (example, prediction) => example.get('answer') === prediction.get('answer'),
        });

        const compiled = await optimizer.compile(new Predict(QA, lm), {
            trainset: [
                new Example({ question: 'Capital of France?', answer: 'Paris' }).withInputs(
                    'question'
                ),
            ],
        });
        await compiled.forward({ question: 'Capital of Peru?' });

        expect(lm.lastPrompt()).toContain('question: Capital of France?');
        expect(lm.lastPrompt()).toContain('answer: Paris');
    });

    it('leaves the compiled module unchanged when nothing passed', async () => {
        const lm = new MockLM({
            responses: ['answer: Berlin', 'answer: from the compiled run'],
        });
        const optimizer = new BootstrapFewShot({
            metric: (example, prediction) => example.get('answer') === prediction.get('answer'),
        });

        const compiled = await optimizer.compile(new Predict(QA, lm), {
            trainset: [
                new Example({ question: 'Capital of France?', answer: 'Paris' }).withInputs(
                    'question'
                ),
            ],
        });
        await compiled.forward({ question: 'Capital of Peru?' });

        expect(compiled.getDemos()).toHaveLength(0);
        expect(lm.lastPrompt()).not.toContain('worked example');
    });
});
