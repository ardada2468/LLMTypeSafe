import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Module } from './module';
import { type Prediction } from './prediction';
import { Example } from './example';
import { TsDspyError } from './errors';
import { Signature, InputField, OutputField } from './signature';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { ReAct } from '../modules/react';
import { MockLM } from '../test-utils';

class QA extends Signature {
    static description = 'Answer the question.';

    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;
}

/** Sub-modules held directly, in an array, and in a record — plus a cycle. */
class Program extends Module {
    retrieve = new Predict('question -> passages');
    steps = [new ChainOfThought('question -> answer'), new Predict('question -> answer')];
    experts: Record<string, Module> = {
        math: new Predict('question -> answer'),
        agent: new ReAct('question -> answer', { tools: {} }),
    };
    self: Module = this;
    note = { label: 'not a module' };

    async forward(inputs: Record<string, any>): Promise<Prediction> {
        return this.steps[1].forward(inputs);
    }
}

class Outer extends Module {
    inner = new Program();
    generate = new Predict(QA);

    async forward(inputs: Record<string, any>): Promise<Prediction> {
        return this.generate.forward(inputs);
    }
}

const demo = (q: string, a: string) =>
    new Example({ question: q, answer: a }).withInputs('question');

describe('composition', () => {
    it('lists descendants with their paths, without self or cycles', () => {
        const program = new Program();

        expect(program.namedSubModules().map(([path]) => path)).toEqual([
            'retrieve',
            'steps[0]',
            'steps[1]',
            'experts.math',
            'experts.agent',
            // Changed in 0.7: ReAct holds its step and extract predictors.
            'experts.agent.react',
            'experts.agent.extract',
        ]);
    });

    it('prefixes the paths of nested composites', () => {
        const paths = new Outer().namedSubModules().map(([path]) => path);
        expect(paths).toContain('inner.steps[0]');
        expect(paths).toContain('generate');
    });

    it('reports a bare predictor as self', () => {
        const predict = new Predict(QA);
        expect(predict.namedPredictors()).toEqual([['self', predict]]);
        expect(predict.predictors()).toEqual([predict]);
    });

    it('lists the Predict instances of a composite, including ReAct’s', () => {
        const program = new Program();

        // Changed in 0.7: ReAct is built from predictors an optimizer can tune.
        expect(program.namedPredictors().map(([path]) => path)).toEqual([
            'retrieve',
            'steps[0]',
            'steps[1]',
            'experts.math',
            'experts.agent.react',
            'experts.agent.extract',
        ]);
        expect(program.predictors()[1]).toBeInstanceOf(ChainOfThought);
    });

    it('deep-copies the tree, preserving classes and sharing models and signatures', () => {
        const lm = new MockLM();
        const program = new Program();
        program.steps[1] = new Predict(QA, { lm, demos: [demo('q', 'a')] });

        const copy = program.deepCopy();
        const copied = copy.steps[1] as Predict;
        const original = program.steps[1] as Predict;

        expect(copy).toBeInstanceOf(Program);
        expect(copy).not.toBe(program);
        expect(copy.steps[0]).toBeInstanceOf(ChainOfThought);
        expect(copy.steps).not.toBe(program.steps);
        expect(copy.experts).not.toBe(program.experts);
        expect(copy.experts.agent).toBeInstanceOf(ReAct);
        // The cycle is reproduced inside the copy rather than pointing back out.
        expect(copy.self).toBe(copy);
        expect(copied).not.toBe(original);
        expect((copied as any).lm).toBe(lm);
        expect((copied as any).signature).toBe(QA);
        expect(copied.moduleId).not.toBe(original.moduleId);

        copied.setDemos([]);
        copied.setInstructions('changed');
        expect(original.getDemos()).toHaveLength(1);
        expect(original.getInstructions()).toBeUndefined();
    });

    it('copies demos so mutating one copy’s examples leaves the other alone', () => {
        const original = new Predict(QA, { demos: [demo('q', 'a')] });
        const copy = original.deepCopy();

        copy.getDemos()[0].set('answer', 'changed');

        expect(original.getDemos()[0].get('answer')).toBe('a');
        expect(copy.getDemos()[0].getInputKeys()).toEqual(['question']);
    });

    it('resets demos and instructions in a resetCopy', () => {
        const program = new Program();
        const target = program.steps[1] as Predict;
        target.setDemos([demo('q', 'a')]);
        target.setInstructions('Be terse.');

        const reset = program.resetCopy();

        for (const predictor of reset.predictors()) {
            expect(predictor.getDemos()).toEqual([]);
            expect(predictor.getInstructions()).toBeUndefined();
        }
        expect(target.getDemos()).toHaveLength(1);
    });
});

describe('instructions', () => {
    it('replace the signature description in the prompt', async () => {
        const lm = new MockLM({ responses: ['answer: A', 'answer: B'] });
        const predict = new Predict(QA, lm);
        const tuned = predict.withInstructions('Reply like a pirate.');

        await predict.forward({ question: 'Q' });
        await tuned.forward({ question: 'Q' });

        expect(lm.calls[0].messages[0].content).toContain('Answer the question.');
        expect(lm.calls[1].messages[0].content).toContain('Reply like a pirate.');
        expect(lm.calls[1].messages[0].content).not.toContain('Answer the question.');
        expect(predict.getInstructions()).toBeUndefined();
        expect(tuned).toBeInstanceOf(Predict);
    });

    it('open the prompt for a string signature, and reach ChainOfThought', async () => {
        const lm = new MockLM({ responses: ['reasoning: because\nanswer: A'] });
        const cot = new ChainOfThought('question -> answer', {
            lm,
            instructions: 'Think carefully.',
        });

        await cot.forward({ question: 'Q' });

        // The adapter states the objective at the end of the system message.
        expect(lm.calls).toHaveLength(1);
        expect(String(lm.calls[0].messages[0].content)).toContain(
            'your objective is:\nThink carefully.'
        );
    });
});

describe('save and load', () => {
    function compiledOuter(): Outer {
        const outer = new Outer();
        outer.generate.setDemos([demo('Capital of France?', 'Paris')]);
        outer.generate.setInstructions('Answer in one word.');
        (outer.inner.steps[0] as Predict).setDemos([
            new Example({ question: 'raw', answer: 'no split' }),
        ]);
        return outer;
    }

    it('dumps plain JSON keyed by predictor path, with metadata', () => {
        const state = compiledOuter().dumpState();

        expect(Object.keys(state).sort()).toEqual(
            [
                'generate',
                'inner.retrieve',
                'inner.steps[0]',
                'inner.steps[1]',
                'inner.experts.math',
                'inner.experts.agent.react',
                'inner.experts.agent.extract',
                'metadata',
            ].sort()
        );
        expect(state.generate).toEqual({
            demos: [
                {
                    data: { question: 'Capital of France?', answer: 'Paris' },
                    inputKeys: ['question'],
                },
            ],
            instructions: 'Answer in one word.',
        });
        expect(state['inner.steps[0]']).toEqual({
            demos: [{ data: { question: 'raw', answer: 'no split' } }],
        });
        expect(state.metadata).toMatchObject({ library: 'ts-dspy' });
        expect(typeof state.metadata.version).toBe('string');
        expect(JSON.parse(JSON.stringify(state))).toEqual(state);
    });

    it('round-trips a composite through loadState into a fresh program', async () => {
        const state = compiledOuter().dumpState();
        const fresh = new Outer();

        expect(fresh.loadState(JSON.parse(JSON.stringify(state)))).toBe(fresh);

        expect(fresh.dumpState()).toEqual({ ...state, metadata: expect.anything() });
        const [restored] = fresh.generate.getDemos();
        expect(restored.getInputs()).toEqual({ question: 'Capital of France?' });

        // And the restored state actually reaches the prompt.
        const lm = new MockLM({ responses: ['answer: Rome'] });
        await fresh.generate.withLM(lm).forward({ question: 'Capital of Italy?' });
        expect(lm.lastPrompt()).toContain('Answer in one word.');
        expect(lm.lastPrompt()).toContain('Capital of France?');
    });

    it('round-trips through a file', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'ts-dspy-state-'));
        try {
            const file = join(dir, 'program.json');
            await compiledOuter().save(file);

            expect(JSON.parse(await readFile(file, 'utf8')).metadata.library).toBe('ts-dspy');
            const loaded = await new Outer().load(file);
            expect(loaded.generate.getInstructions()).toBe('Answer in one word.');
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });

    it('rejects state for a different program, naming the paths', () => {
        const state = compiledOuter().dumpState();
        const other = new Program();

        expect(() => other.loadState(state)).toThrow(TsDspyError);
        expect(() => other.loadState(state)).toThrow(
            /unknown predictor path\(s\): [^;]*\bgenerate\b/
        );
        expect(() => other.loadState(state)).toThrow(/missing predictor path\(s\): retrieve/);
    });

    it('rejects a malformed entry without changing anything', () => {
        const outer = compiledOuter();
        const state = outer.dumpState() as Record<string, any>;
        state['inner.retrieve'] = { demos: 'nope' };

        expect(() => outer.loadState(state)).toThrow(/inner\.retrieve.*malformed/);
        expect(outer.generate.getInstructions()).toBe('Answer in one word.');
    });

    it('rejects a file that is not JSON', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'ts-dspy-state-'));
        try {
            const file = join(dir, 'bad.json');
            await (await import('node:fs/promises')).writeFile(file, '{ nope');
            await expect(new Outer().load(file)).rejects.toThrow(/not valid JSON/);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });
});
