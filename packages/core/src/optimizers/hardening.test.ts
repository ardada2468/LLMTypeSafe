/** Regression tests for optimizer and evaluation failures found in a production-readiness review. */
import { Example } from '../core/example';
import { AuthError, TsDspyError } from '../core/errors';
import { Prediction } from '../core/prediction';
import { evaluate } from '../evaluate/evaluate';
import { exactMatch, tokenF1 } from '../evaluate/metrics';
import { Predict } from '../modules/predict';
import { MockLM } from '../testing/mock-lm';
import { COPRO } from './copro';
import { BootstrapFewShot } from './bootstrap-few-shot';
import { KNNFewShot } from './knn-few-shot';
import { MIPROv2 } from './mipro';
import { BootstrapFewShotWithRandomSearch } from './random-search';
import {
    demoQuestions,
    doublingLM,
    doublingSet,
    exactAnswer,
    reply,
    viewPrompt,
} from './test-support';
import { evaluateProgram, transportOptions } from './utils';

const QA = 'question -> answer';

function unauthorised(): MockLM {
    const lm = new MockLM();
    vi.spyOn(lm, 'chat').mockRejectedValue(new AuthError('mock', '401 invalid api key'));
    return lm;
}

describe('evaluate', () => {
    it('stops once maxErrors is exceeded, with the first failure as cause', async () => {
        const lm = unauthorised();
        const run = evaluate(
            new Predict(QA, lm),
            doublingSet([1, 2, 3, 4, 5, 6, 7, 8]),
            exactAnswer,
            {
                maxErrors: 2,
                concurrency: 1,
            }
        );

        await expect(run).rejects.toThrow(/maxErrors limit of 2/);
        await expect(run).rejects.toMatchObject({ cause: expect.any(AuthError) });
        expect(lm.chat).toHaveBeenCalledTimes(3);
    });

    it('stops when the call signal aborts', async () => {
        const controller = new AbortController();
        controller.abort(new Error('cancelled'));
        const lm = doublingLM(() => true);

        await expect(
            evaluate(new Predict(QA, lm), doublingSet([1, 2, 3]), exactAnswer, {
                callOptions: { signal: controller.signal },
            })
        ).rejects.toThrow('cancelled');
        expect(lm.calls).toHaveLength(0);
    });

    it('scores failures with failureScore', async () => {
        const report = await evaluate(
            new Predict(QA, unauthorised()),
            doublingSet([1, 2]),
            exactAnswer,
            {
                failureScore: -1,
            }
        );
        expect(report.score).toBe(-1);
        expect(report.errorCount).toBe(2);
    });
});

describe('evaluateProgram', () => {
    it('throws when every example failed at the provider', async () => {
        await expect(
            evaluateProgram(
                new Predict(QA, unauthorised()),
                doublingSet([1, 2, 3]),
                exactAnswer
            )
        ).rejects.toThrow(/Every one of 3 examples failed with a provider error/);
    });

    it('still scores a candidate whose output merely failed validation', async () => {
        const lm = new MockLM({ respond: () => 'no fields here at all' });
        const evaluation = await evaluateProgram(
            new Predict('question -> answer: int', lm),
            doublingSet([1, 2]),
            exactAnswer
        );
        expect(evaluation.score).toBe(0);
    });
});

describe('optimizers', () => {
    it('keeps the student sampling options away from the prompt model', async () => {
        expect(
            transportOptions({
                maxTokens: 7,
                stopSequences: ['\n'],
                temperature: 0,
                timeout: 5,
            })
        ).toEqual({ timeout: 5 });

        const promptModel = new MockLM({
            respond: () => reply({ proposed_instruction: 'Double the number.' }),
        });
        await new COPRO({
            metric: exactAnswer,
            breadth: 2,
            depth: 1,
            promptModel,
            callOptions: { maxTokens: 7, stopSequences: ['\n'] },
        }).compile(
            new Predict(
                QA,
                doublingLM(() => true)
            ),
            { trainset: doublingSet([1, 2]) }
        );

        expect(promptModel.calls.length).toBeGreaterThan(0);
        for (const call of promptModel.calls) {
            expect(call.options?.maxTokens).toBeUndefined();
            expect(call.options?.stopSequences).toBeUndefined();
        }
    });

    it('MIPROv2 hands back the student when its own demos beat everything searched', async () => {
        const lm = doublingLM((view) => demoQuestions(view).includes('double 100'));
        const student = new Predict(QA, lm);
        student.setDemos([
            new Example({ question: 'double 100', answer: '200' }).withInputs('question'),
        ]);
        const promptModel = new MockLM({
            respond: (messages) =>
                viewPrompt(messages).outputs.includes('observations')
                    ? reply({ observations: 'Doubling.' })
                    : reply({ proposed_instruction: 'Answer the question.' }),
        });

        const optimizer = new MIPROv2({
            metric: exactAnswer,
            auto: null,
            numCandidates: 4,
            numTrials: 6,
            promptModel,
        });
        const compiled = await optimizer.compile(student, {
            trainset: doublingSet([1, 2, 3, 4]),
        });

        expect(optimizer.results!.bestScore).toBe(1);
        expect(compiled.getDemos()).toHaveLength(1);
    });

    it('random search bootstraps nothing when maxBootstrappedDemos is 0', async () => {
        const teacher = doublingLM(() => true);
        const optimizer = new BootstrapFewShotWithRandomSearch({
            metric: exactAnswer,
            maxBootstrappedDemos: 0,
            maxLabeledDemos: 0,
            numCandidatePrograms: 3,
            teacher,
        });
        await optimizer.compile(
            new Predict(
                QA,
                doublingLM(() => true)
            ),
            {
                trainset: doublingSet([1, 2, 3, 4]),
            }
        );

        expect(teacher.calls).toHaveLength(0);
        expect(
            () =>
                new BootstrapFewShotWithRandomSearch({
                    metric: exactAnswer,
                    numCandidatePrograms: -3,
                })
        ).toThrow(RangeError);
    });
});

describe('tracing programs that run per-call copies', () => {
    it('bootstraps demos through a KNNFewShot program', async () => {
        const trainset = doublingSet([1, 2, 3, 4]);
        const embedder = {
            embed: async (texts: string[]) => texts.map((t) => [t.length, 1]),
        };
        const program = new KNNFewShot({ k: 1, embedder, trainset }).compile(
            new Predict(
                QA,
                doublingLM(() => true)
            )
        );

        const optimizer = new BootstrapFewShot({
            metric: exactAnswer,
            maxBootstrappedDemos: 2,
        });
        const compiled = await optimizer.compile(program, { trainset });

        const [[path, predictor]] = compiled.namedPredictors();
        expect(path).toBe('program');
        expect(predictor.getDemos().length).toBeGreaterThan(0);
    });
});

describe('metrics', () => {
    const pair = (expected: unknown, actual: unknown) =>
        [
            new Example({ q: 'q', a: expected }).withInputs('q'),
            new Prediction({ a: actual }),
        ] as const;

    it('tokenF1 does not treat every object as the same text', () => {
        expect(
            tokenF1(...pair({ city: 'Paris' }, { totally: 'different', n: 5 }))
        ).toBeLessThan(1);
        expect(tokenF1(...pair({ city: 'Paris' }, { city: 'Paris' }))).toBe(1);
    });

    it('exactMatch compares dates by value', () => {
        expect(exactMatch(...pair(new Date(0), new Date(86_400_000)))).toBe(false);
        expect(exactMatch(...pair(new Date(0), new Date(0)))).toBe(true);
    });
});

describe('error type', () => {
    it('is a TsDspyError', async () => {
        await expect(
            evaluateProgram(new Predict(QA, unauthorised()), doublingSet([1]), exactAnswer)
        ).rejects.toBeInstanceOf(TsDspyError);
    });
});
