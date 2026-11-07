import { describe, expect, it } from 'vitest';
import { KNNFewShot, KNNProgram, defaultVectorizeInputs } from './knn-few-shot';
import { Predict } from '../modules/predict';
import type { Embedder } from '../core/embedder';
import {
    demoQuestions,
    doublingLM,
    doublingSet,
    exactAnswer,
    numberIn,
    viewPrompt,
} from './test-support';

/**
 * Embeds "double n" on the unit circle at angle n/10, so cosine similarity
 * falls off with |a - b| and the nearest numbers are the nearest neighbours.
 */
function circleEmbedder(): Embedder & { calls: string[][] } {
    const calls: string[][] = [];
    return {
        calls,
        async embed(texts) {
            calls.push(texts);
            return texts.map((text) => {
                const n = numberIn(text);
                return [Math.cos(n / 10), Math.sin(n / 10)];
            });
        },
    };
}

const trainset = doublingSet([1, 5, 10, 20, 40]);

describe('KNNFewShot', () => {
    it('shows each call the k nearest training examples as demos', async () => {
        const lm = doublingLM(() => true);
        const embedder = circleEmbedder();
        const student = new Predict('question -> answer', lm);
        const program = new KNNFewShot({ k: 2, embedder, trainset }).compile(student);

        await program.forward({ question: 'double 9' });
        expect(demoQuestions(viewPrompt(lm.calls.at(-1)!.messages))).toEqual([
            'double 10',
            'double 5',
        ]);

        await program.forward({ question: 'double 33' });
        expect(demoQuestions(viewPrompt(lm.calls.at(-1)!.messages))).toEqual([
            'double 40',
            'double 20',
        ]);

        // The training set is embedded once; each call embeds only its inputs.
        expect(embedder.calls.map((texts) => texts.length)).toEqual([5, 1, 1]);
        expect(embedder.calls[1]).toEqual(['question: double 9']);
        expect(student.getDemos()).toEqual([]);
    });

    it('wraps the student so its predictors keep a stable path', async () => {
        const program = new KNNFewShot({ k: 1, embedder: circleEmbedder(), trainset }).compile(
            new Predict('question -> answer')
        );
        expect(program).toBeInstanceOf(KNNProgram);
        expect(program.namedPredictors().map(([path]) => path)).toEqual(['program']);
        const neighbours = await program.neighbours({ question: 'double 19' });
        expect(neighbours[0].example.get('question')).toBe('double 20');
        expect(neighbours[0].similarity).toBeCloseTo(Math.cos(0.1));
    });

    it('can bootstrap the neighbours first, keeping only the ones the program gets right', async () => {
        // Only even numbers come out right, so of the neighbours 10 and 5 only 10 survives.
        const lm = doublingLM((_view, n) => n % 2 === 0);
        const program = new KNNFewShot({
            k: 2,
            embedder: circleEmbedder(),
            trainset,
            bootstrap: { metric: exactAnswer, maxBootstrappedDemos: 2 },
        }).compile(new Predict('question -> answer', lm));

        await program.forward({ question: 'double 9' });
        expect(demoQuestions(viewPrompt(lm.calls.at(-1)!.messages))).toEqual(['double 10']);
    });

    it('validates its options', () => {
        expect(() => new KNNFewShot({ k: 1, trainset } as never)).toThrow(/embedder/);
        expect(() => new KNNFewShot({ k: -1, embedder: circleEmbedder(), trainset })).toThrow(
            RangeError
        );
    });

    it('vectorizes inputs as sorted field lines by default', () => {
        expect(defaultVectorizeInputs({ b: 'two', a: 1 })).toBe('a: 1\nb: two');
    });
});
