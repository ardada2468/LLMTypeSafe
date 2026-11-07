import { MultiChainComparison } from './multi-chain-comparison';
import { Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import { MockLM } from '../test-utils';
import { contentToText } from '../utils/content';

const reply =
    '[[ ## rationale ## ]]\nAll agree on 4.\n\n[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]';

const attempts = [
    { reasoning: 'add 2 and 2\nthen check', answer: '4' },
    new Prediction({ reasoning: 'two plus two', answer: '4\nI think' }),
    { rationale: 'count fingers', answer: '5' },
];

describe('MultiChainComparison', () => {
    it('compares m attempts and returns a rationale with the outputs', async () => {
        const lm = new MockLM({ responses: [reply] });
        const result = await new MultiChainComparison('question -> answer', lm).forward({
            question: 'What is 2 + 2?',
            completions: attempts,
        });

        expect(result.rationale).toBe('All agree on 4.');
        expect(result.answer).toBe('4');
        expect(lm.calls).toHaveLength(1);
    });

    it('summarises each attempt as DSPy does, first lines only', async () => {
        const lm = new MockLM({ responses: [reply] });
        await new MultiChainComparison('question -> answer', lm).forward({
            question: 'Q',
            completions: attempts,
        });

        const prompt = lm.calls[0].messages.map((m) => contentToText(m.content)).join('\n');
        expect(prompt).toContain('[[ ## reasoning_attempt_1 ## ]]');
        expect(prompt).toContain('[[ ## reasoning_attempt_3 ## ]]');
        expect(prompt).toContain(
            "«I'm trying to add 2 and 2 I'm not sure but my prediction is 4»"
        );
        expect(prompt).toContain(
            "«I'm trying to two plus two I'm not sure but my prediction is 4»"
        );
        expect(prompt).toContain(
            "«I'm trying to count fingers I'm not sure but my prediction is 5»"
        );
        expect(prompt).not.toContain('then check');
    });

    it('extends the signature with attempt inputs and a leading rationale output', () => {
        const mcc = new MultiChainComparison('question -> answer', { m: 2 });
        const [[, predictor]] = mcc.namedPredictors();
        const sig = (predictor as any).signature;

        expect(Object.keys(sig.getInputFields())).toEqual([
            'question',
            'reasoning_attempt_1',
            'reasoning_attempt_2',
        ]);
        expect(Object.keys(sig.getOutputFields())).toEqual(['rationale', 'answer']);
    });

    it('defaults the comparison temperature to 0.7, beneath the call’s own', async () => {
        const lm = new MockLM({ responses: [reply, reply] });
        const mcc = new MultiChainComparison('question -> answer', lm);

        await mcc.forward({ question: 'Q', completions: attempts });
        await mcc.forward({ question: 'Q', completions: attempts }, { temperature: 0 });

        expect(lm.calls.map((call) => call.options?.temperature)).toEqual([0.7, 0]);
    });

    it('rejects the wrong number of completions', async () => {
        const mcc = new MultiChainComparison('question -> answer', new MockLM());
        await expect(
            mcc.forward({ question: 'Q', completions: attempts.slice(0, 2) })
        ).rejects.toThrow(TsDspyError);
    });

    it('validates m', () => {
        expect(() => new MultiChainComparison('question -> answer', { m: 0 })).toThrow(
            RangeError
        );
    });
});
