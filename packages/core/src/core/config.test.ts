import { configure, getDefaultLM, isCacheEnabled, isTracingEnabled } from './config';
import { Predict } from '../modules/predict';
import { MockLM } from '../test-utils';

describe('configure', () => {
    it('makes a language model available as the default', () => {
        const lm = new MockLM();
        configure({ lm });

        expect(getDefaultLM()).toBe(lm);
    });

    it('is used by modules constructed without an explicit model', async () => {
        const lm = new MockLM({ responses: ['answer: Paris'] });
        configure({ lm });

        const result = await new Predict('question -> answer').forward({ question: 'Q' });

        expect(result.answer).toBe('Paris');
        expect(lm.calls).toHaveLength(1);
    });

    it('lets an explicit model override the default', async () => {
        const defaultLM = new MockLM({ responses: ['answer: from default'] });
        const explicitLM = new MockLM({ responses: ['answer: from explicit'] });
        configure({ lm: defaultLM });

        const result = await new Predict('question -> answer', explicitLM).forward({
            question: 'Q',
        });

        expect(result.answer).toBe('from explicit');
        expect(defaultLM.calls).toHaveLength(0);
    });

    it('toggles cache and tracing flags', () => {
        configure({ lm: new MockLM(), cache: false, tracing: true });

        expect(isCacheEnabled()).toBe(false);
        expect(isTracingEnabled()).toBe(true);

        configure({ cache: true, tracing: false });
        expect(isCacheEnabled()).toBe(true);
        expect(isTracingEnabled()).toBe(false);
    });
});
