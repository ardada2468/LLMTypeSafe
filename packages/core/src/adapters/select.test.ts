import { ChatAdapter } from './chat';
import { JSONAdapter } from './json';
import { XMLAdapter } from './xml';
import { defaultAdapter, resolveAdapter } from './select';
import { Predict } from '../modules/predict';
import { configure, context } from '../core/config';
import { MockLM } from '../test-utils';

/** An adapter that records that it was chosen. */
class Tagged extends ChatAdapter {
    constructor(readonly tag: string) {
        super();
    }
}

describe('adapter selection', () => {
    afterEach(() => configure({ adapter: undefined }));

    it('defaults to ChatAdapter, or JSONAdapter for a structured-output model', () => {
        expect(defaultAdapter(new MockLM())).toBeInstanceOf(ChatAdapter);
        expect(
            defaultAdapter(new MockLM({ capabilities: { supportsStructuredOutput: true } }))
        ).toBeInstanceOf(JSONAdapter);
    });

    it('prefers call option > module > context > global > default', () => {
        const lm = new MockLM();
        const call = new Tagged('call');
        const module = new Tagged('module');
        const scoped = new Tagged('context');
        const global = new Tagged('global');

        configure({ adapter: global });
        expect(resolveAdapter(lm)).toBe(global);
        context({ adapter: scoped }, () => {
            expect(resolveAdapter(lm)).toBe(scoped);
            expect(resolveAdapter(lm, undefined, module)).toBe(module);
            expect(resolveAdapter(lm, call, module)).toBe(call);
        });
    });

    it('is honoured end to end by Predict', async () => {
        const xml = '<answer>Paris</answer>';
        const lm = new MockLM({ respond: () => xml });
        const predict = new Predict('question -> answer', { lm, adapter: new XMLAdapter() });

        // Module option.
        await predict.forward({ question: 'Q' });
        expect(String(lm.calls[0].messages[0].content)).toContain('<answer>');

        // Call option beats it, and is not passed on to the provider.
        lm.setResponses(['{"answer": "Paris"}']);
        await predict.forward(
            { question: 'Q' },
            { adapter: new JSONAdapter(), temperature: 0 }
        );
        expect(String(lm.calls[0].messages[0].content)).toContain('Respond with JSON');
        expect(lm.calls[0].options).toEqual({ temperature: 0 });

        // A context scope applies to modules without their own adapter.
        lm.setResponses([]);
        const plain = new Predict('question -> answer', lm);
        await context({ adapter: new XMLAdapter() }, () => plain.forward({ question: 'Q' }));
        expect(String(lm.calls[0].messages[0].content)).toContain('<answer>');
    });

    it('carries the module adapter through withDemos and withAdapter', () => {
        const xml = new XMLAdapter();
        const predict = new Predict('question -> answer', { adapter: xml });

        expect(predict.withDemos([]).getAdapter()).toBe(xml);
        const json = new JSONAdapter();
        expect(predict.withAdapter(json).getAdapter()).toBe(json);
        expect(predict.getAdapter()).toBe(xml);
    });
});
