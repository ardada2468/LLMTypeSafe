import { z } from 'zod';
import { configure, context } from './config';
import { Module } from './module';
import { type Prediction } from './prediction';
import { Tool } from './tool';
import { clearHistory, inspectHistory } from './trace';
import type {
    Callback,
    LMEndEvent,
    LMStartEvent,
    ModuleEndEvent,
    ModuleStartEvent,
    ToolEndEvent,
    ToolStartEvent,
} from './callbacks';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { ReAct } from '../modules/react';
import { MockLM } from '../test-utils';
import { EchoLM } from './test-lms';

/** Records every event, in order. */
class Recorder implements Callback {
    readonly events: Array<{ hook: string; event: any }> = [];
    onModuleStart(event: ModuleStartEvent) {
        this.events.push({ hook: 'moduleStart', event });
    }
    onModuleEnd(event: ModuleEndEvent) {
        this.events.push({ hook: 'moduleEnd', event });
    }
    onLMStart(event: LMStartEvent) {
        this.events.push({ hook: 'lmStart', event });
    }
    onLMEnd(event: LMEndEvent) {
        this.events.push({ hook: 'lmEnd', event });
    }
    onToolStart(event: ToolStartEvent) {
        this.events.push({ hook: 'toolStart', event });
    }
    onToolEnd(event: ToolEndEvent) {
        this.events.push({ hook: 'toolEnd', event });
    }
    of(hook: string): any[] {
        return this.events.filter((e) => e.hook === hook).map((e) => e.event);
    }
}

/** A composite whose forward calls its children's `forward` directly. */
class Pipeline extends Module {
    first = new Predict('question -> answer');
    second = new Predict('question -> answer');

    async forward(inputs: Record<string, any>): Promise<Prediction> {
        const a = await this.first.forward(inputs);
        return this.second.forward({ question: `${a.answer}2` });
    }
}

describe('callbacks', () => {
    afterEach(() => {
        configure({ callbacks: [], tracing: false });
        clearHistory();
    });

    it('reports module and LM events for a single Predict', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new MockLM({ responses: ['answer: Paris'] });
        const predict = new Predict('question -> answer', lm);

        await predict.forward({ question: 'Capital?' });

        expect(recorder.events.map((e) => e.hook)).toEqual([
            'moduleStart',
            'lmStart',
            'lmEnd',
            'moduleEnd',
        ]);
        const [moduleStart] = recorder.of('moduleStart');
        const [lmStart] = recorder.of('lmStart');
        const [lmEnd] = recorder.of('lmEnd');
        const [moduleEnd] = recorder.of('moduleEnd');

        expect(moduleStart.instance).toBe(predict);
        expect(moduleStart.moduleId).toBe(predict.moduleId);
        expect(moduleStart.inputs).toEqual({ question: 'Capital?' });
        expect(moduleStart.parentCallId).toBeUndefined();

        expect(lmStart.parentCallId).toBe(moduleStart.callId);
        // Adapters speak chat: the prompt travels as messages.
        expect(lmStart.method).toBe('chat');
        expect(lmStart.messages.at(-1).content).toContain('Capital?');
        expect(lmStart.moduleId).toBe(predict.moduleId);
        expect(lmEnd.callId).toBe(lmStart.callId);
        expect(lmEnd.outputs).toBe('answer: Paris');
        expect(lmEnd.usage['mock/mock-model']).toMatchObject({
            promptTokens: 10,
            completionTokens: 5,
            requestCount: 1,
        });

        expect(moduleEnd.callId).toBe(moduleStart.callId);
        expect(moduleEnd.outputs).toEqual({ answer: 'Paris' });
        expect(moduleEnd.usage['mock/mock-model'].totalTokens).toBe(15);
        expect(moduleEnd.durationMs).toBeGreaterThanOrEqual(0);
    });

    it('nests events for a composite whose forward calls children directly', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new EchoLM();
        const pipeline = new Pipeline();

        const result = await context({ lm }, () => pipeline.forward({ question: 'q' }));

        expect(result.get('answer')).toBe('q2');
        const starts = recorder.of('moduleStart');
        expect(starts.map((e) => e.instance)).toEqual([
            pipeline,
            pipeline.first,
            pipeline.second,
        ]);
        expect(starts[1].parentCallId).toBe(starts[0].callId);
        expect(starts[2].parentCallId).toBe(starts[0].callId);

        const lmStarts = recorder.of('lmStart');
        expect(lmStarts[0].parentCallId).toBe(starts[1].callId);
        expect(lmStarts[1].parentCallId).toBe(starts[2].callId);
    });

    it('keeps parent ids correct for concurrent invocations', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        // The first pipeline's calls are the slowest, so the two runs interleave.
        const lm = new EchoLM({ a: { delayMs: 25 }, a2: { delayMs: 5 }, b: { delayMs: 1 } });
        const one = new Pipeline();
        const two = new Pipeline();

        await context({ lm }, () =>
            Promise.all([one.forward({ question: 'a' }), two.forward({ question: 'b' })])
        );

        const starts = recorder.of('moduleStart');
        const byInstance = (module: Module) => starts.find((e) => e.instance === module);
        expect(byInstance(one.first).parentCallId).toBe(byInstance(one).callId);
        expect(byInstance(one.second).parentCallId).toBe(byInstance(one).callId);
        expect(byInstance(two.first).parentCallId).toBe(byInstance(two).callId);
        expect(byInstance(two.second).parentCallId).toBe(byInstance(two).callId);

        // Every LM call is parented by the Predict that made it.
        for (const lmStart of recorder.of('lmStart')) {
            const parent = starts.find((e) => e.callId === lmStart.parentCallId);
            expect(parent.moduleId).toBe(lmStart.moduleId);
        }
    });

    it('emits once per invocation through call(), __call__ and batch()', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new MockLM({
            responses: ['answer: 1', 'answer: 2', 'answer: 3', 'answer: 4'],
        });
        const predict = new Predict('question -> answer', lm);

        await predict.call({ question: 'a' });
        await predict.__call__({ question: 'b' });
        await predict.batch([{ question: 'c' }, { question: 'd' }]);

        expect(recorder.of('moduleStart')).toHaveLength(4);
        expect(recorder.of('moduleEnd')).toHaveLength(4);
    });

    it('emits for copies made by withDemos and deepCopy', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new MockLM({ responses: ['answer: 1', 'answer: 2'] });
        const predict = new Predict('question -> answer', lm);

        await predict.withDemos([]).forward({ question: 'a' });
        await predict.deepCopy().forward({ question: 'b' });

        expect(recorder.of('moduleStart')).toHaveLength(2);
    });

    it('reports a failing invocation with its error', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new MockLM({ responses: [] });

        await expect(
            new Predict('question -> answer', lm).forward({ question: 'Q' })
        ).rejects.toThrow('no more scripted responses');

        expect(recorder.of('lmEnd')[0].error).toBeInstanceOf(Error);
        expect(recorder.of('moduleEnd')[0].error).toBeInstanceOf(Error);
        expect(recorder.of('moduleEnd')[0].outputs).toBeUndefined();
    });

    it('swallows exceptions thrown by callbacks', async () => {
        const after = new Recorder();
        configure({
            callbacks: [
                {
                    onModuleStart: () => {
                        throw new Error('broken observer');
                    },
                    onLMEnd: () => {
                        throw new Error('broken observer');
                    },
                },
                after,
            ],
        });
        const lm = new MockLM({ responses: ['answer: ok'] });

        const result = await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(result.answer).toBe('ok');
        // A later callback still hears everything.
        expect(after.events).toHaveLength(4);
    });

    it('uses scoped callbacks only inside the scope', async () => {
        const global = new Recorder();
        const scoped = new Recorder();
        configure({ callbacks: [global] });
        const lm = new MockLM({ responses: ['answer: 1', 'answer: 2'] });
        const predict = new Predict('question -> answer', lm);

        await context({ callbacks: [scoped] }, () => predict.forward({ question: 'a' }));
        await predict.forward({ question: 'b' });

        expect(scoped.of('moduleStart')).toHaveLength(1);
        expect(global.of('moduleStart')).toHaveLength(1);
    });

    it('reports the single ChainOfThought call', async () => {
        // Changed in 0.7: ChainOfThought makes one call, not two.
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new MockLM({ responses: ['reasoning: because\nanswer: A'] });

        await new ChainOfThought('question -> answer', lm).forward({ question: 'Q' });

        expect(recorder.of('lmStart')).toHaveLength(1);
        expect(recorder.of('moduleStart')).toHaveLength(1);
    });

    it('reports adapter format and parse events', async () => {
        const events: Array<{ hook: string; event: any }> = [];
        configure({
            callbacks: [
                {
                    onAdapterFormat: (event) => events.push({ hook: 'format', event }),
                    onAdapterParse: (event) => events.push({ hook: 'parse', event }),
                },
            ],
        });
        const lm = new MockLM({ responses: ['[[ ## answer ## ]]\nParis'] });

        await new Predict('question -> answer', lm).forward({ question: 'Q' });

        expect(events.map((e) => e.hook)).toEqual(['format', 'parse']);
        expect(events[0].event.adapter).toBe('ChatAdapter');
        expect(events[0].event.inputs).toEqual({ question: 'Q' });
        expect(events[0].event.outputs.at(-1).role).toBe('user');
        expect(events[1].event.inputs).toBe('[[ ## answer ## ]]\nParis');
        expect(events[1].event.outputs).toEqual({ answer: 'Paris' });
    });

    it('reports tool calls made by ReAct, nested under the module', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const lm = new MockLM({
            responses: [
                'next_thought: look it up\nnext_tool_name: weather\nnext_tool_args: {"input": "Paris"}',
                'next_thought: done\nnext_tool_name: finish\nnext_tool_args: {}',
                'reasoning: the tool said so\nanswer: sunny',
            ],
        });
        const agent = new ReAct('question -> answer', {
            lm,
            tools: { weather: async (city: string) => `sunny in ${city}` },
        });

        await agent.forward({ question: 'Weather?' });

        const [toolStart] = recorder.of('toolStart');
        const [toolEnd] = recorder.of('toolEnd');
        const [moduleStart] = recorder.of('moduleStart');
        expect(toolStart.toolName).toBe('weather');
        expect(toolStart.args).toEqual({ input: 'Paris' });
        expect(toolStart.parentCallId).toBe(moduleStart.callId);
        expect(toolEnd.callId).toBe(toolStart.callId);
        expect(toolEnd.outputs).toBe('sunny in Paris');
        // Two steps and the extract call.
        expect(recorder.of('lmStart')).toHaveLength(3);
    });

    it('reports a tool whose arguments fail validation', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder] });
        const tool = Tool.fromFunction(({ n }: { n: number }) => n * 2, {
            name: 'double',
            parameters: z.object({ n: z.number() }),
        });

        await expect(tool.call({ n: 'two' })).rejects.toThrow(/Invalid arguments/);
        expect(recorder.of('toolEnd')[0].error).toBeInstanceOf(Error);
    });

    it('nests trace entries using the same ids callbacks see', async () => {
        const recorder = new Recorder();
        configure({ callbacks: [recorder], tracing: true });
        const lm = new EchoLM();
        const pipeline = new Pipeline();

        await context({ lm }, () => pipeline.forward({ question: 'q' }));

        const entries = inspectHistory();
        const starts = recorder.of('moduleStart');
        expect(entries).toHaveLength(2);
        expect(entries[0].id).toBe(starts[1].callId);
        expect(entries[0].parentId).toBe(starts[0].callId);
        expect(entries[1].parentId).toBe(starts[0].callId);
    });
});
