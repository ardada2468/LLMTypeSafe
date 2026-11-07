import { streamify, type FieldStreamEvent, type StreamEvent } from './streaming';
import { Module } from './module';
import { Prediction } from './prediction';
import { BaseLM } from './base-lm';
import { Tool } from './tool';
import { configure, context } from './config';
import { clearHistory, inspectHistory } from './trace';
import { ValidationError } from './errors';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { MockLM } from '../test-utils';
import type { Callback } from './callbacks';
import type {
    ChatMessage,
    LLMCallOptions,
    ModelCapabilities,
    StreamChunk,
} from '../types/language-model';

const chat = (fields: Record<string, string>) =>
    Object.entries(fields)
        .map(([name, value]) => `[[ ## ${name} ## ]]\n${value}\n\n`)
        .join('') + '[[ ## completed ## ]]';

async function collect<T>(stream: AsyncGenerator<T, unknown, void>): Promise<T[]> {
    const events: T[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

const fields = (events: StreamEvent[]) =>
    events.filter((event): event is FieldStreamEvent => event.type === 'field');

/** Streams scripted chunks and records whether the consumer closed it early. */
class ClosableLM extends BaseLM {
    emitted = 0;
    closed = false;

    constructor(private readonly chunks: string[]) {
        super('closable', 'fake-1');
    }

    async chat(): Promise<string> {
        return this.chunks.join('');
    }

    async *chatStream(): AsyncGenerator<StreamChunk, void, unknown> {
        try {
            for (const chunk of this.chunks) {
                this.emitted += 1;
                yield { content: chunk, done: false };
            }
            yield { content: '', done: true };
        } finally {
            this.closed = true;
        }
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: false,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
        };
    }
}

describe('streamify', () => {
    it('streams a predictor’s fields as deltas, then the prediction', async () => {
        const lm = new MockLM({
            responses: [chat({ answer: 'The capital is Paris.' })],
            chunkSize: 4,
        });
        const program = new Predict('question -> answer', lm);

        const events = await collect(streamify(program)({ question: 'Capital of France?' }));

        const answer = fields(events).filter((event) => event.field === 'answer');
        expect(answer.length).toBeGreaterThan(2);
        expect(answer.every((event) => event.path === 'self')).toBe(true);
        expect(answer.map((event) => event.delta ?? '').join('')).toBe('The capital is Paris.');
        expect(answer.at(-1)).toMatchObject({ done: true, value: 'The capital is Paris.' });

        const last = events.at(-1)!;
        expect(last.type).toBe('prediction');
        expect((last as { prediction: Prediction }).prediction.get('answer')).toBe(
            'The capital is Paris.'
        );
    });

    it('labels each predictor of a composite by its path, through a forward that drops options', async () => {
        const lm = new MockLM({
            responses: [
                chat({ draft: 'rough notes' }),
                chat({ reasoning: 'tidy', answer: 'final' }),
            ],
            chunkSize: 3,
        });
        class Pipeline extends Module {
            draft = new Predict('question -> draft', lm);
            polish = new ChainOfThought('draft -> answer', lm);

            async forward(inputs: Record<string, any>, _options?: LLMCallOptions) {
                const { draft } = await this.draft.forward(inputs);
                return this.polish.forward({ draft });
            }
        }

        const events = fields(await collect(streamify(new Pipeline())({ question: 'Q' })));

        const order = [...new Set(events.map((event) => `${event.path}.${event.field}`))];
        expect(order).toEqual(['draft.draft', 'polish.reasoning', 'polish.answer']);
    });

    it('emits status messages for tool calls by default', async () => {
        const lookup = Tool.fromFunction(
            async ({ city }: { city: string }) => `${city}: sunny`,
            {
                name: 'weather',
            }
        );
        class Agent extends Module {
            async forward(inputs: Record<string, any>) {
                return new Prediction({ report: await lookup.call({ city: inputs.city }) });
            }
        }

        const events = await collect(streamify(new Agent())({ city: 'Paris' }));

        expect(events.filter((event) => event.type === 'status')).toEqual([
            { type: 'status', message: 'Calling tool weather...' },
            {
                type: 'status',
                message: 'Tool calling finished! Querying the LLM with tool calling results...',
            },
        ]);
    });

    it('takes a custom status message provider, and can leave out the prediction', async () => {
        const lm = new MockLM({ responses: [chat({ answer: 'ok' })] });
        const stream = streamify(new Predict('question -> answer', lm), {
            statusMessageProvider: {
                moduleStart: (event) => `Running ${event.moduleId.split('#')[0]}`,
                lmEnd: () => 'Model replied',
            },
            includeFinal: false,
        });

        const events = await collect(stream({ question: 'Q' }));

        const statuses = events.filter((event) => event.type === 'status');
        expect(statuses).toEqual([
            { type: 'status', message: 'Running Predict' },
            { type: 'status', message: 'Model replied' },
        ]);
        expect(events.some((event) => event.type === 'prediction')).toBe(false);
    });

    it('keeps the callbacks already in scope', async () => {
        const seen: string[] = [];
        const existing: Callback = { onModuleStart: (event) => seen.push(event.moduleId) };
        const lm = new MockLM({ responses: [chat({ answer: 'ok' })] });
        const program = new Predict('question -> answer', lm);

        await context({ callbacks: [existing] }, () =>
            collect(streamify(program)({ question: 'Q' }))
        );

        expect(seen).toEqual([program.moduleId]);
    });

    it('delivers fields once, complete, from a model that cannot stream', async () => {
        const lm = new MockLM({
            responses: [chat({ answer: 'Paris' })],
            capabilities: { supportsStreaming: false },
        });

        const events = fields(
            await collect(streamify(new Predict('question -> answer', lm))({ question: 'Q' }))
        );

        expect(events).toEqual([
            {
                type: 'field',
                path: 'self',
                field: 'answer',
                value: 'Paris',
                delta: 'Paris',
                done: true,
            },
        ]);
    });

    it('closes the provider stream and settles the program when the consumer breaks', async () => {
        const lm = new ClosableLM(['[[ ## answer ## ]]\n', 'one ', 'two ', 'three ', 'four']);
        const finished = vi.fn();
        class Wrapper extends Module {
            inner = new Predict('question -> answer', lm);
            async forward(inputs: Record<string, any>) {
                try {
                    return await this.inner.forward(inputs);
                } finally {
                    finished();
                }
            }
        }

        for await (const event of streamify(new Wrapper())({ question: 'Q' })) {
            if (event.type === 'field') break;
        }

        expect(lm.closed).toBe(true);
        expect(lm.emitted).toBeLessThan(5);
        expect(finished).toHaveBeenCalledTimes(1);
    });

    it('rethrows the program’s error after the events before it', async () => {
        const lm = new MockLM({ responses: ['[[ ## answer ## ]]\npartial'], chunkSize: 5 });
        const stream = streamify(new Predict('question -> answer: int', lm));

        const events: StreamEvent[] = [];
        await expect(
            (async () => {
                for await (const event of stream({ question: 'Q' })) events.push(event);
            })()
        ).rejects.toBeInstanceOf(ValidationError);
        expect(fields(events).length).toBeGreaterThan(0);
    });

    it('stops on the caller’s abort signal', async () => {
        await expect(
            collect(
                streamify(new Predict('question -> answer', new MockLM()))(
                    { question: 'Q' },
                    { signal: AbortSignal.abort(new Error('cancelled')) }
                )
            )
        ).rejects.toThrow('cancelled');
    });
});

describe('Predict.stream instrumentation', () => {
    afterEach(() => {
        configure({ tracing: false, callbacks: [] });
        clearHistory();
    });

    it('emits module callbacks around the stream, like forward', async () => {
        const events: string[] = [];
        configure({
            callbacks: [
                {
                    onModuleStart: () => events.push('module:start'),
                    onLMStart: (event) => events.push(`lm:${event.method}`),
                    onModuleEnd: (event) => events.push(`module:end:${event.outputs?.answer}`),
                },
            ],
        });
        const lm = new MockLM({ responses: [chat({ answer: 'Paris' })] });

        const stream = new Predict('question -> answer', lm).stream({ question: 'Q' });
        await collect(stream);

        expect(events).toEqual(['module:start', 'lm:chatStream', 'module:end:Paris']);
    });

    it('records a trace entry with the streamed call', async () => {
        configure({ tracing: true });
        clearHistory();
        const lm = new MockLM({ responses: [chat({ answer: 'Paris' })] });

        const stream = new Predict('question -> answer', lm).stream({ question: 'Q' });
        let result = await stream.next();
        while (!result.done) result = await stream.next();

        const [entry] = inspectHistory();
        expect(entry!.output).toEqual({ answer: 'Paris' });
        expect(entry!.calls).toHaveLength(1);
        expect(entry!.rawLMOutput).toContain('Paris');
        expect(result.value.trace).toBe(entry);
    });

    it('repairs a streamed reply that fails validation when repairAttempts allows', async () => {
        const lm = new MockLM({
            responses: [chat({ answer: 'many' }), chat({ answer: '3' })],
        });

        const snapshots = await collect(
            new Predict('question -> answer: int', lm).stream(
                { question: 'How many?' },
                { repairAttempts: 1 }
            )
        );

        expect(snapshots.at(-1)).toEqual({ answer: 3 });
        expect(lm.calls).toHaveLength(2);
    });

    it('streams with the advice hint when run under Refine-style hints', async () => {
        const { runWithPredictorHints } = await import('./hints');
        const lm = new MockLM({ responses: [chat({ answer: 'ok' })] });
        const predict = new Predict('question -> answer', lm);

        await runWithPredictorHints(new Map([[predict, 'Be brief.']]), () =>
            collect(predict.stream({ question: 'Q' }))
        );

        const prompt = lm.calls[0].messages
            .map((m: ChatMessage) => String(m.content))
            .join('\n');
        expect(prompt).toContain('[[ ## hint_ ## ]]\nBe brief.');
    });
});
