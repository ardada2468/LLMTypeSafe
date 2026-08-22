import { Predict } from './predict';
import { ChainOfThought } from './chain-of-thought';
import { BaseLM } from '../core/base-lm';
import { Prediction } from '../core/prediction';
import { Signature, InputField, OutputField } from '../core/signature';
import { ValidationError } from '../core/errors';
import type {
    ChatMessage,
    LLMCallOptions,
    ModelCapabilities,
    StreamChunk,
} from '../types/language-model';

class QA extends Signature {
    static description = 'Answer a question';

    @InputField({ description: 'the question' })
    question!: string;

    @OutputField({ description: 'the answer' })
    answer!: string;

    @OutputField({ description: 'confidence 0-1', type: 'number' })
    confidence!: number;
}

/**
 * Emits scripted chunks and records how the consumer left the stream, so tests
 * can tell a stream that ran out from one that was closed early.
 */
class StreamingLM extends BaseLM {
    emitted = 0;
    closed = false;
    streamCalls = 0;
    chatCalls = 0;
    lastMessages?: ChatMessage[];
    lastOptions?: LLMCallOptions;

    constructor(
        private readonly chunks: string[],
        private readonly capabilities: Partial<ModelCapabilities> = {}
    ) {
        super('streaming-fake', 'fake-1');
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        this.chatCalls += 1;
        this.lastMessages = messages;
        this.lastOptions = options;
        return this.chunks.join('');
    }

    async *chatStream(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        this.streamCalls += 1;
        this.lastMessages = messages;
        this.lastOptions = options;

        try {
            for (const chunk of this.chunks) {
                this.emitted += 1;
                yield { content: chunk, done: false };
            }
            yield {
                content: '',
                done: true,
                usage: { promptTokens: 5, completionTokens: 7, totalTokens: 12 },
            };
        } finally {
            this.closed = true;
        }
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text', 'json_schema'],
            ...this.capabilities,
        };
    }
}

/** Advertises streaming but never implements `chatStream`, which is optional. */
class NoStreamMethodLM extends BaseLM {
    chatCalls = 0;

    constructor(private readonly reply: string) {
        super('no-stream-fake', 'fake-1');
    }

    async chat(): Promise<string> {
        this.chatCalls += 1;
        return this.reply;
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

/** Sends one chunk and then never sends another, like a stalled connection. */
class StallingLM extends BaseLM {
    constructor(private readonly first: string) {
        super('stalling-fake', 'fake-1');
    }

    async chat(): Promise<string> {
        return this.first;
    }

    async *chatStream(): AsyncGenerator<StreamChunk, void, unknown> {
        yield { content: this.first, done: false };
        await new Promise(() => {});
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text', 'json_schema'],
        };
    }
}

/** JSON split at deliberately awkward points: mid-key, mid-string, after a comma. */
const JSON_CHUNKS = ['{"ans', 'wer": "Pa', 'ris", ', '"confidence": ', '"0.95"', '}'];

async function collect<T>(stream: AsyncGenerator<T, unknown, void>): Promise<T[]> {
    const seen: T[] = [];
    for await (const value of stream) {
        seen.push(value);
    }
    return seen;
}

describe('Predict.stream', () => {
    describe('structured path', () => {
        it('fills fields in as the JSON arrives', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(snapshots.map((snapshot) => snapshot.answer)).toEqual([
                'Pa',
                'Paris',
                'Paris',
                'Paris',
            ]);
        });

        it('validates the final snapshot against the signature', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            // The model sent confidence as a string; only the final, validated
            // snapshot has it coerced to the declared number type.
            expect(snapshots.at(-2)?.confidence).toBe('0.95');
            expect(snapshots.at(-1)).toEqual({ answer: 'Paris', confidence: 0.95 });
        });

        it('returns the validated Prediction as the generator return value', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);

            const stream = new Predict(QA, lm).stream({ question: 'Q' });
            let step = await stream.next();
            while (!step.done) {
                step = await stream.next();
            }

            expect(step.value).toBeInstanceOf(Prediction);
            expect(step.value.answer).toBe('Paris');
            expect(step.value.toObject()).toEqual({ answer: 'Paris', confidence: 0.95 });
        });

        it('asks for the signature schema in the prompt, since chat streams are unconstrained', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);

            await collect(new Predict(QA, lm).stream({ question: 'Capital of France?' }));

            const prompt = lm.lastMessages?.[0].content ?? '';
            expect(prompt).toContain('question: Capital of France?');
            expect(prompt).toContain('Respond with JSON matching this schema');
            expect(prompt).toContain('"confidence"');
        });

        it('throws ValidationError when the completed stream does not match the signature', async () => {
            const lm = new StreamingLM(['{"answer": "Paris", "confidence": "very high"}']);

            await expect(
                collect(new Predict(QA, lm).stream({ question: 'Q' }))
            ).rejects.toThrow(ValidationError);
        });

        it('repeats no snapshot when a chunk adds nothing parseable', async () => {
            const lm = new StreamingLM(['{"answer": "Paris"', ' ', ' ', ', "confidence": 1}']);

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(snapshots).toEqual([
                { answer: 'Paris' },
                { answer: 'Paris', confidence: 1 },
            ]);
        });

        it('repeats no snapshot when the model orders fields its own way', async () => {
            const lm = new StreamingLM(['{"confidence": 0.5, "answer": "Paris"}']);

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(snapshots).toHaveLength(1);
            expect(snapshots[0]).toEqual({ answer: 'Paris', confidence: 0.5 });
        });

        it('reads JSON that follows a preamble containing a bracket', async () => {
            const lm = new StreamingLM([
                'Sure [1] here',
                ':\n{"answer": "Paris", ',
                '"confidence": 1}',
            ]);

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(snapshots.at(-1)).toEqual({ answer: 'Paris', confidence: 1 });
        });
    });

    describe('labelled-text path', () => {
        const capabilities = { supportsStructuredOutput: false };

        it('fills fields in as the labelled text arrives', async () => {
            const chunks = ['answer: Par', 'is\n', 'confidence: 0.', '95'];
            const lm = new StreamingLM(chunks, capabilities);

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(snapshots[0]).toEqual({ answer: 'Par' });
            expect(snapshots.at(-1)).toEqual({ answer: 'Paris', confidence: 0.95 });
        });

        it('sends the plain prompt, with no JSON schema attached', async () => {
            const lm = new StreamingLM(['answer: Paris\nconfidence: 1'], capabilities);

            await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(lm.lastMessages?.[0].content).not.toContain('Respond with JSON');
        });

        it('throws ValidationError when a required field never arrives', async () => {
            const lm = new StreamingLM(['answer: Paris'], capabilities);

            await expect(
                collect(new Predict(QA, lm).stream({ question: 'Q' }))
            ).rejects.toThrow(ValidationError);
        });

        it('shows no snapshot while the buffer is still the field label itself', async () => {
            const lm = new StreamingLM(['answ', 'er: Par', 'is'], capabilities);

            const snapshots = await collect(
                new Predict('question -> answer', lm).stream({ question: 'Q' })
            );

            expect(snapshots).toEqual([{ answer: 'Par' }, { answer: 'Paris' }]);
        });

        it('streams a bare reply that carries no label at all', async () => {
            const lm = new StreamingLM(['Par', 'is'], capabilities);

            const snapshots = await collect(
                new Predict('question -> answer', lm).stream({ question: 'Q' })
            );

            expect(snapshots).toEqual([{ answer: 'Par' }, { answer: 'Paris' }]);
        });
    });

    describe('non-streaming models', () => {
        it('yields once when the model does not support streaming', async () => {
            const lm = new StreamingLM(['answer: Paris\nconfidence: 0.5'], {
                supportsStreaming: false,
                supportsStructuredOutput: false,
            });

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(lm.streamCalls).toBe(0);
            expect(lm.chatCalls).toBe(1);
            expect(snapshots).toEqual([{ answer: 'Paris', confidence: 0.5 }]);
        });

        it('yields once when the model implements no chatStream at all', async () => {
            const lm = new NoStreamMethodLM('answer: Paris\nconfidence: 0.5');

            const snapshots = await collect(new Predict(QA, lm).stream({ question: 'Q' }));

            expect(lm.chatCalls).toBe(1);
            expect(snapshots).toEqual([{ answer: 'Paris', confidence: 0.5 }]);
        });
    });

    describe('cancellation', () => {
        it('closes the provider stream when the consumer breaks out early', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);

            for await (const snapshot of new Predict(QA, lm).stream({ question: 'Q' })) {
                expect(snapshot.answer).toBe('Pa');
                break;
            }

            expect(lm.closed).toBe(true);
            expect(lm.emitted).toBeLessThan(JSON_CHUNKS.length);
        });

        it('rejects and closes the provider stream when the signal aborts mid-stream', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);
            const controller = new AbortController();

            const consume = async () => {
                for await (const _snapshot of new Predict(QA, lm).stream(
                    { question: 'Q' },
                    { signal: controller.signal }
                )) {
                    controller.abort();
                }
            };

            await expect(consume()).rejects.toThrow(/abort/i);
            expect(lm.closed).toBe(true);
            expect(lm.emitted).toBeLessThan(JSON_CHUNKS.length);
        });

        it('rejects while a read is still waiting on a token that never comes', async () => {
            const lm = new StallingLM('{"answer": "Pa');
            const controller = new AbortController();

            const consume = async () => {
                for await (const _snapshot of new Predict(QA, lm).stream(
                    { question: 'Q' },
                    { signal: controller.signal }
                )) {
                    controller.abort();
                }
            };

            // Without racing the read against the signal, this would hang until
            // the test timed out rather than rejecting.
            await expect(consume()).rejects.toThrow(/abort/i);
        });

        it('rejects without calling the model when the signal is already aborted', async () => {
            const lm = new StreamingLM(JSON_CHUNKS);

            await expect(
                collect(
                    new Predict(QA, lm).stream(
                        { question: 'Q' },
                        { signal: AbortSignal.abort() }
                    )
                )
            ).rejects.toThrow(/abort/i);
            expect(lm.streamCalls).toBe(0);
        });
    });

    it('passes call options through to the provider stream', async () => {
        const lm = new StreamingLM(JSON_CHUNKS);

        await collect(
            new Predict(QA, lm).stream({ question: 'Q' }, { temperature: 0.2, maxTokens: 128 })
        );

        expect(lm.lastOptions).toMatchObject({ temperature: 0.2, maxTokens: 128 });
    });
});

describe('ChainOfThought.stream', () => {
    it('reasons first, then streams the answer with the reasoning attached', async () => {
        const lm = new StreamingLM(JSON_CHUNKS);

        const snapshots = await collect(
            new ChainOfThought(QA, lm).stream({ question: 'Capital of France?' })
        );

        // One non-streaming call for the reasoning step, one stream for the answer.
        expect(lm.chatCalls).toBe(1);
        expect(lm.streamCalls).toBe(1);
        expect(snapshots.every((snapshot) => snapshot.reasoning === JSON_CHUNKS.join(''))).toBe(
            true
        );
        expect(snapshots.at(-1)?.answer).toBe('Paris');
        expect(snapshots.at(-1)?.confidence).toBe(0.95);
    });

    it('returns a Prediction carrying both the reasoning and the validated fields', async () => {
        const lm = new StreamingLM(JSON_CHUNKS);

        const stream = new ChainOfThought(QA, lm).stream({ question: 'Q' });
        let step = await stream.next();
        while (!step.done) {
            step = await stream.next();
        }

        expect(step.value).toBeInstanceOf(Prediction);
        expect(step.value.confidence).toBe(0.95);
        expect(step.value.reasoning).toBe(JSON_CHUNKS.join(''));
    });

    it('closes the provider stream when the consumer breaks out early', async () => {
        const lm = new StreamingLM(JSON_CHUNKS);

        for await (const _snapshot of new ChainOfThought(QA, lm).stream({ question: 'Q' })) {
            break;
        }

        expect(lm.closed).toBe(true);
        expect(lm.emitted).toBeLessThan(JSON_CHUNKS.length);
    });
});
