import { existsSync } from 'node:fs';
import { BaseLM } from '../core/base-lm';
import { LMError } from '../core/errors';
import type {
    ChatMessage,
    ILanguageModel,
    LLMCallOptions,
    ModelCapabilities,
} from '../types/language-model';
import {
    hashRequest,
    keyedOptions,
    readCassette,
    snapshot,
    writeCassette,
    type CassetteChatRequest,
    type CassetteEntry,
    type CassetteRequest,
    type CassetteStructuredRequest,
} from './cassette';

/**
 * How a cassette treats a request it has no recording for.
 *
 * - `replay` — never calls a live model; a miss is an error. The mode CI runs in.
 * - `record` — always calls the wrapped model and rewrites the file from scratch.
 * - `auto` — replays what it has, calls the wrapped model for anything else and
 *   appends it. Useful while a test is still being written.
 */
export type CassetteMode = 'replay' | 'record' | 'auto';

export interface CassetteLMOptions {
    /** Path to the cassette JSON file. */
    path: string;
    /** Defaults to `replay`. */
    mode?: CassetteMode;
    /** Live model to record from. Required by `record` and `auto`. */
    lm?: ILanguageModel;
    /** Write the file after each new recording. Defaults to true. */
    autoSave?: boolean;
    /** Override what the cassette reports about the model it stands in for. */
    capabilities?: Partial<ModelCapabilities>;
    /**
     * Model name written into recorded requests. Defaults to the wrapped model's
     * name, then to the name in the cassette.
     */
    model?: string;
}

const MISS = Symbol('cassette-miss');

/**
 * A language model backed by a JSON file of recorded exchanges.
 *
 * Record once against a real provider, commit the cassette, and every later run
 * replays it — no API key, no network, no flake, and a reviewable diff whenever
 * a prompt changes.
 *
 * ```ts
 * const lm = new CassetteLM({ path: 'cassettes/triage.json' });
 * const result = await new Predict(TriageTicket, lm).forward({ ticket });
 * ```
 *
 * Requests are keyed by a hash of the messages, the model name, and the sampling
 * options that change a reply. Identical requests replay in the order they were
 * recorded; once those run out, the last recorded reply repeats.
 */
export class CassetteLM extends BaseLM {
    /** Path to the cassette file. */
    readonly path: string;
    readonly mode: CassetteMode;

    private readonly lm?: ILanguageModel;
    private readonly autoSave: boolean;
    private readonly capabilityOverrides?: Partial<ModelCapabilities>;
    private readonly recorded: CassetteEntry[] = [];
    private readonly byKey = new Map<string, unknown[]>();
    private readonly plays = new Map<string, number>();

    constructor(options: CassetteLMOptions) {
        super('cassette', options.model ?? options.lm?.getModelName() ?? 'cassette');

        this.path = options.path;
        this.mode = options.mode ?? 'replay';
        this.lm = options.lm;
        this.autoSave = options.autoSave ?? true;
        this.capabilityOverrides = options.capabilities;

        if (this.mode !== 'replay' && !this.lm) {
            throw new LMError(
                'cassette',
                `mode '${this.mode}' needs a live \`lm\` to record from`
            );
        }

        // `record` rewrites the file, so it starts from nothing.
        if (this.mode !== 'record') {
            if (existsSync(this.path)) {
                for (const entry of readCassette(this.path)) {
                    this.remember(entry);
                }
            } else if (this.mode === 'replay') {
                throw new LMError(
                    'cassette',
                    `No cassette at ${this.path}. Record one with ` +
                        `new CassetteLM({ path, mode: 'record', lm }).`
                );
            }
        }

        if (!options.model && !options.lm) {
            const recordedModel = this.recorded[0]?.request.model;
            if (recordedModel) {
                this.setModel(recordedModel);
            }
        }
    }

    /** Replay a committed cassette. Never touches the network. */
    static replay(
        path: string,
        options?: Omit<CassetteLMOptions, 'path' | 'mode' | 'lm'>
    ): CassetteLM {
        return new CassetteLM({ ...options, path, mode: 'replay' });
    }

    /** Capture `lm`'s replies into a fresh cassette at `path`. */
    static record(
        path: string,
        lm: ILanguageModel,
        options?: Omit<CassetteLMOptions, 'path' | 'mode' | 'lm'>
    ): CassetteLM {
        return new CassetteLM({ ...options, path, lm, mode: 'record' });
    }

    /** Every exchange currently held, in recorded order. */
    get entries(): readonly CassetteEntry[] {
        return this.recorded;
    }

    /** Write the cassette to disk. Called for you unless `autoSave` is false. */
    save(): void {
        writeCassette(this.path, this.recorded);
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const request: CassetteChatRequest = {
            kind: 'chat',
            model: options?.model ?? this.getModelName(),
            messages: snapshot(messages),
            options: keyedOptions(options),
        };

        return this.play(request, () => this.requireLM().chat(messages, options));
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const request: CassetteStructuredRequest = {
            kind: 'structured',
            model: options?.model ?? this.getModelName(),
            prompt,
            schema: snapshot(schema),
            options: keyedOptions(options),
        };

        return this.play<T>(request, () =>
            this.requireLM().generateStructured<T>(prompt, schema, options)
        );
    }

    /**
     * What the recorded model could do. Taken from the wrapped model while
     * recording; while replaying, only `supportsStructuredOutput` can be inferred
     * from the cassette — enough for a module to take the same path both times.
     * A module that branches on `maxContextLength` or `supportsFunctionCalling`
     * should pass `capabilities` so replay matches what it recorded against.
     * Cassettes never stream.
     */
    getCapabilities(): ModelCapabilities {
        const base = this.lm?.getCapabilities() ?? this.inferCapabilities();
        return { ...base, supportsStreaming: false, ...this.capabilityOverrides };
    }

    private inferCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: false,
            supportsStructuredOutput: this.recorded.some(
                (entry) => entry.request.kind === 'structured'
            ),
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
        };
    }

    private async play<T>(request: CassetteRequest, live: () => Promise<T>): Promise<T> {
        const key = hashRequest(request);

        if (this.mode !== 'record') {
            const played = this.take(key);
            if (played !== MISS) {
                this.recordUsage({ latencyMs: 0 });
                // A copy, so a caller that mutates one reply cannot change what
                // the next replay of the same request returns.
                return snapshot(played) as T;
            }
            if (this.mode === 'replay') {
                throw this.missError(request, key);
            }
        }

        const startedAt = Date.now();
        let response: T;
        try {
            response = await live();
        } catch (error) {
            this.recordError();
            throw error;
        }

        this.remember({ key, request, response: snapshot(response) });
        this.plays.set(key, (this.plays.get(key) ?? 0) + 1);
        this.recordUsage({ latencyMs: Date.now() - startedAt });

        if (this.autoSave) {
            this.save();
        }

        return response;
    }

    /** Take the next recorded reply for `key`, repeating the last once spent. */
    private take(key: string): unknown {
        const responses = this.byKey.get(key);
        if (!responses || responses.length === 0) {
            return MISS;
        }

        const played = this.plays.get(key) ?? 0;
        this.plays.set(key, played + 1);
        return responses[Math.min(played, responses.length - 1)];
    }

    private remember(entry: CassetteEntry): void {
        this.recorded.push(entry);
        const existing = this.byKey.get(entry.key);
        if (existing) {
            existing.push(entry.response);
        } else {
            this.byKey.set(entry.key, [entry.response]);
        }
    }

    private requireLM(): ILanguageModel {
        if (!this.lm) {
            throw new LMError('cassette', 'No live `lm` to record from');
        }
        return this.lm;
    }

    private missError(request: CassetteRequest, key: string): LMError {
        const text =
            request.kind === 'chat' ? (request.messages.at(-1)?.content ?? '') : request.prompt;
        const excerpt = text.length > 160 ? `${text.slice(0, 160)}…` : text;

        return new LMError(
            'cassette',
            `No recorded ${request.kind} response (key ${key}) in ${this.path}, ` +
                `across ${this.recorded.length} entr${this.recorded.length === 1 ? 'y' : 'ies'}. ` +
                `Re-record with mode 'record', or use mode 'auto' to append misses.\n` +
                `Request began: ${excerpt}`
        );
    }
}
