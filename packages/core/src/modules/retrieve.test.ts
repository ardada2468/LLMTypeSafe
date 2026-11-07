import { Retrieve } from './retrieve';
import { ChainOfThought } from './chain-of-thought';
import { InMemoryVectorStore, type Retriever } from '../core/retrieve';
import { type Embedder } from '../core/embedder';
import { Module } from '../core/module';
import { configure, context } from '../core/config';
import { TsDspyError } from '../core/errors';
import { MockLM } from '../test-utils';
import { contentToText } from '../utils/content';
import type { Callback, ModuleStartEvent } from '../core/callbacks';

/**
 * A deterministic bag-of-words embedder over a fixed vocabulary: each
 * dimension counts one keyword, so similarity is keyword overlap.
 */
class KeywordEmbedder implements Embedder {
    calls: string[][] = [];
    readonly vocabulary = ['paris', 'france', 'berlin', 'germany', 'capital', 'river'];

    get dimensions(): number {
        return this.vocabulary.length;
    }

    async embed(texts: string[]): Promise<number[][]> {
        this.calls.push(texts);
        return texts.map((text) => {
            const words = text.toLowerCase().match(/[a-z]+/g) ?? [];
            return this.vocabulary.map((term) => words.filter((word) => word === term).length);
        });
    }
}

const DOCS = [
    { text: 'Berlin is the capital of Germany.', metadata: { id: 'de' } },
    { text: 'Paris is the capital of France.', metadata: { id: 'fr' } },
    'The Seine is a river in France.',
];

describe('InMemoryVectorStore', () => {
    it('returns the k most similar documents, best first, with scores and metadata', async () => {
        const store = new InMemoryVectorStore(new KeywordEmbedder());
        await store.addAll(DOCS);

        const passages = await store.retrieve('capital of Germany', 2);

        expect(passages.map((passage) => passage.text)).toEqual([
            'Berlin is the capital of Germany.',
            'Paris is the capital of France.',
        ]);
        expect(passages[0]!.metadata).toEqual({ id: 'de' });
        expect(passages[0]!.score).toBeGreaterThan(passages[1]!.score!);
    });

    it('embeds documents in one batch and indexes them one at a time too', async () => {
        const embedder = new KeywordEmbedder();
        const store = new InMemoryVectorStore(embedder);
        await store.addAll(DOCS);
        await store.add('Paris has a river.');

        expect(embedder.calls[0]).toHaveLength(3);
        expect(store.size).toBe(4);
        store.clear();
        expect(await store.retrieve('Paris', 3)).toEqual([]);
    });

    it('rejects an embedder that returns the wrong number of vectors', async () => {
        const store = new InMemoryVectorStore({ embed: async () => [[1]] });
        await expect(store.addAll(['a', 'b'])).rejects.toThrow(RangeError);
    });

    it('returns nothing for k = 0 and rejects a negative k', async () => {
        const store = new InMemoryVectorStore(new KeywordEmbedder());
        await store.addAll(DOCS);
        expect(await store.retrieve('Paris', 0)).toEqual([]);
        await expect(store.retrieve('Paris', -1)).rejects.toThrow(RangeError);
    });
});

describe('Retrieve', () => {
    it('returns passages from its own retriever', async () => {
        const store = new InMemoryVectorStore(new KeywordEmbedder());
        await store.addAll(DOCS);

        const result = await new Retrieve({ k: 1, retriever: store }).forward({
            query: 'Germany',
        });

        expect(result.passages).toHaveLength(1);
        expect(result.passages[0]!.text).toContain('Berlin');
    });

    it('uses the configured retriever, scope first, and k from the inputs', async () => {
        const make = (label: string): Retriever => ({
            retrieve: async (query, k) =>
                Array.from({ length: k }, (_, index) => ({
                    text: `${label}:${query}:${index}`,
                })),
        });
        const retrieve = new Retrieve();

        configure({ retriever: make('global') });
        try {
            expect((await retrieve.forward({ query: 'q' })).passages).toHaveLength(3);
            const scoped = await context({ retriever: make('scoped') }, () =>
                retrieve.forward({ query: 'q', k: 1 })
            );
            expect(scoped.passages).toEqual([{ text: 'scoped:q:0' }]);
        } finally {
            configure({ retriever: undefined });
        }
    });

    it('fails clearly without a retriever', async () => {
        await expect(new Retrieve().forward({ query: 'q' })).rejects.toThrow(TsDspyError);
    });
});

describe('RAG: Retrieve composed with ChainOfThought', () => {
    class RAG extends Module {
        retrieve: Retrieve;
        answer = new ChainOfThought('context, question -> answer');

        constructor(retriever: Retriever) {
            super();
            this.retrieve = new Retrieve({ k: 1, retriever });
        }

        async forward(inputs: { question: string }) {
            const { passages } = await this.retrieve.forward({ query: inputs.question });
            return this.answer.forward({
                context: passages.map((passage) => passage.text).join('\n'),
                question: inputs.question,
            });
        }
    }

    it('grounds the answer in the retrieved passages', async () => {
        const store = new InMemoryVectorStore(new KeywordEmbedder());
        await store.addAll(DOCS);
        const lm = new MockLM({
            responses: [
                '[[ ## reasoning ## ]]\nThe context says Paris.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]',
            ],
        });
        const started: string[] = [];
        const recorder: Callback = {
            onModuleStart: (event: ModuleStartEvent) => started.push(event.moduleId),
        };

        const rag = new RAG(store);
        const result = await context({ lm, callbacks: [recorder] }, () =>
            rag.forward({ question: 'What is the capital of France?' })
        );

        expect(result.answer).toBe('Paris');
        const prompt = lm.calls[0].messages.map((m) => contentToText(m.content)).join('\n');
        expect(prompt).toContain('Paris is the capital of France.');
        expect(prompt).not.toContain('Seine');
        expect(started.map((id) => id.split('#')[0])).toEqual([
            'RAG',
            'Retrieve',
            'ChainOfThought',
        ]);
        expect(rag.namedPredictors().map(([path]) => path)).toEqual(['answer']);
    });
});
