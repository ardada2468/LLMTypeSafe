import { cosineSimilarity, type Embedder } from './embedder';

/** One retrieved document. */
export interface Passage {
    text: string;
    /** Relevance to the query; higher is more relevant. Scale is the retriever's own. */
    score?: number;
    /** Anything the caller attached when indexing: an id, a source URL, a title. */
    metadata?: Record<string, unknown>;
}

/**
 * Finds the passages most relevant to a query. The one interface `Retrieve`
 * depends on, so a vector database, a search API or BM25 over local files can
 * each stand behind it.
 */
export interface Retriever {
    /** The `k` most relevant passages, most relevant first. */
    retrieve(query: string, k: number, options?: { signal?: AbortSignal }): Promise<Passage[]>;
}

/** A document to index: bare text, or text with metadata. */
export type DocumentInput = string | { text: string; metadata?: Record<string, unknown> };

interface IndexedDocument {
    text: string;
    metadata?: Record<string, unknown>;
    vector: number[];
}

/**
 * A brute-force vector index held in memory: every query is compared with
 * every document by cosine similarity.
 *
 * Exact, dependency-free, and fast enough for a few tens of thousands of
 * documents — prototypes, tests, a small knowledge base. Reach for a real
 * vector store beyond that; it can implement {@link Retriever} just the same.
 *
 * ```ts
 * const store = new InMemoryVectorStore(embedder);
 * await store.addAll(['Paris is the capital of France.', 'Berlin is in Germany.']);
 * const [top] = await store.retrieve('capital of France', 1);
 * ```
 */
export class InMemoryVectorStore implements Retriever {
    private readonly documents: IndexedDocument[] = [];

    constructor(private readonly embedder: Embedder) {}

    /** Number of documents indexed. */
    get size(): number {
        return this.documents.length;
    }

    /** Embed and index one document. */
    async add(document: DocumentInput, options?: { signal?: AbortSignal }): Promise<void> {
        await this.addAll([document], options);
    }

    /**
     * Embed and index many documents, in one `embed` call so the embedder can
     * batch. Nothing is indexed if embedding fails.
     *
     * @throws RangeError when the embedder returns the wrong number of vectors.
     */
    async addAll(
        documents: readonly DocumentInput[],
        options?: { signal?: AbortSignal }
    ): Promise<void> {
        if (documents.length === 0) return;
        const normalised = documents.map((document) =>
            typeof document === 'string' ? { text: document } : document
        );
        const vectors = await this.embedder.embed(
            normalised.map((document) => document.text),
            options
        );
        if (vectors.length !== normalised.length) {
            throw new RangeError(
                `Embedder returned ${vectors.length} vectors for ${normalised.length} documents.`
            );
        }
        normalised.forEach((document, index) => {
            this.documents.push({
                text: document.text,
                ...(document.metadata ? { metadata: document.metadata } : {}),
                vector: vectors[index]!,
            });
        });
    }

    /**
     * The `k` documents most similar to `query`, highest cosine similarity
     * first, each with its similarity as `score`. Ties keep insertion order.
     */
    async retrieve(
        query: string,
        k: number,
        options?: { signal?: AbortSignal }
    ): Promise<Passage[]> {
        if (!Number.isInteger(k) || k < 0) {
            throw new RangeError(`k must be a non-negative integer, received ${String(k)}`);
        }
        if (k === 0 || this.documents.length === 0) return [];

        const [queryVector] = await this.embedder.embed([query], options);
        if (!queryVector) throw new RangeError('Embedder returned no vector for the query.');

        return this.documents
            .map((document, index) => ({
                document,
                index,
                score: cosineSimilarity(queryVector, document.vector),
            }))
            .sort((a, b) => b.score - a.score || a.index - b.index)
            .slice(0, k)
            .map(({ document, score }) => ({
                text: document.text,
                score,
                ...(document.metadata ? { metadata: { ...document.metadata } } : {}),
            }));
    }

    /** Remove every document. */
    clear(): void {
        this.documents.length = 0;
    }
}
