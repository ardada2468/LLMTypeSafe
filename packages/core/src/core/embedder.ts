/**
 * Turns text into vectors. Retrievers, semantic metrics and example selectors
 * depend on this interface rather than on any one provider's SDK, so a
 * provider package — or a local model — supplies the implementation.
 */
export interface Embedder {
    /**
     * Embed each text, returning one vector per input in the same order.
     * Implementations should batch internally as their provider allows.
     */
    embed(texts: string[], options?: { signal?: AbortSignal }): Promise<number[][]>;
    /** Vector length, when known up front. */
    readonly dimensions?: number;
}

/**
 * Cosine similarity of two vectors, in `[-1, 1]`. Returns `0` when either
 * vector has zero magnitude, rather than `NaN`, so a degenerate embedding
 * ranks last instead of poisoning a sort.
 *
 * @throws RangeError when the vectors differ in length — comparing embeddings
 *   from two different models is a bug worth hearing about.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
    if (a.length !== b.length) {
        throw new RangeError(
            `Cannot compare vectors of different lengths (${a.length} and ${b.length}).`
        );
    }

    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
