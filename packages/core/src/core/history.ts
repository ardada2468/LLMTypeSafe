/**
 * Earlier turns of a conversation, for a signature input declared as
 * `history`.
 *
 * Each entry holds one past turn's field values — the inputs that were sent
 * and the outputs that came back — keyed by the signature's own field names:
 *
 * ```ts
 * const history = new History([
 *     { question: 'What is the capital of France?', answer: 'Paris' },
 * ]);
 * await chat.forward({ question: 'And its population?', history });
 * ```
 *
 * An adapter renders a history field as real prior user/assistant message
 * pairs rather than as inline text, so the model sees the conversation the way
 * a chat model is trained to read one — and the field itself never appears in
 * the field listing or the user turn.
 */
export class History {
    readonly messages: Array<Record<string, unknown>>;

    constructor(messages: Array<Record<string, unknown>> = []) {
        // Copied, so a caller appending to their own array afterwards does not
        // rewrite a history a prediction was already made from.
        this.messages = messages.map((message) => ({ ...message }));
    }

    /** A new history with `turn` appended; this one is left untouched. */
    with(turn: Record<string, unknown>): History {
        return new History([...this.messages, turn]);
    }
}

/**
 * Accept a {@link History} or anything shaped like one (`{ messages: [...] }`),
 * which is what a history looks like after a JSON round trip.
 */
export function toHistory(value: unknown): History | undefined {
    if (value instanceof History) return value;
    if (
        typeof value === 'object' &&
        value !== null &&
        Array.isArray((value as { messages?: unknown }).messages)
    ) {
        const messages = (value as { messages: unknown[] }).messages.filter(
            (entry): entry is Record<string, unknown> =>
                typeof entry === 'object' && entry !== null && !Array.isArray(entry)
        );
        return new History(messages);
    }
    return undefined;
}
