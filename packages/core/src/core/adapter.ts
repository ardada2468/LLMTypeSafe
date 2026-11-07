import type { ChatMessage } from '../types/language-model';
import type { SignatureLike } from './signature';
import type { Example } from './example';

/**
 * Turns a signature, demos and inputs into chat messages, and a model's reply
 * back into output fields.
 *
 * @experimental A placeholder for the adapter layer arriving in a later
 * release: the interface may still change, and the built-in modules do not
 * consult it yet. It exists now so `settings` can carry one through scopes.
 */
export interface Adapter {
    /** Short name reported in adapter callback events. */
    readonly name?: string;
    format(
        signature: SignatureLike,
        demos: Example[],
        inputs: Record<string, unknown>,
        options?: Record<string, unknown>
    ): ChatMessage[];
    parse(
        signature: SignatureLike,
        completion: string,
        options?: Record<string, unknown>
    ): Record<string, unknown>;
}
