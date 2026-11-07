/**
 * The adapter layer now lives in `src/adapters`. This re-export keeps imports
 * of `core/adapter` working.
 */
export {
    Adapter,
    ChatAdapter,
    JSONAdapter,
    XMLAdapter,
    TwoStepAdapter,
    defaultAdapter,
    resolveAdapter,
} from '../adapters';
export type {
    AdapterCallInput,
    AdapterFormatInput,
    ChatAdapterOptions,
    TwoStepAdapterOptions,
} from '../adapters';
