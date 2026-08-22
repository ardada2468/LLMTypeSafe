// Core types
export * from './types';

// Core classes
export {
    Signature,
    InputField,
    OutputField,
    signature,
    isZodSignature,
} from './core/signature';
export type {
    AnyZodSignature,
    SignatureLike,
    SignatureSource,
    ZodSignature,
    ZodSignatureDefinition,
} from './core/signature';
export { Module } from './core/module';
export type { BatchOptions, BatchResult } from './core/module';
export { BaseLM } from './core/base-lm';
export { Prediction } from './core/prediction';
export { Example } from './core/example';
export {
    configure,
    getDefaultLM,
    getCache,
    clearCache,
    isCacheEnabled,
    isTracingEnabled,
} from './core/config';
export type { ConfigureOptions, TraceHandler } from './core/config';

// Caching
export { MemoryCache } from './core/cache';
export type { Cache, MemoryCacheOptions, MaybePromise } from './core/cache';

// Tracing
export { inspectHistory, clearHistory } from './core/trace';
export type { TraceSpan } from './core/trace';

// Errors
export {
    TsDspyError,
    ValidationError,
    LMError,
    RateLimitError,
    AuthError,
    ContextLengthError,
    ContentFilterError,
    TimeoutError,
    classify,
} from './core/errors';
export type { FieldValidationIssue, ErrorDiscriminators, LMErrorClass } from './core/errors';

// Validation self-repair
export {
    buildRepairPrompt,
    buildRepairObservation,
    describeValidationIssues,
    isRepeatedFailure,
    listFailingFields,
    MAX_REPAIR_ATTEMPTS,
} from './core/repair';
export type { RepairFormat } from './core/repair';

// Modules
export { Predict } from './modules/predict';
export type { StreamOptions, PredictionStream, PartialOutput } from './modules/predict';
export { ChainOfThought } from './modules/chain-of-thought';
export { RespAct } from './modules/respact';
export type { ToolFunction, ToolWithDescription, ToolDefinition } from './modules/respact';

// Evaluation
export * from './evaluate';

// Utilities
export { buildPrompt, parseOutput } from './utils/parsing';
export { fieldConfigToZod, buildOutputSchema, buildOutputJsonSchema } from './utils/schema';
export { parsePartialJson } from './utils/partial-json';
export type { PartialJsonOptions } from './utils/partial-json';
export { mapWithConcurrency, DEFAULT_CONCURRENCY } from './utils/pool';
export type { MapWithConcurrencyOptions, SettledResult } from './utils/pool';
