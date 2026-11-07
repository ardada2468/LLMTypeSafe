// Core types
export * from './types';

// Core classes
export {
    Signature,
    InputField,
    OutputField,
    signature,
    isZodSignature,
    ImageField,
    IMAGE_FIELD_TYPE,
    isImageFieldType,
} from './core/signature';
export type {
    AnyZodSignature,
    SignatureLike,
    SignatureSource,
    ZodSignature,
    ZodSignatureDefinition,
} from './core/signature';
export { Module } from './core/module';
export type {
    BatchOptions,
    BatchResult,
    PredictorState,
    ProgramState,
    ProgramStateMetadata,
    SerializedDemo,
} from './core/module';
export { BaseLM } from './core/base-lm';
export { Prediction } from './core/prediction';
export { Example } from './core/example';
export {
    configure,
    settings,
    context,
    withSettings,
    getDefaultLM,
    getCache,
    clearCache,
    isCacheEnabled,
    isTracingEnabled,
    getCallbacks,
    isUsageTrackingEnabled,
} from './core/config';
export type {
    ConfigureOptions,
    SettingsOverrides,
    TraceHandler,
    ModelPricing,
    PricingTable,
} from './core/config';
export { VERSION } from './core/version';

// Callbacks
export type {
    Callback,
    CallbackEventBase,
    CallbackEndBase,
    ModuleLike,
    ModuleStartEvent,
    ModuleEndEvent,
    LMMethod,
    LMStartEvent,
    LMEndEvent,
    ToolStartEvent,
    ToolEndEvent,
    AdapterFormatEvent,
    AdapterParseEvent,
} from './core/callbacks';

// Usage
export { recordLMUsage, sumUsage } from './core/usage';
export type { LMUsage, UsageDelta } from './core/usage';

// LM middleware
export {
    wrapLM,
    retry,
    fallback,
    defaults,
    rateLimit,
    retryWithBackoff,
    isRetryableLMError,
    dispatchLM,
} from './core/middleware';
export type {
    LMRequest,
    LMNext,
    LMMiddleware,
    BackoffOptions,
    RetryOptions,
    FallbackOptions,
    RateLimitOptions,
} from './core/middleware';

// Contracts
export { Tool } from './core/tool';
export type { ToolOptions, ToolInit, ToolParameters } from './core/tool';
export { cosineSimilarity } from './core/embedder';
export type { Embedder } from './core/embedder';
export type { Adapter } from './core/adapter';

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
    MaxStepsError,
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
export type { PredictOptions } from './modules/predict';
export type { StreamOptions, PredictionStream, PartialOutput } from './modules/predict';
export { ChainOfThought } from './modules/chain-of-thought';
export { RespAct } from './modules/respact';
export type {
    ToolFunction,
    ToolWithDescription,
    ToolDefinition,
    ToolParameterSchema,
    RespActEvent,
    RespActOptions,
} from './modules/respact';

// Optimizers
export { LabeledFewShot, BootstrapFewShot } from './optimizers';
// `Metric` and `MetricResult` are exported once, from './evaluate' below.
export type {
    DemoModule,
    LabeledFewShotOptions,
    LabeledFewShotConfig,
    BootstrapFewShotOptions,
    BootstrapFewShotConfig,
    BootstrapProgressEvent,
} from './optimizers';

// Evaluation
export * from './evaluate';

// Utilities
export { buildPrompt, buildPromptContent, parseOutput, renderDemos } from './utils/parsing';
export type { PromptOptions, RenderDemosOptions, DemoFormat } from './utils/parsing';
export {
    textPart,
    imagePart,
    isImagePart,
    imageToUrl,
    imageMediaType,
    normalizeImageSource,
    toContentParts,
    hasImageContent,
    contentToText,
} from './utils/content';
export { fieldConfigToZod, buildOutputSchema, buildOutputJsonSchema } from './utils/schema';
export { parsePartialJson } from './utils/partial-json';
export type { PartialJsonOptions } from './utils/partial-json';
export { mapWithConcurrency, DEFAULT_CONCURRENCY } from './utils/pool';
export type { MapWithConcurrencyOptions, SettledResult } from './utils/pool';
