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
    HISTORY_FIELD_TYPE,
    isHistoryFieldType,
    image,
    history,
    extendSignature,
    toSignatureClass,
    getSignatureInstructions,
} from './core/signature';
export type {
    AnyZodSignature,
    SignatureLike,
    SignatureSource,
    ZodSignature,
    ZodSignatureDefinition,
    FieldSpec,
    FieldSpecs,
    ExtendSignatureOptions,
} from './core/signature';
export { History } from './core/history';
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
    getRetriever,
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

export {
    circuitBreaker,
    concurrencyLimit,
    budget,
    CircuitOpenError,
    BudgetExceededError,
} from './core/resilience';
export { buildCacheKey } from './core/cache';
export type { CacheKeyInput } from './core/cache';
export type {
    CircuitState,
    CircuitBreakerOptions,
    ConcurrencyLimitOptions,
    BudgetOptions,
    BudgetLimit,
} from './core/resilience';

// Contracts
export { Tool } from './core/tool';
export type { ToolOptions, ToolInit, ToolParameters } from './core/tool';
export { cosineSimilarity } from './core/embedder';
export type { Embedder } from './core/embedder';

// Retrieval
export { InMemoryVectorStore } from './core/retrieve';
export type { Passage, Retriever, DocumentInput } from './core/retrieve';

// Program streaming
export { streamify, defaultStatusMessageProvider } from './core/streaming';
export type {
    StreamEvent,
    StatusStreamEvent,
    FieldStreamEvent,
    PredictionStreamEvent,
    StatusMessageProvider,
    StreamifyOptions,
    StreamifiedProgram,
} from './core/streaming';

// Adapters
export {
    Adapter,
    ChatAdapter,
    JSONAdapter,
    XMLAdapter,
    TwoStepAdapter,
    defaultAdapter,
    resolveAdapter,
    signatureLayout,
} from './adapters';
export type {
    AdapterCallInput,
    AdapterField,
    AdapterFormatInput,
    ChatAdapterOptions,
    SignatureLayout,
    TwoStepAdapterOptions,
} from './adapters';
export { providerOptionsFor, mergeRequestBody } from './core/provider-options';

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
    retryAfterMsFrom,
} from './core/errors';
export type { FieldValidationIssue, ErrorDiscriminators, LMErrorClass } from './core/errors';

// Validation self-repair
export {
    buildRepairPrompt,
    buildRepairInstruction,
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
export { ChainOfThought, REASONING_DESCRIPTION } from './modules/chain-of-thought';
export type { WithReasoning } from './modules/chain-of-thought';
export { ReAct, formatTrajectory } from './modules/react';
export type {
    ToolFunction,
    ToolWithDescription,
    ToolDefinition,
    ToolParameterSchema,
    ReActEvent,
    ReActOptions,
    ReActExtras,
    ReActOutput,
} from './modules/react';
export { RespAct } from './modules/respact';
export type { RespActEvent, RespActOptions } from './modules/respact';
export { BestOfN } from './modules/best-of-n';
export type { BestOfNOptions, RewardFn } from './modules/best-of-n';
export { Refine, OfferFeedback } from './modules/refine';
export type { RefineOptions } from './modules/refine';
export { Parallel, ParallelError } from './modules/parallel';
export type {
    ExecPair,
    ParallelOptions,
    ParallelOutcome,
    FailedExec,
} from './modules/parallel';
export { MultiChainComparison } from './modules/multi-chain-comparison';
export type { MultiChainComparisonOptions, Completion } from './modules/multi-chain-comparison';
export { ProgramOfThought, runSandboxed, extractCode } from './modules/program-of-thought';
export type { RunSandboxedOptions } from './modules/program-of-thought';
export type { ProgramOfThoughtOptions, SandboxResult } from './modules/program-of-thought';
export { Retrieve } from './modules/retrieve';
export type { RetrieveOptions, RetrieveOutput } from './modules/retrieve';

// Optimizers
export {
    LabeledFewShot,
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    KNNFewShot,
    KNNProgram,
    defaultVectorizeInputs,
    Ensemble,
    EnsembleProgram,
    majority,
    COPRO,
    TPESampler,
    MIPROv2,
    MIPRO_AUTO_PRESETS,
    MIPRO_TIPS,
    SIMBA,
    GEPA,
    GEPA_AUTO_CANDIDATES,
    evaluateProgram,
    runWithTrace,
    bootstrapTraces,
    proposeInstructions,
    summarizeDataset,
    describeProgram,
    sampleMinibatch,
    createRng,
    shuffled,
} from './optimizers';
// `Metric` and `MetricResult` are exported once, from './evaluate' below.
export type {
    DemoModule,
    LabeledFewShotOptions,
    LabeledFewShotConfig,
    BootstrapFewShotOptions,
    BootstrapFewShotConfig,
    BootstrapProgressEvent,
    BootstrapFewShotWithRandomSearchOptions,
    RandomSearchCandidate,
    RandomSearchResults,
    KNNFewShotOptions,
    KNNNeighbour,
    EnsembleOptions,
    MajorityOptions,
    ReduceFn,
    COPROOptions,
    COPROAttempt,
    COPROResults,
    TPEOptions,
    TPESpace,
    TPETrial,
    MIPROv2Options,
    MIPROTrial,
    MIPROResults,
    SIMBAOptions,
    SIMBAResults,
    SIMBAStep,
    SIMBAStrategy,
    GEPAOptions,
    GEPACandidate,
    GEPADetailedResults,
    GEPAIterationLog,
    BaseOptimizerOptions,
    CompileConfig,
    OptimizerProgressEvent,
    ProgramEvaluation,
    TracedRun,
    BootstrapTracesOptions,
    BootstrapTracesResult,
    ProposeInstructionsOptions,
    Rng,
} from './optimizers';

// Evaluation
export * from './evaluate';

// Utilities
export { parseOutput } from './utils/parsing';
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
export {
    fieldConfigToZod,
    buildOutputSchema,
    buildOutputJsonSchema,
    getInputFieldConfigs,
    getOutputFieldConfigs,
    validateOutputs,
} from './utils/schema';
export { parsePartialJson } from './utils/partial-json';
export type { PartialJsonOptions } from './utils/partial-json';
export { mapWithConcurrency, DEFAULT_CONCURRENCY } from './utils/pool';
export type { MapWithConcurrencyOptions, SettledResult } from './utils/pool';
