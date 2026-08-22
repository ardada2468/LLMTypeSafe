// Core types
export * from './types';

// Core classes
export { Signature, InputField, OutputField } from './core/signature';
export { Module } from './core/module';
export { BaseLM } from './core/base-lm';
export { Prediction } from './core/prediction';
export { Example } from './core/example';
export { configure, getDefaultLM, isCacheEnabled, isTracingEnabled } from './core/config';

// Errors
export { TsDspyError, ValidationError, LMError } from './core/errors';
export type { FieldValidationIssue } from './core/errors';

// Modules
export { Predict } from './modules/predict';
export type { PredictOptions } from './modules/predict';
export { ChainOfThought } from './modules/chain-of-thought';
export { RespAct } from './modules/respact';
export type { ToolFunction, ToolWithDescription, ToolDefinition } from './modules/respact';

// Optimizers
export { LabeledFewShot, BootstrapFewShot } from './optimizers';
export type {
    Metric,
    MetricResult,
    DemoModule,
    LabeledFewShotOptions,
    LabeledFewShotConfig,
    BootstrapFewShotOptions,
    BootstrapFewShotConfig,
    BootstrapProgressEvent,
} from './optimizers';

// Utilities
export { buildPrompt, parseOutput, renderDemos } from './utils/parsing';
export { fieldConfigToZod, buildOutputSchema, buildOutputJsonSchema } from './utils/schema';
