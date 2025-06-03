import { ILanguageModel, UsageStats } from './language-model';

export interface ModuleConfig {
    lm?: ILanguageModel;
    signature?: string | any;
    maxRetries?: number;
    temperature?: number;
}

export interface TraceEntry {
    moduleId: string;
    timestamp: Date;
    duration: number;
    input: Record<string, any>;
    output: Record<string, any>;
    rawLMInput: string;
    rawLMOutput: string;
    usage: UsageStats;
}

export interface MetricFunction {
    (prediction: any, example: any, trace?: TraceEntry): boolean | number;
}

export interface OptimizerOptions {
    maxBootstrapExamples?: number;
    maxEvaluationExamples?: number;
    temperature?: number;
    numTrials?: number;
} 