import type { UsageStats } from './language-model';

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
