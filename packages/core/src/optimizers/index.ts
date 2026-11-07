export { LabeledFewShot } from './labeled-few-shot';
export type { LabeledFewShotOptions, LabeledFewShotConfig } from './labeled-few-shot';

export { BootstrapFewShot } from './bootstrap-few-shot';
export type {
    BootstrapFewShotOptions,
    BootstrapFewShotConfig,
    BootstrapProgressEvent,
} from './bootstrap-few-shot';

export { BootstrapFewShotWithRandomSearch } from './random-search';
export type {
    BootstrapFewShotWithRandomSearchOptions,
    RandomSearchCandidate,
    RandomSearchResults,
} from './random-search';

export { KNNFewShot, KNNProgram, defaultVectorizeInputs } from './knn-few-shot';
export type { KNNFewShotOptions, KNNNeighbour } from './knn-few-shot';

export { Ensemble, EnsembleProgram, majority } from './ensemble';
export type { EnsembleOptions, MajorityOptions, ReduceFn } from './ensemble';

export { COPRO } from './copro';
export type { COPROOptions, COPROAttempt, COPROResults } from './copro';

export { TPESampler } from './tpe';
export type { TPEOptions, TPESpace, TPETrial } from './tpe';

export { MIPROv2, MIPRO_AUTO_PRESETS, MIPRO_TIPS } from './mipro';
export type { MIPROv2Options, MIPROTrial, MIPROResults } from './mipro';

export { SIMBA } from './simba';
export type { SIMBAOptions, SIMBAResults, SIMBAStep, SIMBAStrategy } from './simba';

export { GEPA, GEPA_AUTO_CANDIDATES } from './gepa';
export type { GEPAOptions, GEPACandidate, GEPADetailedResults, GEPAIterationLog } from './gepa';

export {
    evaluateProgram,
    runWithTrace,
    bootstrapTraces,
    proposeInstructions,
    summarizeDataset,
    describeProgram,
    sampleMinibatch,
} from './utils';
export type {
    BaseOptimizerOptions,
    CompileConfig,
    OptimizerProgressEvent,
    ProgramEvaluation,
    TracedRun,
    BootstrapTracesOptions,
    BootstrapTracesResult,
    ProposeInstructionsOptions,
} from './utils';

export { createRng, shuffled } from './random';
export type { Rng } from './random';

export type { Metric, MetricResult, DemoModule } from './types';
