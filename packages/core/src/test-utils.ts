/**
 * Internal alias for the published test doubles.
 *
 * The doubles now live in `./testing` and ship as `@ts-dspy/core/testing`; this
 * file stays so core's own tests can keep importing `../test-utils`.
 */

export { MockLM } from './testing/mock-lm';
export type { MockLMOptions } from './testing/mock-lm';
