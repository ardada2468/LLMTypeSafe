/**
 * Test doubles for ts-dspy programs, published as `@ts-dspy/core/testing`.
 *
 * `MockLM` scripts replies inline and records every call; `CassetteLM` replays a
 * JSON file of exchanges captured once against a real provider. Neither reaches
 * the network, so a suite needs no API key.
 *
 * Kept out of the main entry point deliberately: `CassetteLM` reads and writes
 * files, and nothing that ships to production should pull that in.
 */

export { MockLM } from './mock-lm';
export type { MockLMOptions } from './mock-lm';

export { CassetteLM } from './cassette-lm';
export type { CassetteLMOptions, CassetteMode } from './cassette-lm';

export { hashRequest, readCassette, writeCassette } from './cassette';
export type {
    CassetteChatRequest,
    CassetteEntry,
    CassetteRequest,
    CassetteStructuredRequest,
} from './cassette';
