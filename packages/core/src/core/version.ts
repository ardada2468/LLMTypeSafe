/**
 * Replaced with the package version by tsup's `define` at build time. Read
 * from source (tests, `tsx`), it is not defined, and the version reads `dev`.
 */
declare const __TS_DSPY_VERSION__: string | undefined;

/** The version of `@ts-dspy/core` in use, recorded in saved program state. */
export const VERSION: string =
    typeof __TS_DSPY_VERSION__ === 'string' ? __TS_DSPY_VERSION__ : 'dev';
