import { defineConfig } from 'tsup';

export default defineConfig({
    // `src/testing/index.ts` backs the `@ts-dspy/core/testing` subpath export;
    // package.json declares it, so it must be built or publint fails.
    entry: ['src/index.ts', 'src/testing/index.ts'],
    format: ['esm', 'cjs'],
    // Both entries share BaseLM, LMError, and friends. Without splitting the CJS
    // build inlines a second copy of each into dist/testing, so a MockLM would
    // not be `instanceof BaseLM` and a cassette's LMError would not be caught by
    // `instanceof LMError` under `require`.
    splitting: true,
    dts: true,
    sourcemap: true,
    clean: true,
    target: 'es2022',
});
