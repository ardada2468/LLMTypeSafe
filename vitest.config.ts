import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
    resolve: {
        // Resolve workspace packages to their sources so tests never run
        // against a stale `dist/` build.
        alias: [
            {
                find: /^@ts-dspy\/core\/testing$/,
                replacement: src('./packages/core/src/testing/index.ts'),
            },
            { find: /^@ts-dspy\/core$/, replacement: src('./packages/core/src/index.ts') },
        ],
    },
    test: {
        globals: true,
        environment: 'node',
        include: ['packages/*/src/**/*.test.ts'],
        coverage: {
            provider: 'v8',
            include: ['packages/*/src/**/*.ts'],
            exclude: [
                'packages/*/src/**/*.test.ts',
                'packages/*/src/index.ts',
                'packages/*/src/test-utils.ts',
            ],
            thresholds: { lines: 85, statements: 85, functions: 80, branches: 75 },
        },
    },
});
