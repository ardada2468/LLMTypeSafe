# Contributing

Thanks for helping. This page is the short version of how changes land.

## Setup

Requires Node.js 22 or newer (`.nvmrc` pins the version CI builds with).

```bash
git clone https://github.com/ardada2468/LLMTypeSafe.git
cd LLMTypeSafe
npm install
npm run build      # core first — the other packages typecheck against its declarations
npm test
```

## Layout

| Path                 | What lives there                                               |
| -------------------- | -------------------------------------------------------------- |
| `packages/core`      | Signatures, adapters, modules, optimizers, evaluation, testing |
| `packages/openai`    | OpenAI and OpenAI-compatible providers                         |
| `packages/anthropic` | Anthropic provider                                             |
| `packages/gemini`    | Gemini provider                                                |
| `packages/ai-sdk`    | Vercel AI SDK bridge                                           |
| `packages/mcp`       | Model Context Protocol client and server helpers               |
| `packages/otel`      | OpenTelemetry instrumentation                                  |
| `examples`           | Runnable examples, typechecked in CI                           |
| `site`               | The static docs site, deployed to GitHub Pages                 |

## Before you open a pull request

Run what CI runs:

```bash
npm run build
npm run lint
npm run typecheck
npm run format:check
npm test -- --coverage
npm run verify:packaging   # if you touched a package.json, exports or build config
```

And check that:

- **Tests cover the change.** Tests never touch the network — use `MockLM` or
  `CassetteLM` from `@ts-dspy/core/testing`, or stub the provider SDK as the
  existing provider tests do. A bug fix comes with a test that fails without it.
- **Library code does not write to the console.** Surface information through
  return values, thrown errors or callbacks; ESLint enforces this.
- **Errors are typed.** Throw a `TsDspyError` subclass, never a bare `Error`,
  so callers can branch on it.
- **Public API changes are documented** in the README (and MIGRATION.md if
  they break something).
- **There is a changeset.** Run `npx changeset`, pick the packages and bump
  type, and describe the change for users. For a change that ships no release
  (docs, CI, tests), run `npx changeset --empty`. CI fails a pull request that
  changes a published package without one.

## Commit and PR conventions

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `fix(core): …`, `feat(openai): …`, `docs: …`, `chore(ci): …`.
- Keep pull requests focused; a refactor and a behaviour change are two PRs.
- Breaking changes need a `major` (or, below 1.0, `minor`) changeset and a
  MIGRATION.md entry.

## Adding a provider

Extend `BaseLM` from `@ts-dspy/core`, map the vendor SDK's errors onto the core
error classes with `classify`, report usage through `recordUsage`, and declare
`capabilities` honestly — adapters choose their strategy from them. Copy the
structure of `packages/anthropic`, including its `tsup.config.ts`, and add the
package to `build:providers` in the root `package.json` and to the packaging
loops in `.github/workflows/ci.yml` and `scripts/verify-packaging.js`.

## Releasing

Maintainers only — see [RELEASING.md](RELEASING.md).

## Conduct and security

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).
