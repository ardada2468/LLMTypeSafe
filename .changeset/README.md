# Changesets

This folder holds [changesets](https://github.com/changesets/changesets): short
notes describing changes that should appear in a release.

Add one with `npx changeset` whenever you change a published package. On merge to
`main`, the release workflow opens a version PR; merging that PR publishes to npm
and writes each package's `CHANGELOG.md`.

The `@ts-dspy/*` packages are versioned together (`fixed` in `config.json`), so a
release bumps all of them to the same version and rewrites the internal
`@ts-dspy/core` ranges automatically — the drift that shipped
`@ts-dspy/openai@0.4.2` depending on `@ts-dspy/core@^0.3.0` cannot recur.
