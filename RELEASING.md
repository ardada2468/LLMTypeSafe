# Releasing

Releases run on [changesets](https://github.com/changesets/changesets) and publish
through [npm trusted publishing](https://docs.npmjs.com/trusted-publishers) (OIDC).

## Day to day

Add a changeset with any change to a published package:

```bash
npx changeset
```

Pick the packages and bump type, and describe the change for the changelog. For a
change that should ship no release — docs, CI, tests — use `npx changeset --empty`.
CI fails a pull request that touches a published package without one.

On merge to `main`, the release workflow opens a **Version Packages** pull request
that applies the pending changesets: bumping versions, rewriting the internal
`@ts-dspy/*` ranges, and updating changelogs. Merging _that_ publishes to npm.

Nothing publishes without the full CI gate set passing first — `release.yml` calls
`ci.yml` and depends on it.

## Credentials

There is no `NPM_TOKEN`. Trusted publishing exchanges the GitHub Actions OIDC
identity for a short-lived, job-scoped npm credential, so there is no long-lived
secret to leak and nothing to rotate. npm now caps granular tokens at 90 days, so
a token-based setup would need rotating every quarter; this does not.

Requirements, all already configured in `release.yml`:

- `id-token: write` permission on the publishing job
- npm >= 11.5.1 (pinned explicitly, not inherited from the runner image)
- No `NODE_AUTH_TOKEN` — npm detects the OIDC environment on its own

Provenance attestations are published automatically; the `--provenance` flag is
not needed and is deliberately absent.

## One-time setup per package

For each package, on npmjs.com → the package → **Settings** → **Trusted Publisher**,
choose **GitHub Actions** and enter:

| Field             | Value         |
| ----------------- | ------------- |
| Organization/user | `ardada2468`  |
| Repository        | `LLMTypeSafe` |
| Workflow filename | `release.yml` |

The workflow filename must match exactly, since npm authorizes that specific
workflow rather than the repository as a whole.

### The new-package wrinkle

npm requires a package to **already exist** before a trusted publisher can be
configured for it. `@ts-dspy/core`, `@ts-dspy/openai`, and `@ts-dspy/gemini` are
published, so they can be configured right away.

`@ts-dspy/anthropic` is new. Publish it once by hand, then configure its trusted
publisher like the others:

```bash
npm login                       # session-based, expires after two hours
npm publish -w @ts-dspy/anthropic --access public
```

Or use the helper built for this gap, which publishes a minimal placeholder so the
package exists and can be configured:

```bash
npx setup-npm-trusted-publish @ts-dspy/anthropic
```

Do this **before** merging the Version Packages pull request. Otherwise the release
publishes three packages and fails on the fourth, leaving the release half-applied.

## Troubleshooting

**`ENEEDAUTH` / 401 on publish** — the trusted publisher is not configured for that
package, or the workflow filename in its npm settings does not match `release.yml`.

**`npm error code EUSAGE ... --provenance`** — provenance is automatic under trusted
publishing; remove any explicit flag or `NPM_CONFIG_PROVENANCE`.

**Publishing is skipped entirely** — there were no pending changesets, so there was
nothing to release. Check that the Version Packages pull request was merged.
