# Security Policy

## Supported versions

The `@ts-dspy/*` packages are versioned together. Security fixes land on the
latest minor release line; older lines are not patched.

| Version | Supported |
| ------- | --------- |
| 0.7.x   | Yes       |
| < 0.7   | No        |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/ardada2468/LLMTypeSafe/security/advisories/new)
(the **Report a vulnerability** button on the repository's Security tab).
Include:

- the affected package and version,
- a description of the issue and its impact,
- a minimal reproduction, if you have one.

You can expect an acknowledgement within 3 business days and a status update
within 10. Once a fix is released, the advisory is published and you are
credited unless you ask not to be.

## Scope and threat model

Things worth knowing when you assess a report, or deploy the library:

- **Model output is untrusted input.** Everything a model returns is parsed and
  validated against the declared signature before your code sees it. A parsing
  or validation bypass — a value reaching your code with the wrong type, or
  model-controlled keys altering object prototypes — is a vulnerability.
- **`ProgramOfThought` is not a security boundary.** It runs model-written
  JavaScript in a `node:vm` context, which Node documents as not being a
  sandbox. Do not expose it to untrusted prompts without real isolation (a
  container, a separate process with dropped privileges, or a remote sandbox).
  Escapes from `node:vm` itself are out of scope.
- **Tools run with your process's privileges.** A `ReAct` agent calls the tools
  you hand it with arguments the model chose. Arguments are validated against
  each tool's schema, but authorisation — whether _this_ request may perform
  _that_ action — is the tool author's responsibility.
- **Prompt injection** against an application built on this library is an
  application-level concern, not a library vulnerability, unless the library
  makes it materially worse than calling the provider directly.
- **Secrets.** API keys are passed to provider SDKs and are never written to
  traces, cassettes, cache keys or error messages. A code path that leaks one
  is a vulnerability.
- **Telemetry.** `@ts-dspy/otel` does not record prompt or completion content
  unless `captureContent` is set. The library itself sends no telemetry.

## Supply chain

- Releases are published from GitHub Actions using npm trusted publishing
  (OIDC) with provenance attestations; there is no long-lived publish token.
  Verify a release with `npm audit signatures`.
- Dependencies are kept current by Dependabot, and pull requests are scanned by
  CodeQL and dependency review.
