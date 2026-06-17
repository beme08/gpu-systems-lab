# Security Policy

## Reporting a vulnerability

If you find a security issue in `gpu-systems-lab`, please report it
privately. Do not open a public GitHub issue for it.

The preferred channel is **GitHub Security Advisories**:

- Go to the [Security tab](https://github.com/beme08/gpu-systems-lab/security/advisories/new)
  of this repository and open a new draft advisory.

If you cannot use GitHub Advisories, email
`beme08@users.noreply.github.com` (a GitHub-provided noreply address
that forwards to the maintainer). Please include a short
reproduction, the commit SHA, and your assessment of impact.

## What to expect

- **Acknowledgement** within 7 days of the report.
- **Triage** within 14 days: confirm, scope, and decide on a fix
  versus a workaround.
- **Fix** for confirmed issues is targeted for the next minor release
  of the CLI. Critical issues may be patched out-of-band.
- **Credit** in the release notes is given on request, with the
  identifier you choose (handle, real name, or anonymous).

## Scope

In scope:

- Code execution from a malicious `roadmap.json` or `storage.json`
  (the CLI uses `json.load`; the only way to inject is to coerce the
  user into running a tampered file).
- Path-handling bugs in the `gpu done --bench <path>` validator that
  could let a recorded benchmark path leak the user's directory
  layout against their will.
- Any subprocess that would be run by the CLI. (The current CLI
  intentionally runs no subprocess, so the bar for "in scope" is
  "the next version that adds one.")

Out of scope:

- The contents of `notes/`, `reports/`, `labs/`, and `hardware.md`.
  These are per-machine user artifacts; they are not part of the
  shipped product and are not tracked in this repo (with the
  exception of stub templates).
- Dependencies of `typer` and `rich`. Report those upstream.
