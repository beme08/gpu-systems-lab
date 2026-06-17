# .githooks/

Opt-in pre-commit hooks for `gpu-systems-lab`.

## Enable

```sh
git config core.hooksPath .githooks
```

This applies only to your local clone; it is not enabled by default
when someone else clones the repo.

## What it does

`pre-commit` scans the **staged diff** (`git diff --cached`) for
likely secrets and non-allowlist personal emails. It only inspects
**added lines**, so untracked local files (`hardware.md`, your own
notes in `notes/`, etc.) do not block commits.

It checks for:

- OpenAI API keys (`sk-...`, `sk-proj-...`)
- AWS access key IDs (`AKIA...`)
- GitHub personal access tokens (`ghp_...`, `gho_...`,
  `github_pat_...`)
- PEM private key headers (`-----BEGIN ... PRIVATE KEY-----`)
- Staged additions of dotenv files (`.env`, `.env.local`,
  `.env.production`, `.env.development`); templates
  (`.env.example`, `.env.sample`, `.env.template`) are allowed.
- Personal email addresses on added lines, with an allowlist for
  `*.test`, `*.example`, `*.sample`, `example.com`, `example.org`,
  `example.net`, `company.com`, `noreply@github.com`, and
  `users.noreply.github.com`.

## What it does NOT do

- It does not scan the working tree, unstaged changes, or commit
  history. Run `git log -p | grep ...` separately if you need
  history-level scans.
- It does not call any external service. All matching is local
  `grep`.
- It does not enforce anything on push; GitHub's own push
  protection and secret scanning cover server-side cases.

## Bypass intentionally

```sh
git commit --no-verify
```

Use this when you have a real reason to commit something the hook
would block (a known-safe example, a documented test fixture, etc.).
The hook prints this hint every time it runs.

## Adding a pattern

Edit `.githooks/pre-commit`. The `SECRET_PATTERNS` array uses the
format `'<regex>|<human label>'`. The `EMAIL_ALLOWLIST` array uses
extended-regex fragments (no `|` separators needed; they are joined
with `|` at runtime).

Re-run `bash -n .githooks/pre-commit` to syntax-check.
