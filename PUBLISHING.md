# Publishing gpu-systems-lab

This project is currently **local-only** — it lives on this machine and is
not pushed to any remote. `git remote -v` is empty.

The roadmap task `cuda_installed` is the first task a new user runs, and
its deliverable (`notes/hardware.md`) is a per-machine file. That's one
reason this repo is not set up to be public by default.

When you are ready to publish, here is the short recipe.

## One-time setup

```sh
# Re-auth the GitHub CLI (the current token is reported as invalid).
gh auth login

# Create the repo and push in one step. Choose public or private.
gh repo create beme08/gpu-systems-lab \
    --public \
    --source=. \
    --description "A terminal-based training loop for learning CUDA, GPU profiling, and AI systems engineering." \
    --remote=origin \
    --push
```

For a private repo, swap `--public` for `--private`.

## After publishing

```sh
git push                     # future commits
gh repo view --web           # open the repo in the browser
```

## What is in the repo

- `gpu.py` — the CLI.
- `roadmap.json` — the curriculum data (tasks, tracks, milestones, resources, scoring).
- `storage.json` — local progress (created on first run).
- `labs/` — example CUDA kernel + PyTorch benchmark.
- `notes/hardware.md` — the `cuda_installed` deliverable; safe to keep committed on a Mac (no NVIDIA GPU), but other contributors will have their own.
- `reports/gpu-week1-report.md` — placeholder for the week-1 synthesis.
- `README.md`, `PUBLISHING.md` (this file), `LICENSE` (MIT), `pyproject.toml`.

## When you go public

`notes/hardware.md` documents *your* machine. Two options:

1. **Keep it committed** — it shows what a Mac (no NVIDIA) looks like, which is a useful data point for other Mac users.
2. **Move to a per-machine pattern** — replace with a `notes/hardware.example.md` template and add `notes/hardware.md` to `.gitignore`, so each clone fills in its own.

If you do not want to decide now, leave it as-is; the file is small and clearly labelled.
