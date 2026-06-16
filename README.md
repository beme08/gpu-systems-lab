# ai-systems-trainer (gpu)

A local terminal program that walks you through a GPU systems roadmap:
CUDA, Nsight, PyTorch benchmarking, bottleneck analysis, and later
Triton / LLM systems work.

It is not a checklist app. It is a small **AI Systems Engineer simulator**.

## Design Philosophy

The goal is to simulate how an AI Systems Engineer thinks:

1. What is the workload?
2. What hardware is it running on?
3. What is the bottleneck?
4. How do I measure it?
5. What optimization should I try?
6. Did the benchmark prove improvement?

The tool tracks both **task progress** and **skill growth**, and forces
written reflection on the answers that actually matter.

Every task in `roadmap.json` carries:

- `mode`        - the kind of lab (System Lab, Kernel Lab, Profiling Lab, ...)
- `objective`   - the observable you are trying to produce
- `constraint`  - the bottleneck or limit you must respect
- `deliverable` - the artifact you must produce (file, note, report)
- `skills`      - which skill bars it bumps, and by how much

Some tasks additionally require:

- a **bottleneck classification** (compute / memory / sync / launch / transfer)
- a **written reality check** (2-3 sentences, stored locally)

## What It Does (current version)

| Command           | Purpose |
|-------------------|---------|
| `python train.py start`    | Begin the program, show first task. |
| `python train.py next`     | Show the next pending task. |
| `python train.py status`   | Progress bar + current task + skill tree. |
| `python train.py skills`   | Skill tree only. |
| `python train.py done <id>`| Mark task complete (prompts for bottleneck / reality check when needed). |
| `python train.py check`    | Run a pending reality check on demand. |
| `python train.py explain`  | Show recorded bottleneck + reality-check log. |
| `python train.py tasks`    | List all task ids with completion marks. |
| `python train.py reset`    | Wipe all progress (asks first). |

State is stored in `storage.json` next to `train.py`. It is local JSON,
no server, no telemetry.

## Install

```bash
pip install typer rich
```

Python 3.9+. No GPU required to *run the CLI* - you only need a GPU
to actually execute the labs.

## Layout

```
.
├── train.py             # the CLI (single file for v0.1)
├── roadmap.json         # tasks, skills, prompts
├── storage.json         # local progress (created on first run)
├── labs/
│   ├── vector_add/      # first CUDA kernel
│   └── pytorch/         # matmul benchmark
├── notes/
│   └── hardware.md      # fill this in for task #1
└── reports/
    └── gpu-week1-report.md  # final synthesis
```

## Build Order

This project was built bottom-up so you can ship each version on its own:

| Version | What | Status |
|---------|------|--------|
| v0.1 | Checklist + progress bar + local storage | done |
| v0.2 | Skill tree (6 skills, weighted by task) | done |
| v0.3 | Bottleneck engine (prompt on `done`) | done |
| v0.4 | Reality checks (stored, on-demand `check`) | done |
| v0.5 | Open-source polish (LICENSE, pyproject, swap roadmaps) | next |

## Swapping Roadmaps

`roadmap.json` is the single source of content. To add a new track, copy
it and edit:

```bash
cp roadmap.json roadmap.triton.json
# edit ids, titles, skills, prompts
mv storage.json storage.gpu.json     # keep your current progress
```

The schema lives in `train.py:load_roadmap()`. Skills must be declared
in the top-level `skills` array; the storage layer backfills new skill
ids automatically.

A `--roadmap` flag is the obvious next addition; for now, symlink or
swap the file.

## License

MIT.
