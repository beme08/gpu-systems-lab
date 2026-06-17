# gpu-systems-lab

> A terminal-based training loop for learning CUDA, GPU profiling, and AI systems engineering.
> GitHub: <https://github.com/beme08/gpu-systems-lab>

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
| `gpu start`    | First run: full walkthrough (welcome, what this is, shape, commands, first task). Later runs: quiet. |
| `gpu next`     | Show the next pending task. |
| `gpu status`   | Progress bar + current task + tracks + skill tree + Pending panel + Next-hint line + Last-activity line + Next-milestone line. |
| `gpu skills`   | Skill tree only. |
| `gpu done <id> [--bench <path>]` | Mark task complete (prompts for bottleneck / reality check when needed; attach a benchmark artifact with `--bench`). |
| `gpu resources` | List roadmap resources (papers, libraries, tools), with `--domain`, `--tag`, `--difficulty` filters and `--open <id>`. Use `--domain compute` to see platforms. |
| `gpu compute`  | Show where to run GPU tasks (compute platforms with Good-for / Not-good-for columns; `--open <id>` launches a URL). |
| `gpu score` | Show weighted program + per-track + per-milestone score. |
| `gpu check`    | Run a pending reality check on demand. |
| `gpu explain`  | Show recorded bottleneck + reality-check log. |
| `gpu tasks`    | List all task ids with completion marks. |
| `gpu reset`    | Wipe all progress (asks first). |

State is stored in `storage.json` next to `gpu.py`. It is local JSON,
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
├── gpu.py               # the CLI (single file)
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

## Project links

- [SECURITY.md](SECURITY.md) - how to report a vulnerability.
- [CONTRIBUTING.md](CONTRIBUTING.md) - how to add a task, milestone, or resource.

## Build Order

This project was built bottom-up so you can ship each version on its own:

| Version | What | Status |
|---------|------|--------|
| v0.1 | Checklist + progress bar + local storage | done |
| v0.2 | Skill tree (6 skills, weighted by task) | done |
| v0.3 | Bottleneck engine (prompt on `done`) | done |
| v0.4 | Reality checks (stored, on-demand `check`) | done |
| v0.5 | Track-aware schema (cuda / llm / systems / distillation) | done |
| v0.6 | Open-source polish (LICENSE, pyproject, command rebrand to `gpu`) | done |
| v0.7 | DAG-ready schema + scoring metadata + `resources` + `gpu` alias | done |
| v0.8 | Benchmark logging (file + summary) + `gpu resources` + `gpu score` panel | done |
| v0.9 | First-run walkthrough: welcome + design philosophy + program shape + commands + first task | done |
| v0.10 | Surgical hint pass: 'When done' preview on task cards, Pending panel + Next hint in `status`, re-run detection on `done`, unknown-id tip | done |
| v0.11 | Timestamps (`completed_at`) + 'Last activity' line in `status` + reality-check prompt scaffold + 'Next milestone' line + `gpu done <completed>` tip | done |
| v0.12 | Multi-week curriculum (Week 2 Triton, Week 3 LLM inference, Week 4 GPU serving - 18 new tasks, 3 new milestones) + 5 platform resources + `gpu compute` command + 'On Mac?' walkthrough hint | done |
| v0.13 | Milestone-aware end-of-curriculum panel (replaces v0.6 'Phase 1 complete' copy in `done` / `start` / `next`) | done |
| v0.14 | Compute-platform task wiring: `compute_paths` on Week 2+ tasks + per-task 'Run on:' hint in the task card + expanded 'On Mac?' walkthrough hint | done |
| v0.15 | Distillation track (6 Week 5+ tasks: int8 PTQ, LoRA, teacher-student KD, eval comparison, int4 quant, synthesis) + 3 new skills + 5th milestone | done |

## Tracks

Tracks are declared in `roadmap.json` under the `tracks` key and shown
in `gpu status` as a per-track progress panel between the current task
and the skill tree. The CUDA track has Week 1 + Week 2 (Triton) tasks (14 total).
The LLM track has Week 3 (LLM inference) tasks (6 total).
The Systems track has Week 4 (GPU serving) tasks (6 total).
The Distillation track is a placeholder for v0.13.
To add tasks to a track, just add a task with `"track": "llm"` (etc.)
to the `tasks` array — it will appear in that track's bar automatically.

### Resources

Resources (papers, libraries, tools) live in `roadmap.json` under the
`resources` key and are exposed in v0.8 via `gpu resources`. Each entry
has `id`, `title`, `url`, `domains` (list), `tags` (list), and
`difficulty` (beginner | intermediate | advanced).

First resource: **ThunderKittens** (HazyResearch) — a kernels DSL for
tensor-core attention-style workloads, advanced difficulty. Lives under
the kernels domain.

### Compute platforms

The roadmap also carries compute-platform resources (Colab, Kaggle, Unsloth,
RunPod, Lambda Labs, ...) tagged with `domain: compute`. Two ways to see them:

- `gpu compute` - focused view with Good-for / Not-good-for columns and a
  Mac preamble.
- `gpu resources --domain compute` - same data in the standard resources
  table.

`gpu compute --open <id>` launches the URL in your default browser.

### Benchmarks

Every `gpu done` can record a benchmark artifact:

```bash
gpu done <id> --bench <path>
```

The CLI expands `~` and repo-relative paths, prints a one-line summary
prompt, and stores `{path, summary, recorded_at}` in `storage.json` under
`benchmarks[<id>]`. The path is recorded even if the file does not
exist (light validation; a warning goes to stderr). Run
`gpu explain --id <id>` to see the recorded benchmark. The schema is
`v0.8`; `load_storage` backfills the `benchmarks` key for older files.

## Swapping Roadmaps

`roadmap.json` is the single source of content. To add a new track, copy
it and edit:

```bash
cp roadmap.json roadmap.triton.json
# edit ids, titles, skills, prompts
mv storage.json storage.gpu.json     # keep your current progress
```

The schema lives in `gpu.py:load_roadmap()`. Skills must be declared
in the top-level `skills` array; the storage layer backfills new skill
ids automatically.

A `--roadmap` flag is the obvious next addition; for now, symlink or
swap the file.

## License

MIT.
