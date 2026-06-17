# Contributing to gpu-systems-lab

Thanks for your interest. This document covers the small set of
edits that are easy to land: adding tasks, milestones, and resources.
Larger changes (new commands, schema changes, dependency changes)
should start as a GitHub issue so we can discuss shape first.

## Adding a task

Tasks live in `roadmap.json` under the top-level `tasks` list.
Append a new entry with the existing schema:

```json
{
  "id": "my_new_task",
  "milestone": "<milestone id from the milestones dict>",
  "track": "cuda | llm | systems | distillation",
  "mode": "GPU System Lab | CUDA Kernel Lab | Profiling Lab | ...",
  "title": "Short, present-tense title",
  "objective": "The observable you are trying to produce.",
  "constraint": "The bottleneck or limit you must respect.",
  "deliverable": "The artifact the user must produce (file path or note).",
  "score": { "impact": 1, "depth": 1, "reproducibility": 1 },
  "skills": { "cuda_execution_model": 5 },
  "prerequisites": ["<other task id>"],
  "commands": ["$ <command>", "$ <other command>"],
  "bottleneck_pick": false,
  "reality_check": ""
}
```

Only `id`, `milestone`, `track`, `title`, `objective`, `constraint`,
`deliverable`, and `score` are required. `commands`,
`bottleneck_pick`, and `reality_check` are optional. The score
formula is `impact * depth * reproducibility`; the program total
normalizes to 0-100 per track and program-wide.

After adding the entry, run `gpu done <id>` to confirm the new task
is wired up, and `gpu status` to confirm the next-task hint
advances.

## Adding a milestone

Milestones live in `roadmap.json` under the top-level `milestones`
dict. Declare the milestone first, then reference its id from each
task's `milestone` field:

```json
"milestones": {
  "<new_id>": {
    "title": "My new milestone",
    "tasks": ["task_id_1", "task_id_2"]
  }
}
```

`gpu status` will surface a "Next milestone" line that points at
the first milestone in declared order that is not 100% done. Add
new milestones at the end of the dict unless the ordering
intentionally matters.

## Adding a resource

Resources live in `roadmap.json` under the top-level `resources`
list. The most common kind is a compute platform (consumed by
`gpu compute` and `gpu resources --domain compute`):

```json
{
  "id": "my_provider",
  "title": "My Provider",
  "url": "https://example.com/",
  "domains": ["compute"],
  "tags": ["cloud-gpu", "linux"],
  "difficulty": "beginner | intermediate | advanced",
  "good_for": "Triton, vLLM, ...",
  "not_good_for": "Anything that needs ...",
  "notes": "Free, on-demand, ..."
}
```

The `good_for` / `not_good_for` / `notes` fields are optional; they
are rendered only by `gpu compute`. The `domains` field is what
`gpu resources --domain <x>` matches against.

## Testing your change

This repo has no test framework. Smoke checks before opening a PR:

- `gpu done <id>` on the new task: prompts fire as expected
  (bottleneck, reality, summary).
- `gpu status`: tracks and milestones roll up correctly.
- `gpu score`: per-track and per-milestone percentages are sane.
- `gpu explain --id <id>`: bottleneck / reality / benchmark
  sections all render when present, stay quiet when absent.
- `gpu compute` and `gpu resources --domain compute`: new resources
  appear in the right places.

## Style

A few conventions worth keeping:

- No new top-level dependencies. `typer` and `rich` are the only
  declared deps; add to that list only with prior discussion.
- No subprocess calls from the CLI. The current `gpu.py` reads
  `roadmap.json` and `storage.json`; it does not run `nvidia-smi`,
  `nsys`, `ncu`, or any task command. Stay that way unless there
  is a real reason.
- The `--bench` path validator is light: it expands `~`, resolves
  relative paths against the repo root, and warns to stderr if the
  resolved path does not exist. It does not copy, hash, or read
  the file. Keep it that way.
- If you add a new prompt to `gpu done`, add a backfill in
  `load_storage` for legacy `storage.json` files so older local
  progress loads cleanly.
- `roadmap.json` is the source of truth for the curriculum. Do not
  hard-code task ids, milestone titles, or skill names anywhere
  else in the code.
