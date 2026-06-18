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
  "compute_paths": ["runpod", "lambda_labs"],
  "bottleneck_pick": false,
  "reality_check": ""
}

`compute_paths` is a list of resource ids from the top-level
`resources` list. Set it on any Week 2+ task that has `commands[]`
and that needs a Linux NVIDIA GPU box. The per-task card renders
it as `Run on: RunPod, Lambda Labs. See \`gpu resources --domain
compute\` for setup notes.` Use the existing compute platforms
(google_colab, kaggle_notebooks, unsloth, runpod, lambda_labs)
unless you have a real reason to add a new resource.

## Adding teaching prompts (v0.16)

`teaching_prompts` is a list of objects on a task. Each entry has:

- `question`: a free-text question to ask the user.
- `expected_keywords`: a list of substrings the answer is checked
  against (case-insensitive). 2+ hits = "match".
- `follow_up_if_match`: the feedback string when 2+ keywords hit.
- `follow_up_if_miss`: the feedback string when 0-1 keywords hit.

When `gpu done <id>` (or `gpu teach <id>`) hits a task with
`teaching_prompts`, it walks the list, prompts the user, computes
feedback via `give_teaching_feedback`, and appends the record to
`storage.json["teaching"][task_id]`. The log is visible via
`gpu explain --id <id>`. Tasks without `teaching_prompts` are
unaffected.

Hand-coded feedback in v0.16, richer layered feedback in v0.18,
optional LLM feedback in v0.17.

v0.18 layered feedback (in `give_teaching_feedback`):

  1. `common_misconceptions` (regex match) -> specific "not quite" line.
  2. `expected_answers` (regex match, in order) -> first matching response.
  3. `expected_keywords` (substring count >= 2) -> `follow_up_if_match`.
  4. No match -> `follow_up_if_miss`, or a "Consider: <keywords>" line.

The more specific the rule that fires, the better the feedback.
Misconceptions fire BEFORE keyword counts because a common wrong
answer is more useful to call out than a generic miss. Both new
fields are optional and backward-compatible: a prompt with only
`expected_keywords` behaves exactly like v0.16.

Example v0.18 prompt:

```json
{
  "question": "Why is vector add usually memory-bound?",
  "expected_keywords": ["memory", "bandwidth"],
  "expected_answers": [
    {"pattern": "memory.*bandwidth|read.*write",
     "response": "Right - the bottleneck is bytes-per-flop."}
  ],
  "common_misconceptions": [
    {"pattern": "compute.?(bound|heavy)",
     "response": "Not quite - vector add does ~1 FLOP per element."}
  ],
  "follow_up_if_match": "Good. Now: how would you change the kernel?",
  "follow_up_if_miss": "Hint: think about bytes-per-flop."
}
```

v0.17 LLM feedback (additive, never blocking):

`gpu teach <id> --llm` adds an LLM-generated feedback line in
addition to the hand-coded one. v0.18 enriches the LLM system
prompt with the task's `objective` and `deliverable` so the
feedback is anchored to the specific task, not generic. The LLM
call uses stdlib (urllib) only - no `openai` SDK, no subprocess.
If `OPENAI_API_KEY` is missing or the API call fails, the
hand-coded feedback still records and the command exits 0. The
LLM feedback is stored as `llm_feedback` on the teaching record
and rendered as a 5th column in `gpu explain --id <id>`.
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

Run the unit test suite from the repo root:

```bash
pip install -e ".[dev]"
pytest
```

The test suite is ~110 unit tests covering regex matching, cache lookup,
score math, storage backfill, render functions, and prompt flows. It
runs in under a second. CI runs the same command on every push to
`main` (see `.github/workflows/test.yml`). New units in `gpu.py` should
land with corresponding tests in `tests/`.

For end-to-end smoke checks (the v0.x manual ritual, still useful):

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
