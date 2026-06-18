#!/usr/bin/env python3
"""
gpu-systems-lab: a local CLI that walks you through a GPU systems
roadmap (CUDA, Nsight, PyTorch, profiling, kernels).

Design philosophy: this is not just a checklist. The goal is to simulate
how an AI Systems Engineer thinks:

  1. What is the workload?
  2. What hardware is it running on?
  3. What is the bottleneck?
  4. How do I measure it?
  5. What optimization should I try?
  6. Did the benchmark prove improvement?

The tool tracks both task progress and skill growth.

v0.1  progress + checklist
v0.2  skill tree
v0.3  bottleneck engine
v0.4  reality checks
"""

from __future__ import annotations

import json
import os
import re as _re
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
ROADMAP_PATH = ROOT / "roadmap.json"
STORAGE_PATH = ROOT / "storage.json"

app = typer.Typer(
    add_completion=False,
    help="gpu-systems-lab - walk through a GPU systems roadmap.",
)
console = Console()

BOTTLENECK_CHOICES = [
    ("compute_bound",        "Compute bound (FLOPs / SM throughput)"),
    ("memory_bound",         "Memory bound (DRAM / L2 / shared mem bandwidth)"),
    ("sync_bound",           "Sync bound (CPU<->GPU stalls, events, streams)"),
    ("launch_overhead",      "Kernel launch overhead"),
    ("host_device_transfer", "Host <-> device transfer (PCIe / NVLink)"),
]
BOTTLENECK_IDS = {b[0] for b in BOTTLENECK_CHOICES}


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

def default_storage(roadmap: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "started": False,
        "completed": [],
        "skills": {s["id"]: 0 for s in roadmap.get("skills", [])},
        "bottlenecks": {},   # task_id -> bottleneck id
        "reality": {},       # task_id -> user's written answer
        "benchmarks": {},    # task_id -> {path, summary, recorded_at}
        "welcomed": False,   # True once gpu start has shown the first-run walkthrough
        "completed_at": {},  # task_id -> ISO 8601 UTC string ("when was this task done")
        "teaching": {},      # task_id -> list of {question, answer, feedback, asked_at} (v0.16)
        "llm_cache": {},     # "task_id|prompt_index|answer_norm" -> {model, text, cached_at} (v0.20)
    }


def load_roadmap() -> Dict[str, Any]:
    if not ROADMAP_PATH.exists():
        console.print(f"[red]roadmap.json not found at {ROADMAP_PATH}[/red]")
        raise typer.Exit(1)
    try:
        return json.loads(ROADMAP_PATH.read_text())
    except PermissionError as e:
        # EPERM here is the macOS "com.apple.provenance" sandbox xattr
        # returning a synthetic permission error on certain reads. The file
        # itself is fine; it just has a 3-byte sentinel xattr that some
        # process contexts reject. Re-create the file from outside any
        # sandboxed app (e.g. your normal terminal) to clear the xattr.
        home_hint = '$HOME/Documents/gpu/roadmap.json'
        repo_dir = '$HOME/Documents/gpu'
        console.print(
            f"[red]PermissionError reading {ROADMAP_PATH} ({e}).[/red]\n"
            "[yellow]macOS com.apple.provenance xattr is blocking this read.\n"
            "The file content is fine; the xattr was added by a sandboxed app.\n"
            f"Fix:  cp '{home_hint}' /tmp/roadmap.json && mv /tmp/roadmap.json '{home_hint}'\n"
            "Or:   re-create it from git in your own terminal:\n"
            f"        git -C '{repo_dir}' show HEAD:roadmap.json > '{home_hint}'"
        )
        raise typer.Exit(1)




def load_storage() -> Dict[str, Any]:
    roadmap = load_roadmap()
    if not STORAGE_PATH.exists():
        return default_storage(roadmap)
    try:
        data = json.loads(STORAGE_PATH.read_text())
    except json.JSONDecodeError:
        return default_storage(roadmap)
    # backfill any new skill keys
    for sid in {s["id"] for s in roadmap.get("skills", [])}:
        data.setdefault("skills", {}).setdefault(sid, 0)
    for key in ("completed", "bottlenecks", "reality", "benchmarks"):
        data.setdefault(key, {} if key != "completed" else [])
    data.setdefault("welcomed", False)
    data.setdefault("completed_at", {})
    data.setdefault("teaching", {})
    data.setdefault("llm_cache", {})
    return data


def save_storage(data: Dict[str, Any]) -> None:
    STORAGE_PATH.write_text(json.dumps(data, indent=2))


# ----------------------------------------------------------------------------
# Task lookup
# ----------------------------------------------------------------------------

def task_by_id(roadmap: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    for t in roadmap["tasks"]:
        if t["id"] == task_id:
            return t
    return None


def next_task(tasks: List[Dict[str, Any]], completed: List[str]) -> Optional[Dict[str, Any]]:
    for t in tasks:
        if t["id"] not in completed:
            return t
    return None


def all_task_ids(roadmap: Dict[str, Any]) -> List[str]:
    return [t["id"] for t in roadmap["tasks"]]


# ----------------------------------------------------------------------------
# Benchmark path expansion (Q4 - light validation)
# ----------------------------------------------------------------------------

def expand_bench_path(raw: str, repo_root: Path) -> str:
    """Expand a user-supplied benchmark path.

    Rules (light validation only - never raise):
      1. A leading ``~`` is expanded to ``$HOME``.
      2. Relative paths are resolved against ``repo_root`` (the directory
         containing ``roadmap.json`` / ``storage.json``).
      3. The post-tilde string is *not* checked for existence here; the
         caller decides what to do with it.

    The returned string is what gets stored in ``storage.benchmarks`` so the
    user sees the path they typed.
    """
    expanded = os.path.expanduser(raw)
    p = Path(expanded)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return str(p)


# ----------------------------------------------------------------------------
# Score helpers
# ----------------------------------------------------------------------------

def _task_raw_score(task: Dict[str, Any]) -> int:
    """Return a task's raw score (impact * depth * reproducibility), or 0."""
    s = task.get("score")
    if not isinstance(s, dict):
        return 0
    try:
        return int(s["impact"]) * int(s["depth"]) * int(s["reproducibility"])
    except (KeyError, TypeError, ValueError):
        return 0


def _track_score(track_tasks: List[Dict[str, Any]], completed: List[str]) -> int:
    """Return a 0-100 integer score for a slice of tasks."""
    earned = sum(_task_raw_score(t) for t in track_tasks if t["id"] in completed)
    possible = sum(_task_raw_score(t) for t in track_tasks)
    if possible == 0:
        return 0
    return int(round(earned * 100 / possible))


# ----------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------

def render_progress(tasks: List[Dict[str, Any]], completed: List[str]) -> None:
    total = len(tasks)
    done = sum(1 for t in tasks if t["id"] in completed)
    pct = (done / total * 100) if total else 0
    filled = int(round(pct / 100 * 10))
    bar = "█" * filled + "░" * (10 - filled)
    line = f"[bold blue]Progress[/bold blue]  {done}/{total} tasks complete  [cyan]{bar}[/cyan] {pct:5.1f}%"
    console.print(line)


def render_task(task: Dict[str, Any], is_current: bool = True, roadmap: Optional[Dict[str, Any]] = None) -> None:
    title = task["title"]
    if is_current:
        title = f"NEXT: {title}"
    panel = Panel(
        render_task_body(task, roadmap=roadmap),
        title=f"[bold]{title}[/bold]",
        border_style="cyan" if is_current else "white",
    )
    console.print(panel)


def render_task_body(task: Dict[str, Any], roadmap: Optional[Dict[str, Any]] = None) -> Text:
    text = Text()
    text.append("Mode:       ", style="bold")
    text.append(f"{task.get('mode', '?')}\n")
    if task.get("constraint"):
        text.append("Constraint: ", style="bold")
        text.append(f"{task['constraint']}\n")
    text.append("Objective:  ", style="bold")
    text.append(f"{task.get('objective', '?')}\n")
    text.append("Deliverable:", style="bold")
    text.append(f" {task.get('deliverable', '?')}\n")

    cmds = task.get("commands") or []
    if cmds:
        text.append("\nCommands:\n", style="bold")
        for c in cmds:
            text.append(f"  $ {c}\n", style="green")

    if task.get("bottleneck_pick"):
        text.append("\nBottleneck question:\n", style="bold yellow")
        text.append("  Classify the main bottleneck (compute / memory / sync / launch / transfer).\n")
    if task.get("bottleneck_hint"):
        text.append("\nBottleneck reflection:\n", style="bold yellow")
        text.append(f"  {task['bottleneck_hint']}\n")
    if task.get("bottleneck") and not task.get("bottleneck_pick"):
        text.append("\nConstraint: ", style="bold yellow")
        text.append(f"expected bottleneck = {task['bottleneck']}\n")
    if task.get("reality_check"):
        text.append("\nReality check:\n", style="bold magenta")
        text.append(f"  {task['reality_check']}\n")

    skill_deltas = task.get("skills") or {}
    if skill_deltas:
        text.append("\nSkill deltas:\n", style="bold")
        for k, v in skill_deltas.items():
            text.append(f"  {k} +{v}\n")

    prereqs = task.get("prerequisites") or []
    if prereqs:
        text.append("\nPrerequisites: ", style="bold")
        text.append(", ".join(prereqs) + "\n")

    score = task.get("score")
    if isinstance(score, dict):
        try:
            i = int(score["impact"])
            d_ = int(score["depth"])
            r = int(score["reproducibility"])
            product = i * d_ * r
        except (KeyError, TypeError, ValueError):
            i = d_ = product = None
            r = None
        if i is not None:
            text.append("\nScore: ", style="bold")
            text.append(f"{i} x {d_} x {r} = {product}\n")

    # Compute-paths hint (v0.14). Reads `task["compute_paths"]` (a list of
    # resource ids) and resolves each to a human title from the roadmap's
    # resources list. Renders as one line: "Run on: RunPod, Lambda Labs.
    # See `gpu resources --domain compute` for setup notes." Only renders
    # when the field is present and non-empty, and only when a roadmap is
    # available to resolve the ids. Quiet otherwise, matching the Q5 spec
    # for `gpu explain`.
    paths = task.get("compute_paths") or []
    if paths and roadmap is not None:
        res_by_id = {r["id"]: r for r in (roadmap.get("resources") or [])}
        names = []
        for pid in paths:
            r = res_by_id.get(pid)
            if r is not None:
                names.append(r.get("title", pid))
        if names:
            text.append("\nRun on: ", style="bold blue")
            text.append(", ".join(names) + ".\n", style="blue")
            text.append(
                "  See `gpu resources --domain compute` for setup notes.\n",
                style="dim",
            )

    # When-done preview (v0.10). Tells the user what `gpu done <id>` will
    # ask before they get there, so the prompts are not a surprise.
    if task.get("bottleneck_pick") or task.get("reality_check") or (task.get("commands") or task.get("bottleneck_pick")):
        text.append("\nWhen done:\n", style="bold cyan")
        if task.get("bottleneck_pick"):
            text.append(
                f"  gpu done {task['id']} will ask you to classify the bottleneck.\n",
                style="cyan",
            )
        if task.get("reality_check"):
            q = task["reality_check"]
            if len(q) > 80:
                q = q[:77].rstrip() + "..."
            text.append(
                f"  gpu done {task['id']} will ask: \"{q}\"\n",
                style="magenta",
            )
        if task.get("commands") or task.get("bottleneck_pick"):
            text.append(
                f"  Tip: attach a benchmark with `gpu done {task['id']} --bench <path>`.\n",
                style="green",
            )

    return text


def render_tracks(roadmap: Dict[str, Any], completed: List[str]) -> None:
    """Show per-track progress bars.

    Iterates ``roadmap['tracks']`` in declared order. For each track, counts
    tasks whose ``track`` field matches and how many of those are in
    ``completed``. Tasks with no ``track`` field are bucketed under a
    synthetic ``untracked`` group that is *not* shown.
    """
    tracks_meta = roadmap.get("tracks") or {}
    tasks = roadmap.get("tasks") or []
    if not tracks_meta:
        return

    name_w = max((len(meta.get("title", tid)) for tid, meta in tracks_meta.items()), default=10)
    console.print(Panel("Tracks", border_style="cyan"))
    for tid, meta in tracks_meta.items():
        title = meta.get("title", tid)
        track_tasks = [t for t in tasks if t.get("track") == tid]
        total = len(track_tasks)
        done = sum(1 for t in track_tasks if t["id"] in completed)
        pct = (done / total * 100) if total else 0
        filled = int(round(pct / 100 * 10))
        bar = "█" * filled + "░" * (10 - filled)
        console.print(f"  {title:<{name_w}} {bar} {done}/{total}  {pct:5.1f}%")


def render_walkthrough(roadmap: Dict[str, Any], storage: Dict[str, Any]) -> None:
    """First-run orientation: program, shape, commands, first task.

    Designed to be shown exactly once per install. Re-running ``gpu start``
    after the first run is intentionally quiet - the user already saw this.
    """
    # 1. Welcome panel
    console.print(Panel(
        f"[bold]{roadmap.get('program', '?')}[/bold]\n\n{roadmap.get('goal', '')}",
        title="Welcome",
        border_style="green",
    ))

    # 2. What this is
    console.print(Panel(
        "This is not a checklist. It is a small simulator for how an AI "
        "Systems Engineer thinks:\n\n"
        "  1. What is the workload?\n"
        "  2. What hardware is it running on?\n"
        "  3. What is the bottleneck?\n"
        "  4. How do I measure it?\n"
        "  5. What optimization should I try?\n"
        "  6. Did the benchmark prove improvement?\n\n"
        "Every task records progress, bumps skill bars, and (for some) "
        "asks you to classify the bottleneck and write a short reality check.",
        title="What this is",
        border_style="cyan",
    ))

    # 3. Shape of the program
    tracks_meta = roadmap.get("tracks") or {}
    milestones = roadmap.get("milestones") or {}
    tasks = roadmap.get("tasks") or []
    total_tasks = len(tasks)
    track_lines = []
    for tid, meta in tracks_meta.items():
        title = meta.get("title", tid)
        n = sum(1 for t in tasks if t.get("track") == tid)
        track_lines.append(f"  {title:<22} {n} task(s)")
    milestone_lines = [
        f"  {m.get('title', mid):<30} {len(m.get('tasks') or [])} task(s)"
        for mid, m in milestones.items()
    ]
    shape_body = (
        f"[bold]Tasks:[/bold]  {total_tasks} total across {len(tracks_meta)} tracks\n\n"
        f"[bold]Tracks:[/bold]\n" + "\n".join(track_lines) + "\n\n"
        f"[bold]Milestones:[/bold]\n" + "\n".join(milestone_lines)
    )
    console.print(Panel(shape_body, title="Shape of the program", border_style="magenta"))

    # 4. Commands you will use
    cmds = Table(show_header=False, box=None, padding=(0, 2))
    cmds.add_column(style="cyan", no_wrap=True)
    cmds.add_column(style="white")
    cmds.add_row("gpu start",   "begin the program, show the first task (this view)")
    cmds.add_row("gpu next",    "show the next pending task")
    cmds.add_row("gpu status",  "progress bar, current task, tracks, skill tree")
    cmds.add_row("gpu skills",  "skill tree only")
    cmds.add_row("gpu done <id> [--bench <path>]", "mark a task complete (prompts for bottleneck / reality when needed; --bench attaches a benchmark artifact)")
    cmds.add_row("gpu check",   "run a pending reality check on demand")
    cmds.add_row("gpu explain", "show recorded bottleneck + reality-check + benchmark log")
    cmds.add_row("gpu explain --id <id>", "focused per-task view")
    cmds.add_row("gpu tasks",   "list all task ids with completion marks")
    cmds.add_row("gpu resources", "list papers / libraries / tools (filter --domain / --tag / --difficulty, --open <id>)")
    cmds.add_row("gpu score",   "weighted program + per-track + per-milestone score")
    cmds.add_row("gpu reset",   "wipe all progress (asks first)")
    console.print(Panel(cmds, title="Commands you will use", border_style="blue"))

    # 4b. Next milestone (v0.11). One line, derived from roadmap + storage.
    console.print(
        f"[bold magenta]{_next_milestone_line(roadmap, storage.get('completed') or [])}[/bold magenta]"
    )

    # 5. First task
    first = next_task(tasks, storage.get("completed") or [])
    if first is not None:
        console.print(Panel(
            "Your first task is below. The 'Commands' lines are what you "
            "actually run in your terminal; the 'Deliverable' is what you "
            "produce. When you are done, run\n\n"
            "  [bold]gpu done " + first["id"] + "[/bold]\n\n"
            "(add [bold]--bench <path>[/bold] to attach a benchmark artifact).",
            title="Your first task",
            border_style="yellow",
        ))
        # 'On Mac?' hint (v0.12, expanded v0.14, v0.15). Fires when the
        # next task is in a GPU-heavy track AND the milestone is past
        # Week 1 (Triton, LLM, systems, or distillation - all of which
        # need a Linux NVIDIA box for the full version). Suppressed for
        # Week 1's commands-bearing tasks, which the user can attempt
        # on a Mac and fall back to notes. v0.14 also points at
        # `gpu resources --domain compute`, the per-task canonical
        # command.
        if (first.get("track") in {"cuda", "systems", "llm", "distillation"}
                and first.get("milestone") != "gpu_reality_check_week1"):
            console.print(
                "[dim]On Mac? Most Week 2+ tasks need a Linux NVIDIA GPU box. "
                "Type `gpu compute` for the 5 platforms, or `gpu resources --domain compute` "
                "for per-task platform guidance.[/dim]"
            )
        render_task(first, is_current=True, roadmap=roadmap)
    else:
        console.print("[green]All tasks complete. Run `gpu score` to see the result.[/green]")


def render_skill_tree(skills_meta: List[Dict[str, str]], skill_vals: Dict[str, int]) -> None:
    console.print(Panel("AI Systems Engineer Skill Tree", border_style="magenta"))
    for s in skills_meta:
        sid = s["id"]
        name = s["name"]
        val = skill_vals.get(sid, 0)
        val = max(0, min(100, val))
        filled = int(round(val / 100 * 10))
        bar = "█" * filled + "░" * (10 - filled)
        console.print(f"  {name:<22} {bar} {val:3d}%")


def _count_pending_prompts(roadmap: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, int]:
    """Count reality checks, bottlenecks, and bench opportunities still open."""
    tasks = roadmap.get("tasks") or []
    completed = set(data.get("completed") or [])
    reality_pending = sum(
        1 for t in tasks
        if t.get("reality_check") and t["id"] not in (data.get("reality") or {})
    )
    bottleneck_pending = sum(
        1 for t in tasks
        if t.get("bottleneck_pick") and t["id"] not in (data.get("bottlenecks") or {})
    )
    bench_attachable = sum(
        1 for t in tasks
        if (t.get("commands") or t.get("bottleneck_pick"))
        and t["id"] in completed
        and t["id"] not in (data.get("benchmarks") or {})
    )
    return {
        "reality": reality_pending,
        "bottleneck": bottleneck_pending,
        "bench": bench_attachable,
    }


def _render_pending(roadmap: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Append a small 'Pending' footer to status. v0.10."""
    counts = _count_pending_prompts(roadmap, data)
    body = (
        f"Reality checks to answer:           {counts['reality']}\n"
        f"Bottleneck classifications pending: {counts['bottleneck']}\n"
        f"Benchmarks to attach:               {counts['bench']}"
    )
    console.print(Panel(body, title="Pending", border_style="yellow"))


def _next_hint(roadmap: Dict[str, Any], data: Dict[str, Any]) -> str:
    """Compute a single next-action line. v0.10.

    Priority (highest first):
      1. Pending reality check via `gpu check`
      2. Pending bottleneck classification on the next task
      3. Bench artifact unviewed (recorded but not seen)
      4. Next pending task via `gpu done <id>`
      5. All done -> `gpu score`
    """
    tasks = roadmap.get("tasks") or []
    completed = set(data.get("completed") or [])

    # 1. Pending reality checks (any task with reality_check not yet answered)
    for t in tasks:
        if t.get("reality_check") and t["id"] not in (data.get("reality") or {}):
            return f"`gpu check` (a reality check is pending for {t['id']})"

    # 2. Next task that needs a bottleneck classification
    for t in tasks:
        if t["id"] in completed:
            continue
        if t.get("bottleneck_pick") and t["id"] not in (data.get("bottlenecks") or {}):
            return f"`gpu done {t['id']}` (will ask for a bottleneck classification)"

    # 3. Bench artifact recorded but never viewed
    for t in tasks:
        if t["id"] in (data.get("benchmarks") or {}):
            return f"`gpu explain --id {t['id']}` (you have a recorded benchmark to review)"

    # 4. Next pending task
    for t in tasks:
        if t["id"] not in completed:
            return f"`gpu done {t['id']}`"

    # 5. All done
    return "`gpu score` to see the final result"


def _format_relative(then: datetime, now: datetime) -> str:
    """Render a human 'Xh ago' / 'Xd ago' / 'just now' string."""
    delta = now - then
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _last_activity_line(data: Dict[str, Any]) -> str:
    """Single-line 'Last activity: ...' for the status panel. v0.11."""
    completed_at = data.get("completed_at") or {}
    if not completed_at:
        return "Last activity: never"
    # Find the most recent entry by timestamp; tolerate malformed values.
    best_id = None
    best_dt = None
    for tid, ts in completed_at.items():
        try:
            dt = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best_id = tid
    if best_dt is None:
        return "Last activity: unknown"
    return f"Last activity: {_format_relative(best_dt, datetime.now(timezone.utc))} ({best_id})"


def _next_milestone_line(roadmap: Dict[str, Any], completed: List[str]) -> str:
    """Single-line 'Next milestone: ...' for the walkthrough and status. v0.11."""
    completed_set = set(completed or [])
    milestones = roadmap.get("milestones") or {}
    for mid, m in milestones.items():
        ids = m.get("tasks") or []
        if not ids:
            continue
        done = sum(1 for tid in ids if tid in completed_set)
        if done < len(ids):
            title = m.get("title", mid)
            return f"Next milestone: {title} ({done}/{len(ids)} tasks)"
    return "All milestones complete."



def _end_of_curriculum_panel(roadmap: Dict[str, Any], completed: List[str]) -> Panel:
    """Panel shown by `done` and `start`/`next` when no successor task exists. v0.13.

    - Mid-curriculum boundary (a milestone just finished, more remain): the
      panel reads "Milestone boundary" and points to the next milestone via
      the same `_next_milestone_line` helper that `status` and the walkthrough
      already use. Border: cyan (transitional).
    - Fully done (all milestones complete): the panel reads "Curriculum
      complete" and shows a short end-of-program summary. Border: green.
    """
    nxt = _next_milestone_line(roadmap, completed or [])
    if nxt == "All milestones complete.":
        n_tasks = len(roadmap.get("tasks") or [])
        n_milestones = len(roadmap.get("milestones") or {})
        body = (
            f"All {n_tasks} tasks complete across the {n_milestones} milestones.\n"
            "Run `gpu score` for the program total or `gpu resources` for the next "
            "learning path."
        )
        return Panel(body, title="Curriculum complete", border_style="green")
    last_id = (completed or [None])[-1]
    if last_id is not None:
        last_task = task_by_id(roadmap, last_id)
        last_title = (last_task or {}).get("title", last_id) if last_task else last_id
        body = (
            f"Last task in this milestone: {last_title}\n"
            f"{nxt}\n"
            "Run `gpu next` to continue."
        )
    else:
        body = (
            f"{nxt}\n"
            "Run `gpu next` to continue."
        )
    return Panel(body, title="Milestone boundary", border_style="cyan")


# ----------------------------------------------------------------------------
# Bottleneck prompt
# ----------------------------------------------------------------------------

def prompt_bottleneck(task: Dict[str, Any]) -> str:
    """Prompt the user to classify the bottleneck for a task.

    Returns the chosen id. Raises typer.Exit on invalid input / skip.
    """
    console.print(Panel(
        "Bottleneck Engine\n\n"
        "What was the main bottleneck for this task?",
        title="Bottleneck Engine",
        border_style="yellow",
    ))
    for i, (bid, desc) in enumerate(BOTTLENECK_CHOICES, 1):
        console.print(f"  [{i}] {bid} - {desc}")
    raw = typer.prompt("Enter bottleneck id (or 'skip')").strip().lower()
    if raw in ("skip", "s", ""):
        return ""
    # accept either name or number
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(BOTTLENECK_CHOICES):
            return BOTTLENECK_CHOICES[idx][0]
    if raw in BOTTLENECK_IDS:
        return raw
    console.print("[red]Invalid bottleneck. Use id or number from the list.[/red]")
    raise typer.Exit(1)


def prompt_reality(task: Dict[str, Any]) -> str:
    console.print(Panel(
        f"Reality Check\n\n{task['reality_check']}",
        title="Reality Check",
        border_style="magenta",
    ))
    console.print(
        "[dim]A good answer: ~2 sentences, names the bottleneck or principle, "
        "gives a number when you can.[/dim]",
    )
    console.print(
        "[dim]Example: 'Vector add is memory-bound because each thread reads "
        "2 + writes 1 element, but only does 1 FLOP.'[/dim]",
    )
    console.print("(type 'skip' to leave blank)")
    ans = typer.prompt("Your answer")
    return ans.strip()


# ----------------------------------------------------------------------------
# Teaching prompts (v0.16)
# ----------------------------------------------------------------------------

def _pattern_in(pattern: str, answer_lower: str) -> bool:
    """Case-insensitive regex search. Returns True on first match.

    Compiles each pattern fresh; the call sites pass small lists (1-3
    entries) so this is fine. If the pattern is malformed, returns
    False and lets the caller fall through to the next check.
    """
    try:
        return _re.search(pattern, answer_lower, _re.IGNORECASE) is not None
    except _re.error:
        return False


def give_teaching_feedback(answer: str, prompt: Dict[str, Any]) -> str:
    """Hand-coded feedback for a teaching answer. v0.16 + v0.18.

    v0.18 layered feedback: the more specific the rule that fires, the
    better the feedback. Order is:

      1. common_misconceptions (regex match) -> specific "not quite" line.
         This fires BEFORE the keyword check because a common wrong answer
         is more useful to call out than a generic miss.
      2. expected_answers (regex match, in order) -> first matching response.
         A specific "right - the bottleneck is X" line.
      3. expected_keywords (substring match, count >= 2) -> follow_up_if_match.
         The v0.16 fallback. Kept as-is for backward compatibility.
      4. No match -> follow_up_if_miss, or a "Consider: <keywords>" line
         built from the expected_keywords list.

    The hand-coded fallback is what `gpu teach` always uses; the LLM
    path (`--llm`) is additive, see ask_llm_feedback.
    """
    ans = (answer or "").lower()

    # 1. Common misconceptions (specific "wrong answer" feedback).
    for entry in (prompt.get("common_misconceptions") or []):
        if isinstance(entry, dict) and _pattern_in(entry.get("pattern", ""), ans):
            return entry.get("response") or "Not quite."

    # 2. Expected answers (specific "right answer" feedback, in order).
    for entry in (prompt.get("expected_answers") or []):
        if isinstance(entry, dict) and _pattern_in(entry.get("pattern", ""), ans):
            return entry.get("response") or "Good."

    # 3. Keyword count (v0.16 fallback).
    kws = [str(k).lower() for k in (prompt.get("expected_keywords") or [])]
    hits = sum(1 for k in kws if k and k in ans)
    if hits >= 2:
        return prompt.get("follow_up_if_match") or "Good."

    # 4. Miss fallback.
    return prompt.get("follow_up_if_miss") or "Consider: " + ", ".join(kws[:3]) + "."


# `re` is the stdlib module; aliased as _re at the top of gpu.py to avoid
# shadowing by any future import. The helper above uses _re explicitly.


# v0.20: model allowlist for the optional --llm feedback path. Users pick
# via the GPU_LLM_MODEL env var; default is gpt-4o-mini (cheapest of the
# known-good set). The allowlist keeps typos from silently burning credits
# on a non-existent model.
LLM_ALLOWED_MODELS = {
    "gpt-4o-mini":  "cheapest, default",
    "gpt-4o":       "higher quality, ~15x cost",
    "gpt-4.1-mini": "newest mini, good middle ground",
    "gpt-4.1":      "newest full, most expensive",
}


def _llm_cache_key(task_id: str, prompt_index: int, answer: str) -> str:
    """Stable cache key. `answer` is normalized (lowercased, stripped)."""
    norm = (answer or "").strip().lower()
    return f"{task_id}|{prompt_index}|{norm}"


def ask_llm_feedback(
    answer: str,
    question: str,
    api_key: str,
    task_context: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    stream_cb: Optional[Any] = None,
    cache: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    prompt_index: Optional[int] = None,
) -> Optional[str]:
    """Optional LLM-generated feedback for a teaching answer. v0.17 + v0.18.

    Calls the OpenAI Chat Completions API (gpt-4o-mini by default) over
    HTTPS using stdlib only (no `openai` SDK, no subprocess). Returns
    the assistant's text reply, or None on any error. The caller should
    treat None as 'LLM unavailable' and fall back to the hand-coded
    feedback; this function never raises.

    v0.18: `task_context` is an optional dict with keys "objective" and
    "deliverable" (and optionally "title"). When supplied, the system
    prompt is enriched with the task context so the LLM feedback is
    anchored to the specific task instead of being generic. The system
    prompt cap keeps the total under ~600 tokens to stay cheap.

    Cost guard: max 200 tokens per call. Time guard: 8s timeout.

    v0.20 additions:
      - `model` selects the OpenAI model; falls back to GPU_LLM_MODEL
        env var, then "gpt-4o-mini". Unknown models fall back to the
        default silently (avoids burning credits on a typo).
      - `stream_cb(token)` is called for each streamed delta when the
        API is called in streaming mode. The assembled text is still
        returned. If `stream_cb` is None, the request is non-streaming
        (matches v0.17/v0.18 behavior).
      - `cache` is an optional dict-like (storage.llm_cache). If a key
        is hit, the cached text is returned without an API call. The
        env var GPU_LLM_NO_CACHE=1 disables lookup + store.
      - `task_id` + `prompt_index` identify the cache key when `cache`
        is supplied. Both are required to read or write the cache.
    """
    if not api_key:
        return None

    # --- v0.20: resolve model from arg > env > default, validated against allowlist.
    chosen = model or os.environ.get("GPU_LLM_MODEL", "") or "gpt-4o-mini"
    if chosen not in LLM_ALLOWED_MODELS:
        chosen = "gpt-4o-mini"

    # --- v0.20: cache lookup (unless bypassed).
    use_cache = (
        cache is not None
        and task_id is not None
        and prompt_index is not None
        and os.environ.get("GPU_LLM_NO_CACHE", "") != "1"
    )
    cache_key = None
    if use_cache:
        cache_key = _llm_cache_key(task_id, prompt_index, answer)
        hit = cache.get(cache_key)
        if hit and hit.get("text"):
            return hit["text"]

    try:
        import json as _json
        import urllib.request
        import urllib.error
        system = (
            "You are a brief, encouraging tutor for a GPU systems engineering learner. "
            "Given the question and the user's answer, give 1-2 sentences of feedback. "
            "If the answer is right, affirm it. If it misses, name the key idea they should "
            "add. No preamble, no bullet points, no markdown."
        )
        if task_context:
            ctx_parts = []
            if task_context.get("title"):
                ctx_parts.append(f"Task: {task_context['title']}")
            if task_context.get("objective"):
                ctx_parts.append(f"Objective: {task_context['objective']}")
            if task_context.get("deliverable"):
                ctx_parts.append(f"Deliverable: {task_context['deliverable']}")
            if ctx_parts:
                system = system + " " + " | ".join(ctx_parts)
        body = _json.dumps({
            "model": chosen,
            "max_tokens": 200,
            "stream": bool(stream_cb),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        if stream_cb:
            text_parts: list = []
            with urllib.request.urlopen(req, timeout=8) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip()
                    if not line.startswith("data:"):
                        continue
                    payload_s = line[len("data:"):].strip()
                    if payload_s == "[DONE]":
                        break
                    try:
                        evt = _json.loads(payload_s)
                    except ValueError:
                        continue
                    for ch in evt.get("choices") or []:
                        delta = (ch.get("delta") or {}).get("content")
                        if delta:
                            text_parts.append(delta)
                            try:
                                stream_cb(delta)
                            except Exception:
                                pass
            assembled = "".join(text_parts).strip()
            if not assembled:
                return None
            if use_cache and cache_key is not None:
                cache[cache_key] = {
                    "model": chosen,
                    "text": assembled,
                    "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            return assembled
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            return None
        msg = (choices[0].get("message") or {}).get("content") or ""
        msg = msg.strip() or None
        if msg and use_cache and cache_key is not None:
            cache[cache_key] = {
                "model": chosen,
                "text": msg,
                "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        return msg
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, OSError):
        return None


def prompt_teaching(
    task: Dict[str, Any],
    data: Dict[str, Any],
    task_id: str,
    index: int = 0,
    use_llm: bool = False,
    stream_cb: Optional[Any] = None,
) -> None:
    """Run ONE step of the multi-step teaching flow for a task. v0.16 + v0.17.

    Renders `task["teaching_prompts"][index]`, reads a free-text answer
    (the user can type 'skip' to leave blank), computes feedback via
    `give_teaching_feedback`, and appends the record to
    `data["teaching"][task_id]`. The caller (the `done` flow or the
    `gpu teach` command) is responsible for looping and passing the
    correct `index`.

    If `use_llm` is True and the OPENAI_API_KEY env var is set, also
    calls `ask_llm_feedback` and stores the LLM response as
    `llm_feedback` on the record. The hand-coded feedback is always
    stored alongside (as `feedback`); the LLM feedback is additive.
    Falls back silently to None if the LLM call fails.

    Quiet no-op if `index` is out of range for the task's
    `teaching_prompts` list.
    """
    prompts = task.get("teaching_prompts") or []
    if index < 0 or index >= len(prompts):
        return
    p = prompts[index]
    console.print(Panel(
        f"Teaching Prompt ({index + 1}/{len(prompts)})\n\n{p['question']}",
        title="Teaching",
        border_style="blue",
    ))
    try:
        ans = typer.prompt("Your answer (or 'skip')").strip()
    except (typer.Abort, EOFError, KeyboardInterrupt):
        ans = "skip"
    feedback = "" if ans.lower() == "skip" else give_teaching_feedback(ans, p)
    # v0.20: print hand-coded feedback BEFORE the LLM call. The LLM
    # call (especially with streaming) takes a moment, and the user
    # should see the deterministic feedback first.
    if feedback:
        console.print(f"[blue]Feedback:[/blue] {feedback}")
    llm_feedback = None
    if use_llm and ans.lower() != "skip":
        task_context = None
        if isinstance(task, dict):
            task_context = {
                k: task.get(k) for k in ("title", "objective", "deliverable")
                if task.get(k)
            }
        llm_feedback = ask_llm_feedback(
            ans,
            p["question"],
            os.environ.get("OPENAI_API_KEY", ""),
            task_context=task_context,
            stream_cb=stream_cb,
            cache=data.get("llm_cache") if isinstance(data, dict) else None,
            task_id=task_id,
            prompt_index=index,
        )
    record = {
        "question": p["question"],
        "answer": ans,
        "feedback": feedback,
        "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if llm_feedback is not None:
        record["llm_feedback"] = llm_feedback
    data.setdefault("teaching", {}).setdefault(task_id, []).append(record)
    if llm_feedback:
        # v0.20: stream_cb path already printed tokens, skip duplication.
        if stream_cb is not None:
            return
        # v0.20: tag the LLM feedback line as cached / fresh / live.
        cache_dict = data.get("llm_cache") if isinstance(data, dict) else None
        cache_key = _llm_cache_key(task_id, index, ans) if cache_dict is not None else None
        if cache_key and cache_dict.get(cache_key, {}).get("text") == llm_feedback:
            tag = "[dim](cached)[/dim]"
        elif cache_key and cache_key in cache_dict and \
                cache_dict[cache_key].get("cached_at", "").startswith(
                    datetime.now(timezone.utc).isoformat(timespec="seconds")[:14]
                ):
            tag = "[dim](fresh)[/dim]"
        else:
            tag = "[dim](live)[/dim]"
        console.print(f"[magenta]LLM feedback:[/magenta] {llm_feedback} {tag}")


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

@app.command()
def start(
):
    """Begin the program and walk you through what is next.

    First run: shows the full walkthrough (welcome, what this is, shape of
    the program, command list, first task). Subsequent runs: stay quiet
    and just show the current task - use `gpu next` for that.
    """
    roadmap = load_roadmap()
    data = load_storage()

    if not data["started"]:
        data["started"] = True
        render_walkthrough(roadmap, data)
        data["welcomed"] = True
        save_storage(data)
        return

    # Already started.
    if not data.get("welcomed"):
        # started was set in an earlier version of the tool (no welcomed
        # key was ever written). Show the walkthrough once and remember.
        render_walkthrough(roadmap, data)
        data["welcomed"] = True
        save_storage(data)
        return

    # Re-runs are intentionally quiet. Just show the current task.
    render_progress(roadmap["tasks"], data["completed"])
    task = next_task(roadmap["tasks"], data["completed"])
    if task:
        render_task(task, roadmap=roadmap)
    else:
        console.print(_end_of_curriculum_panel(roadmap, data["completed"]))


@app.command(name="next")
def cmd_next(
):
    """Show the next pending task."""
    roadmap = load_roadmap()
    data = load_storage()
    render_progress(roadmap["tasks"], data["completed"])
    task = next_task(roadmap["tasks"], data["completed"])
    if task:
        render_task(task, roadmap=roadmap)
    else:
        console.print(_end_of_curriculum_panel(roadmap, data["completed"]))


@app.command()
def status(
):
    """Show progress, current task, skill tree, and what's pending."""
    roadmap = load_roadmap()
    data = load_storage()
    render_progress(roadmap["tasks"], data["completed"])
    task = next_task(roadmap["tasks"], data["completed"])
    if task:
        render_task(task, roadmap=roadmap)
    else:
        console.print("[green]All tasks complete.[/green]")
    render_tracks(roadmap, data["completed"])
    render_skill_tree(roadmap.get("skills", []), data.get("skills", {}))
    _render_pending(roadmap, data)
    console.print(f"[bold cyan]Next:[/bold cyan] {_next_hint(roadmap, data)}")
    console.print(f"[dim]{_last_activity_line(data)}[/dim]")
    console.print(f"[bold magenta]{_next_milestone_line(roadmap, data['completed'])}[/bold magenta]")


@app.command()
def skills(
):
    """Show the skill tree only."""
    roadmap = load_roadmap()
    data = load_storage()
    render_skill_tree(roadmap.get("skills", []), data.get("skills", {}))


@app.command()
def done(
    task_id: str = typer.Argument(..., help="Task id to mark complete"),
    bench: Optional[str] = typer.Option(
        None,
        "--bench",
        help="Path to a benchmark artifact for this task (e.g. notes/hardware.md).",
    ),
):
    """Mark a task complete (may prompt for bottleneck / reality check)."""
    roadmap = load_roadmap()
    data = load_storage()

    task = task_by_id(roadmap, task_id)
    if task is None:
        console.print(f"[red]Unknown task id: {task_id}[/red]")
        console.print("Valid ids:")
        for t in roadmap["tasks"]:
            console.print(f"  - {t['id']}")
        typer.echo(
            "Tip: run `gpu tasks` for a numbered view, or `gpu status` for the current task.",
            err=True,
        )
        raise typer.Exit(1)

    if task_id in data["completed"]:
        console.print(f"[yellow]Task {task_id} is already complete.[/yellow]")
        typer.echo(
            f"Tip: `gpu explain --id {task_id}` to see the recorded "
            f"bottleneck / reality / benchmark.",
            err=True,
        )
        return

    # Re-run detection (v0.10). The user is starting `gpu done` for a task
    # they have not completed yet, but already have a recorded prompt for
    # (likely from `gpu check` or a previous interrupted session). Surface
    # the data-state so the prompts are not a surprise.
    if task.get("reality_check") and task_id in data.get("reality", {}):
        typer.echo(
            f"Tip: a reality check for {task_id} is already recorded; "
            f"`gpu explain --id {task_id}` to view it.",
            err=True,
        )
    if task.get("bottleneck_pick") and task_id in data.get("bottlenecks", {}):
        typer.echo(
            f"Tip: a bottleneck for {task_id} is already recorded as "
            f"{data['bottlenecks'][task_id]}; "
            f"`gpu explain --id {task_id}` to view it.",
            err=True,
        )

    # Bottleneck engine
    if task.get("bottleneck_pick"):
        bid = prompt_bottleneck(task)
        if bid:
            data["bottlenecks"][task_id] = bid
    elif task.get("bottleneck"):
        # pre-classified: just record it
        data["bottlenecks"][task_id] = task["bottleneck"]

    # Reality check
    if task.get("reality_check"):
        ans = prompt_reality(task)
        if ans and ans.lower() != "skip":
            data["reality"][task_id] = ans

    data["completed"].append(task_id)
    data["completed_at"][task_id] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for skill, points in (task.get("skills") or {}).items():
        current = data["skills"].get(skill, 0)
        data["skills"][skill] = max(0, min(100, current + points))

    # Benchmark capture (v0.8). Light path validation only: expand ~,
    # resolve relative paths against the repo root, warn to stderr if the
    # resolved path does not exist, but never block. Never copy / hash / read
    # the file. (See expand_bench_path docstring.)
    if bench:
        resolved = expand_bench_path(bench, ROOT)
        if not os.path.exists(resolved):
            typer.echo(
                f"warning: benchmark file not found: {resolved} - recording anyway",
                err=True,
            )
        try:
            summary = typer.prompt("One-line benchmark summary", default="")
        except (typer.Abort, EOFError, KeyboardInterrupt):
            summary = ""
        data["benchmarks"][task_id] = {
            "path": resolved,
            "summary": summary.strip(),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    elif task.get("bottleneck_pick") or task.get("commands"):
        # Hint only - never blocks, never records.
        typer.echo(
            "Tip: attach a benchmark file with --bench <path> (e.g. "
            "`gpu done " + task_id + " --bench notes/hardware.md`).",
            err=True,
        )

    # Teaching flow (v0.16). Multi-step prompts with hand-coded feedback.
    # Fires AFTER the bench summary so the user is in a single, predictable
    # prompt cadence: bottleneck -> reality -> bench summary -> teaching.
    # The user can 'skip' at any step. Quiet if the task has no
    # teaching_prompts.
    teaching = task.get("teaching_prompts") or []
    if teaching:
        for i in range(len(teaching)):
            already = len(data.get("teaching", {}).get(task_id, []))
            if already > i:
                # Already answered; skip without re-asking.
                continue
            prompt_teaching(task, data, task_id, i)
            recent = data.get("teaching", {}).get(task_id, [])
            if recent and recent[-1].get("answer", "").lower() == "skip":
                break

    save_storage(data)
    console.print(f"[green]Completed:[/green] {task['title']}")

    render_progress(roadmap["tasks"], data["completed"])

    nxt = next_task(roadmap["tasks"], data["completed"])
    if nxt:
        render_task(nxt, roadmap=roadmap)
    else:
        console.print(_end_of_curriculum_panel(roadmap, data["completed"]))


@app.command()
def reset(
):
    """Reset all progress."""
    if not typer.confirm("Are you sure you want to reset progress?"):
        console.print("Reset cancelled.")
        return
    roadmap = load_roadmap()
    data = default_storage(roadmap)
    save_storage(data)
    console.print("[green]Progress reset.[/green]")


@app.command()
def explain(
    task_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Show bottleneck + reality + benchmark for a single task.",
    ),
):
    """Show bottleneck / reality / benchmark answers recorded so far.

    With no arguments, prints global logs (bottlenecks, reality checks,
    benchmarks). With ``--id <task_id>``, prints a focused 3-block view
    for that task: BENCHMARK is only shown when a record exists (Q5 -
    absent benchmarks are quiet).
    """
    data = load_storage()

    # Per-task view
    if task_id:
        _render_explain_task(task_id, data)
        return

    # Global view
    has_any = (
        data.get("bottlenecks")
        or data.get("reality")
        or data.get("benchmarks")
    )
    if not has_any:
        console.print("[yellow]No bottleneck / reality / benchmark records yet.[/yellow]")
        return

    if data.get("bottlenecks"):
        bt = Table(title="Bottleneck Log")
        bt.add_column("Task", style="cyan")
        bt.add_column("Bottleneck", style="yellow")
        for tid, bid in data["bottlenecks"].items():
            bt.add_row(tid, bid)
        console.print(bt)

    if data.get("reality"):
        rt = Table(title="Reality Check Log")
        rt.add_column("Task", style="cyan")
        rt.add_column("Answer", style="magenta", overflow="fold")
        for tid, ans in data["reality"].items():
            rt.add_row(tid, ans)
        console.print(rt)

    if data.get("benchmarks"):
        bk = Table(title="Benchmark Log")
        bk.add_column("Task", style="cyan")
        bk.add_column("Path", style="green", overflow="fold")
        bk.add_column("Summary", style="white", overflow="fold")
        bk.add_column("Recorded", style="dim")
        for tid, rec in data["benchmarks"].items():
            bk.add_row(
                tid,
                rec.get("path", ""),
                rec.get("summary", ""),
                rec.get("recorded_at", ""),
            )
        console.print(bk)

    if data.get("teaching"):
        tk = Table(title="Teaching Log")
        tk.add_column("Task", style="cyan")
        tk.add_column("Prompts", style="dim", justify="right")
        tk.add_column("Last asked", style="dim")
        for tid, log in data["teaching"].items():
            last = log[-1].get("asked_at", "") if log else ""
            tk.add_row(tid, str(len(log)), last)
        console.print(tk)


def _render_explain_task(task_id: str, data: Dict[str, Any]) -> None:
    """Render the focused per-task view used by ``gpu explain --id``."""
    bid = data.get("bottlenecks", {}).get(task_id)
    ans = data.get("reality", {}).get(task_id)
    rec = data.get("benchmarks", {}).get(task_id)

    console.print(Panel(f"Task: [bold]{task_id}[/bold]", border_style="cyan"))

    if bid:
        console.print(Panel(f"bottleneck = {bid}", title="BOTTLENECK", border_style="yellow"))
    if ans:
        console.print(Panel(ans, title="REALITY", border_style="magenta"))
    if rec:
        body = (
            f"path:      {rec.get('path', '')}\n"
            f"summary:   {rec.get('summary', '')}\n"
            f"recorded:  {rec.get('recorded_at', '')}"
        )
        console.print(Panel(body, title="BENCHMARK", border_style="green"))

    # Teaching log (v0.16 + v0.17). Q5-style: quiet when absent.
    teaching_log = data.get("teaching", {}).get(task_id) or []
    if teaching_log:
        from rich.table import Table as _TT
        has_llm = any("llm_feedback" in e for e in teaching_log)
        tt = _TT(title="TEACHING", show_header=True, header_style="bold blue")
        tt.add_column("#", style="dim", width=3)
        tt.add_column("Question", style="cyan", overflow="fold")
        tt.add_column("Answer", style="white", overflow="fold")
        tt.add_column("Feedback", style="blue", overflow="fold")
        if has_llm:
            tt.add_column("LLM", style="magenta", overflow="fold")
        for i, entry in enumerate(teaching_log, 1):
            row = [
                str(i),
                entry.get("question", ""),
                entry.get("answer", "") or "[dim](skipped)[/dim]",
                entry.get("feedback", "") or "[dim](no feedback)[/dim]",
            ]
            if has_llm:
                row.append(entry.get("llm_feedback", "") or "[dim](none)[/dim]")
            tt.add_row(*row)
        console.print(tt)

    if not (bid or ans or rec or teaching_log):
        console.print("[yellow]No records for this task.[/yellow]")


@app.command()
def check(
):
    """Run a random / current-task reality check on demand."""
    roadmap = load_roadmap()
    data = load_storage()
    # find a task that has a reality_check and is not yet answered
    pending = [t for t in roadmap["tasks"]
               if t.get("reality_check") and task_by_id(roadmap, t["id"])["id"] not in data["reality"]]
    if not pending:
        console.print("[green]No outstanding reality checks.[/green]")
        return
    task = pending[0]
    console.print(Panel(
        f"REALITY CHECK\n\n{task['reality_check']}",
        title=task["title"],
        border_style="magenta",
    ))
    try:
        ans = typer.prompt("Your answer (or 'skip')")
    except (typer.Abort, EOFError, KeyboardInterrupt):
        console.print("[yellow]Skipped.[/yellow]")
        raise typer.Exit(0)
    if ans.strip() and ans.strip().lower() != "skip":
        data["reality"][task["id"]] = ans.strip()
        save_storage(data)
        console.print("[green]Stored.[/green]")


@app.command()
def teach(
    task_id: str = typer.Argument(..., help="Task id to run the teaching flow on"),
    llm: bool = typer.Option(
        False,
        "--llm",
        help="Also generate LLM feedback (requires OPENAI_API_KEY env var; v0.17).",
    ),
):
    """Run the interactive teaching flow for a task (v0.16 + v0.17).

    Walks the task's `teaching_prompts` list, with hand-coded feedback at
    each step. The user can type 'skip' to leave an answer blank. Existing
    answers (from a prior `gpu done` or `gpu teach` run) are not re-asked.
    The teaching log is stored in `storage.json` under the `teaching` key
    and is visible via `gpu explain --id <task_id>`.

    Pass --llm to also generate an LLM feedback line per answer (v0.17).
    Requires the OPENAI_API_KEY env var. If the env var is missing or the
    API call fails, the hand-coded feedback still records and the command
    exits 0; the LLM path is additive, never blocking.

    v0.20 additions:
      - Set GPU_LLM_MODEL to pick from a 4-model allowlist
        (gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-4.1). Default gpt-4o-mini.
        Unknown values fall back to the default silently.
      - Responses are cached in storage.llm_cache. Repeated calls with
        the same (task_id, prompt_index, answer) do not re-bill OpenAI.
        Set GPU_LLM_NO_CACHE=1 to bypass.
      - Fresh LLM responses are streamed token-by-token; cache hits
        print a single line with a (cached) / (fresh) / (live) tag.

    Tasks without a `teaching_prompts` field are a no-op with a clear message.
    """
    roadmap = load_roadmap()
    data = load_storage()
    task = task_by_id(roadmap, task_id)
    if task is None:
        console.print(f"[red]Unknown task id: {task_id}[/red]")
        console.print("Valid ids:")
        for t in roadmap["tasks"]:
            console.print(f"  - {t['id']}")
        raise typer.Exit(1)
    teaching = task.get("teaching_prompts") or []
    if not teaching:
        console.print(f"[yellow]No teaching prompts for {task_id}.[/yellow]")
        console.print("Teaching is opt-in per task. Add a 'teaching_prompts' field")
        console.print("to the task in roadmap.json to enable it.")
        return
    answered = len(data.get("teaching", {}).get(task_id, []))
    has_llm = any("llm_feedback" in e for e in data.get("teaching", {}).get(task_id, []))
    if answered >= len(teaching) and not (llm and not has_llm):
        console.print(f"[green]Teaching for {task_id} already complete ({answered}/{len(teaching)} prompts answered).[/green]")
        console.print(f"Run `gpu explain --id {task_id}` to view the recorded log.")
        return
    if answered >= len(teaching) and llm and not has_llm:
        # The user is asking to add LLM feedback to the existing log. Reset
        # the "already answered" state so the loop re-runs and adds new
        # records with llm_feedback set. We keep the existing entries in
        # storage.json by NOT deleting them; the loop will append new ones.
        # But the loop's `already > i` gate would skip re-asking. The
        # simplest fix: clear the entry under data['teaching'][task_id] so
        # the loop re-prompts. Existing entries are removed - we want a
        # clean re-run with LLM feedback attached.
        if task_id in data.get("teaching", {}):
            del data["teaching"][task_id]
        console.print(
            f"[blue]--llm: re-running {len(teaching)} prompts to attach LLM feedback.[/blue]"
        )
    if llm and not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[yellow]--llm requested but OPENAI_API_KEY is not set.[/yellow]\n"
            "Falling back to hand-coded feedback only. To enable LLM feedback:\n"
            "  export OPENAI_API_KEY=sk-...   (then re-run)"
        )
    for i in range(len(teaching)):
        already = len(data.get("teaching", {}).get(task_id, []))
        if already > i:
            continue
        # v0.20: stream tokens when --llm is on AND the cache will miss.
        # If the cache will hit, we let prompt_teaching print the (cached)
        # tag line via the standard post-call path (no streaming).
        stream_cb = None
        if llm and os.environ.get("OPENAI_API_KEY"):
            cache_dict = data.get("llm_cache", {}) if isinstance(data, dict) else {}
            # We don't know the user's answer yet (prompt_teaching asks),
            # so we can't pre-compute the cache key. Default to streaming
            # for fresh answers; prompt_teaching will short-circuit on
            # cache hit and skip the post-call line if stream_cb is set.
            # To still show a (cached) tag for hits, the teach command
            # checks the latest record after the call and prints the
            # cached line if no streaming happened.
            streamed = {"on": False}
            def _stream_cb(token: str, _s=streamed) -> None:
                if not _s["on"]:
                    console.print("[magenta]LLM feedback:[/magenta] ", end="")
                    _s["on"] = True
                console.print(token, end="", highlight=False)
            stream_cb = _stream_cb
        if stream_cb is not None:
            console.print()  # spacer before the streaming line
        prompt_teaching(task, data, task_id, i, use_llm=llm, stream_cb=stream_cb)
        # If streaming didn't fire (cache hit), print the cached line.
        if stream_cb is not None and not streamed["on"]:
            recent = data.get("teaching", {}).get(task_id, [])
            if recent and recent[-1].get("llm_feedback"):
                cache_dict2 = data.get("llm_cache", {})
                ans = recent[-1].get("answer", "")
                key = _llm_cache_key(task_id, i, ans)
                tag = "[dim](cached)[/dim]" if key in cache_dict2 else "[dim](live)[/dim]"
                console.print(f"[magenta]LLM feedback:[/magenta] {recent[-1]['llm_feedback']} {tag}")
        if stream_cb is not None and streamed["on"]:
            console.print()  # newline after streamed text
        recent = data.get("teaching", {}).get(task_id, [])
        if recent and recent[-1].get("answer", "").lower() == "skip":
            break
    save_storage(data)
    console.print(f"[green]Teaching log saved for {task_id}.[/green]")


@app.command()
def tasks(
):
    """List all task ids in order."""
    roadmap = load_roadmap()
    data = load_storage()
    for i, t in enumerate(roadmap["tasks"], 1):
        mark = "[green]x[/green]" if t["id"] in data["completed"] else "[ ]"
        console.print(f"  {mark} {i:>2}. {t['id']:<18}  {t['title']}")


# ----------------------------------------------------------------------------
# v0.8: resources + score commands
# ----------------------------------------------------------------------------

@app.command()
def resources(
    domain: Optional[str] = typer.Option(None, "--domain", help="Filter by domain (e.g. kernels)."),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag (e.g. cuda)."),
    difficulty: Optional[str] = typer.Option(
        None, "--difficulty", help="Filter by difficulty (beginner|intermediate|advanced)."
    ),
    open_id: Optional[str] = typer.Option(
        None, "--open", help="Open the URL for a resource id in the default browser."
    ),
):
    """List resources from the roadmap (papers, libraries, tools)."""
    roadmap = load_roadmap()
    items = list(roadmap.get("resources") or [])

    # --open is a side-effecting action; do it before filtering and exit.
    if open_id is not None:
        match = next((r for r in items if r.get("id") == open_id), None)
        if match is None:
            console.print(f"[red]Unknown resource id: {open_id}[/red]")
            raise typer.Exit(1)
        url = match.get("url", "")
        if not url:
            console.print(f"[red]Resource {open_id} has no url.[/red]")
            raise typer.Exit(1)
        console.print(f"Opening [cyan]{url}[/cyan] in your default browser...")
        webbrowser.open(url)
        return

    if domain is not None:
        items = [r for r in items if domain in (r.get("domains") or [])]
    if tag is not None:
        items = [r for r in items if tag in (r.get("tags") or [])]
    if difficulty is not None:
        items = [r for r in items if r.get("difficulty") == difficulty]

    if not items:
        console.print("[yellow]No resources match the current filters.[/yellow]")
        return

    table = Table(title="Resources", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Domains", style="green")
    table.add_column("Tags", style="magenta")
    table.add_column("Difficulty", style="yellow")
    table.add_column("URL", style="blue", overflow="fold")
    for r in items:
        table.add_row(
            r.get("id", ""),
            r.get("title", ""),
            ", ".join(r.get("domains") or []),
            ", ".join(r.get("tags") or []),
            r.get("difficulty", ""),
            r.get("url", ""),
        )
    console.print(table)


@app.command()
def compute(
    open_id: Optional[str] = typer.Option(
        None, "--open", help="Open the URL for a compute platform id in the default browser."
    ),
):
    """Show where to run GPU tasks (compute platforms for the curriculum)."""
    roadmap = load_roadmap()
    items = [r for r in (roadmap.get("resources") or [])
             if "compute" in (r.get("domains") or [])]

    if open_id is not None:
        match = next((r for r in items if r.get("id") == open_id), None)
        if match is None:
            console.print(f"[red]Unknown compute platform id: {open_id}[/red]")
            raise typer.Exit(1)
        url = match.get("url", "")
        if not url:
            console.print(f"[red]Compute platform {open_id} has no url.[/red]")
            raise typer.Exit(1)
        console.print(f"Opening [cyan]{url}[/cyan] in your default browser...")
        webbrowser.open(url)
        return

    if not items:
        console.print(
            "[yellow]No compute platforms declared in the roadmap. "
            "Add resources with domain=compute to see them here.[/yellow]"
        )
        return

    console.print(
        "[dim]On a Mac? Colab or Kaggle for the basics. "
        "For Triton / vLLM / serving, rent a Linux GPU box (RunPod or Lambda Labs).[/dim]"
    )
    table = Table(title="Where to run GPU tasks", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Good for", style="green")
    table.add_column("Not good for", style="red")
    table.add_column("URL", style="blue", overflow="fold")
    for r in items:
        table.add_row(
            r.get("id", ""),
            r.get("title", ""),
            r.get("good_for", ""),
            r.get("not_good_for", ""),
            r.get("url", ""),
        )
    console.print(table)


@app.command()
def score(
):
    """Show weighted program + per-track + per-milestone scores."""
    roadmap = load_roadmap()
    data = load_storage()
    completed = set(data.get("completed") or [])
    tasks = roadmap.get("tasks") or []

    # Program total
    total = _track_score(tasks, list(completed))
    possible = sum(_task_raw_score(t) for t in tasks)
    earned = sum(_task_raw_score(t) for t in tasks if t["id"] in completed)
    console.print(Panel(
        f"[bold]{total} / 100[/bold]  (raw: {earned} / {possible})",
        title="GPU Reality Score",
        border_style="cyan",
    ))

    # Per-track rollup
    tracks_meta = roadmap.get("tracks") or {}
    track_tasks = {tid: [t for t in tasks if t.get("track") == tid] for tid in tracks_meta}
    if tracks_meta:
        console.print(Panel("Score by track", border_style="cyan"))
        name_w = max((len(meta.get("title", tid)) for tid, meta in tracks_meta.items()), default=10)
        for tid, meta in tracks_meta.items():
            t_tasks = track_tasks.get(tid, [])
            t_score = _track_score(t_tasks, list(completed))
            t_done = sum(1 for t in t_tasks if t["id"] in completed)
            filled = int(round(t_score / 100 * 10))
            bar = "█" * filled + "░" * (10 - filled)
            console.print(
                f"  {meta.get('title', tid):<{name_w}} {bar} "
                f"{t_score:3d}/100  ({t_done}/{len(t_tasks)} tasks)"
            )

    # Per-milestone rollup
    milestones = roadmap.get("milestones") or {}
    if milestones:
        console.print(Panel("Score by milestone", border_style="cyan"))
        mname_w = max((len(m.get("title", mid)) for mid, m in milestones.items()), default=10)
        for mid, m in milestones.items():
            ids = m.get("tasks") or []
            m_tasks = [t for t in tasks if t["id"] in ids]
            m_score = _track_score(m_tasks, list(completed))
            m_done = sum(1 for t in m_tasks if t["id"] in completed)
            filled = int(round(m_score / 100 * 10))
            bar = "█" * filled + "░" * (10 - filled)
            console.print(
                f"  {m.get('title', mid):<{mname_w}} {bar} "
                f"{m_score:3d}/100  ({m_done}/{len(m_tasks)} tasks)"
            )


# Alias: `gpu` with no subcommand behaves like `gpu start`. Applied at
# import time so it works whether the CLI is invoked as `python3 gpu.py`
# or via the installed `gpu` console script (which calls app() directly,
# bypassing the __main__ guard).
if len(sys.argv) == 1:
    sys.argv.append("start")

if __name__ == "__main__":
    app()
