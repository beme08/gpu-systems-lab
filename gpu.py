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
import sys
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
    }


def load_roadmap() -> Dict[str, Any]:
    if not ROADMAP_PATH.exists():
        console.print(f"[red]roadmap.json not found at {ROADMAP_PATH}[/red]")
        raise typer.Exit(1)
    return json.loads(ROADMAP_PATH.read_text())


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
    for key in ("completed", "bottlenecks", "reality"):
        data.setdefault(key, {} if key != "completed" else [])
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


def render_task(task: Dict[str, Any], is_current: bool = True) -> None:
    title = task["title"]
    if is_current:
        title = f"NEXT: {title}"
    panel = Panel(
        render_task_body(task),
        title=f"[bold]{title}[/bold]",
        border_style="cyan" if is_current else "white",
    )
    console.print(panel)


def render_task_body(task: Dict[str, Any]) -> Text:
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
    console.print("(type 'skip' to leave blank)")
    ans = typer.prompt("Your answer")
    return ans.strip()


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

@app.command()
def start(
):
    """Start the program and show the first task."""
    roadmap = load_roadmap()
    data = load_storage()
    if not data["started"]:
        data["started"] = True
        save_storage(data)
        console.print(Panel(
            f"[bold]{roadmap['program']}[/bold]\n\n{roadmap['goal']}",
            title="Welcome",
            border_style="green",
        ))
    else:
        console.print("[yellow]Program already started. Showing current state.[/yellow]")

    render_progress(roadmap["tasks"], data["completed"])
    task = next_task(roadmap["tasks"], data["completed"])
    if task:
        render_task(task)
    else:
        console.print("[green]All tasks complete.[/green]")


@app.command()
def next(
):
    """Show the next pending task."""
    roadmap = load_roadmap()
    data = load_storage()
    render_progress(roadmap["tasks"], data["completed"])
    task = next_task(roadmap["tasks"], data["completed"])
    if task:
        render_task(task)
    else:
        console.print(Panel(
            "Phase 1 complete. Now write and review gpu-week1-report.md.",
            title="GPU Reality Check Complete",
            border_style="green",
        ))


@app.command()
def status(
):
    """Show progress, current task, and skill tree."""
    roadmap = load_roadmap()
    data = load_storage()
    render_progress(roadmap["tasks"], data["completed"])
    task = next_task(roadmap["tasks"], data["completed"])
    if task:
        render_task(task)
    else:
        console.print("[green]All tasks complete.[/green]")
    render_tracks(roadmap, data["completed"])
    render_skill_tree(roadmap.get("skills", []), data.get("skills", {}))


@app.command()
def skills(
):
    """Show the skill tree only."""
    roadmap = load_roadmap()
    data = load_storage()
    render_skill_tree(roadmap.get("skills", []), data.get("skills", {}))


@app.command()
def done(task_id: str = typer.Argument(..., help="Task id to mark complete")
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
        raise typer.Exit(1)

    if task_id in data["completed"]:
        console.print(f"[yellow]Task {task_id} is already complete.[/yellow]")
        return

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

    for skill, points in (task.get("skills") or {}).items():
        current = data["skills"].get(skill, 0)
        data["skills"][skill] = max(0, min(100, current + points))

    save_storage(data)
    console.print(f"[green]Completed:[/green] {task['title']}")

    render_progress(roadmap["tasks"], data["completed"])

    nxt = next_task(roadmap["tasks"], data["completed"])
    if nxt:
        render_task(nxt)
    else:
        console.print(Panel(
            "Phase 1 complete. Now write and review gpu-week1-report.md.",
            title="GPU Reality Check Complete",
            border_style="green",
        ))


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
):
    """Show bottleneck answers and reality check answers recorded so far."""
    data = load_storage()
    has_any = data.get("bottlenecks") or data.get("reality")
    if not has_any:
        console.print("[yellow]No bottleneck / reality answers recorded yet.[/yellow]")
        return

    table = Table(title="Bottleneck Log")
    table.add_column("Task", style="cyan")
    table.add_column("Bottleneck", style="yellow")
    for tid, bid in data.get("bottlenecks", {}).items():
        table.add_row(tid, bid)
    if data.get("bottlenecks"):
        console.print(table)

    if data.get("reality"):
        rt = Table(title="Reality Check Log")
        rt.add_column("Task", style="cyan")
        rt.add_column("Answer", style="magenta", overflow="fold")
        for tid, ans in data["reality"].items():
            rt.add_row(tid, ans)
        console.print(rt)


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
def tasks(
):
    """List all task ids in order."""
    roadmap = load_roadmap()
    data = load_storage()
    for i, t in enumerate(roadmap["tasks"], 1):
        mark = "[green]x[/green]" if t["id"] in data["completed"] else "[ ]"
        console.print(f"  {mark} {i:>2}. {t['id']:<18}  {t['title']}")


# Alias: `gpu` with no subcommand behaves like `gpu start`. Applied at
# import time so it works whether the CLI is invoked as `python3 gpu.py`
# or via the installed `gpu` console script (which calls app() directly,
# bypassing the __main__ guard).
if len(sys.argv) == 1:
    sys.argv.append("start")

if __name__ == "__main__":
    app()
