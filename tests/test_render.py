"""Tests for render functions (v0.22 smoke: don't crash on edge cases)."""
from __future__ import annotations

import pytest

import gpu


def test_render_task_body_minimal_task():
    """A minimal task (id + title only) returns a non-empty Text object."""
    from rich.text import Text
    task = {"id": "x", "title": "Minimal"}
    body = gpu.render_task_body(task)
    assert isinstance(body, Text)
    assert len(body) > 0


def test_render_task_body_with_score():
    """A task with a score renders the score line."""
    task = {"id": "x", "title": "T", "score": {"impact": 3, "depth": 2, "reproducibility": 5}}
    body = gpu.render_task_body(task)
    # The score is rendered as "Score: 3 x 2 x 5 = 30"
    assert "30" in body.plain or "Score" in body.plain


def test_render_task_body_with_bottleneck_hint():
    """A task with bottleneck_hint includes the hint text."""
    task = {"id": "x", "title": "T", "bottleneck_hint": "memory_bound"}
    body = gpu.render_task_body(task)
    assert "memory_bound" in body.plain


def test_render_task_body_with_compute_paths():
    """A task with compute_paths + roadmap includes a platform line."""
    roadmap = gpu.load_roadmap()
    # Find a resource id from the actual roadmap (compute_paths resolves
    # against roadmap["resources"]).
    compute_resources = [r["id"] for r in roadmap.get("resources", [])
                        if "compute" in (r.get("domains") or [])]
    if not compute_resources:
        pytest.skip("no compute resources in roadmap")
    task = {"id": "x", "title": "T", "compute_paths": [compute_resources[0]]}
    body = gpu.render_task_body(task, roadmap=roadmap)
    assert "Run on" in body.plain


def test_render_task_body_with_reality_check():
    """A task with reality_check renders the reality header."""
    task = {"id": "x", "title": "T", "reality_check": "What's the bottleneck?"}
    body = gpu.render_task_body(task)
    assert "What's the bottleneck?" in body.plain


def test_render_task_body_with_teaching_prompts():
    """A task with teaching_prompts renders the body without crashing.

    The teaching prompts themselves are surfaced by the v0.10 preview
    line ("When done, gpu done <id> will ask you to classify…") rather
    than the body itself; here we just check the body is non-empty.
    """
    task = {
        "id": "x", "title": "T",
        "teaching_prompts": [{"question": "q1"}, {"question": "q2"}],
    }
    body = gpu.render_task_body(task)
    assert len(body) > 0


def test_render_task_body_full_real_task():
    """A real curriculum task with all fields renders without crashing."""
    from rich.text import Text
    roadmap = gpu.load_roadmap()
    t = roadmap["tasks"][0]  # cuda_installed
    body = gpu.render_task_body(t, roadmap=roadmap)
    assert isinstance(body, Text)
    assert len(body) > 0


def test_render_progress_empty(mock_console):
    """An empty completed list renders a 0/N progress bar."""
    tasks = [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]
    gpu.render_progress(tasks, [])
    assert len(mock_console) >= 1


def test_render_progress_full(mock_console):
    """A fully completed curriculum renders 100%."""
    tasks = [{"id": "a"}, {"id": "b"}]
    gpu.render_progress(tasks, ["a", "b"])
    assert len(mock_console) >= 1


def test_render_walkthrough_fresh(mock_console):
    """render_walkthrough on a fresh storage prints the welcome."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    gpu.render_walkthrough(roadmap, data)
    # Multiple panels printed
    assert len(mock_console) >= 3


def test_render_skill_tree_empty(mock_console):
    """render_skill_tree with zero skills doesn't crash."""
    gpu.render_skill_tree([], {})
    # No assertion on count; just that it didn't raise
