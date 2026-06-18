"""Tests for next_task (curriculum ordering)."""
from __future__ import annotations

import pytest

import gpu


def _tasks(*ids):
    return [{"id": i, "title": i} for i in ids]


def test_next_task_returns_first_when_empty():
    """An empty completed list returns the first task."""
    tasks = _tasks("a", "b", "c")
    assert gpu.next_task(tasks, [])["id"] == "a"


def test_next_task_skips_completed():
    """Completed tasks are skipped."""
    tasks = _tasks("a", "b", "c", "d")
    assert gpu.next_task(tasks, ["a"])["id"] == "b"
    assert gpu.next_task(tasks, ["a", "b"])["id"] == "c"
    assert gpu.next_task(tasks, ["a", "b", "c"])["id"] == "d"


def test_next_task_returns_none_when_all_done():
    """All tasks completed -> None."""
    tasks = _tasks("a", "b", "c")
    assert gpu.next_task(tasks, ["a", "b", "c"]) is None


def test_next_task_empty_task_list():
    """An empty task list returns None."""
    assert gpu.next_task([], []) is None
    assert gpu.next_task([], ["anything"]) is None


def test_next_task_ignores_unknown_completed_ids():
    """Completed ids that don't match any task are ignored (no crash)."""
    tasks = _tasks("a", "b")
    # "ghost" is not a real task; should be ignored
    assert gpu.next_task(tasks, ["ghost"])["id"] == "a"


def test_next_task_real_curriculum_first_is_cuda_installed():
    """Sanity check: the first task in the real curriculum is `cuda_installed`."""
    roadmap = gpu.load_roadmap()
    first = gpu.next_task(roadmap["tasks"], [])
    assert first is not None
    assert first["id"] == "cuda_installed"
