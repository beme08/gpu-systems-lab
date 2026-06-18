"""Tests for _count_pending_prompts (v0.10 status panel)."""
from __future__ import annotations

import pytest

import gpu


def test_fresh_storage_pending_counts():
    """A fresh storage (no completions) -> all reality/bottleneck tasks are pending."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    counts = gpu._count_pending_prompts(roadmap, data)
    # The real curriculum has 4 reality_check tasks (v0.19) and 9
    # bottleneck_pick tasks. None are completed, so all are pending.
    assert counts["reality"] == 4
    assert counts["bottleneck"] == 9
    # bench_attachable is 0 because no tasks are completed yet
    assert counts["bench"] == 0


def test_completed_tasks_reduce_pending():
    """Completing a task reduces the relevant pending count."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    # Pretend cuda_installed is completed
    data["completed"].append("cuda_installed")
    # cuda_installed has no reality_check or bottleneck_pick, so pending
    # counts should not change from the fresh-storage case
    counts = gpu._count_pending_prompts(roadmap, data)
    assert counts["reality"] == 4
    assert counts["bottleneck"] == 9


def test_answering_reality_reduces_pending():
    """Recording a reality[task_id] removes the task from reality_pending."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    data["reality"]["vector_add"] = "memory bound"
    counts = gpu._count_pending_prompts(roadmap, data)
    assert counts["reality"] == 3
    assert counts["bottleneck"] == 9


def test_recording_bottleneck_reduces_pending():
    """Recording a bottlenecks[task_id] removes the task from bottleneck_pending."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    data["bottlenecks"]["bandwidth_test"] = "compute_bound"
    counts = gpu._count_pending_prompts(roadmap, data)
    assert counts["reality"] == 4
    assert counts["bottleneck"] == 8


def test_bench_attachable_counts_completed_without_benchmark():
    """bench counts tasks that are completed but have no benchmark record yet."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    # Complete bandwidth_test (has commands, no benchmark yet)
    data["completed"].append("bandwidth_test")
    counts = gpu._count_pending_prompts(roadmap, data)
    assert counts["bench"] == 1
    # Now attach a benchmark -> bench drops
    data["benchmarks"]["bandwidth_test"] = {"path": "x", "summary": "s", "recorded_at": "now"}
    counts = gpu._count_pending_prompts(roadmap, data)
    assert counts["bench"] == 0


def test_empty_roadmap_returns_zeros():
    """An empty roadmap returns all-zero counts (no division by zero risk)."""
    empty = {"tasks": []}
    data = {"completed": [], "reality": {}, "bottlenecks": {}, "benchmarks": {}}
    counts = gpu._count_pending_prompts(empty, data)
    assert counts == {"reality": 0, "bottleneck": 0, "bench": 0}
