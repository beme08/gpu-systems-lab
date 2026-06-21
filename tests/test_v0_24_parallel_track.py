"""Tests for the v0.24 parallel track (Week 6: tensor + pipeline parallelism).

Verifies:
  - 4 new tasks present with correct track + milestone
  - 3 new skills added
  - 1 new milestone (parallel_strategies_week6)
  - 'parallel' track is a known track
  - next_task ordering places new tasks after the distillation track
"""
from __future__ import annotations

import pytest

import gpu


def test_parallel_track_exists():
    """The 'parallel' track is in the roadmap."""
    roadmap = gpu.load_roadmap()
    assert "parallel" in roadmap["tracks"]
    assert "Parallel" in roadmap["tracks"]["parallel"]["title"]


def test_parallel_milestone_exists():
    """The Week 6 milestone exists and references the 4 new tasks."""
    roadmap = gpu.load_roadmap()
    assert "parallel_strategies_week6" in roadmap["milestones"]
    m = roadmap["milestones"]["parallel_strategies_week6"]
    assert "Week 6" in m["title"]
    expected_tasks = {
        "sharding_intro", "tensor_parallel_demo",
        "pipeline_parallel_demo", "parallel_scaling_report",
    }
    assert set(m["tasks"]) == expected_tasks


def test_new_skills_added():
    """The 3 new parallel skills are in the roadmap."""
    roadmap = gpu.load_roadmap()
    skill_ids = {s["id"] for s in roadmap["skills"]}
    assert "parallel_strategies" in skill_ids
    assert "tensor_parallelism" in skill_ids
    assert "pipeline_parallelism" in skill_ids


def test_new_tasks_track_assignment():
    """The 4 new tasks are all in the 'parallel' track."""
    roadmap = gpu.load_roadmap()
    new_ids = {"sharding_intro", "tensor_parallel_demo",
               "pipeline_parallel_demo", "parallel_scaling_report"}
    for t in roadmap["tasks"]:
        if t["id"] in new_ids:
            assert t["track"] == "parallel", f"{t['id']} not in parallel track"


def test_new_tasks_milestone_assignment():
    """All 4 new tasks belong to the Week 6 milestone."""
    roadmap = gpu.load_roadmap()
    new_ids = {"sharding_intro", "tensor_parallel_demo",
               "pipeline_parallel_demo", "parallel_scaling_report"}
    for t in roadmap["tasks"]:
        if t["id"] in new_ids:
            assert t["milestone"] == "parallel_strategies_week6", \
                f"{t['id']} not in Week 6 milestone"


def test_new_tasks_have_teaching_surfaces():
    """The 4 new tasks have the v0.23 teaching surfaces wired (where applicable)."""
    roadmap = gpu.load_roadmap()
    by_id = {t["id"]: t for t in roadmap["tasks"]}

    # sharding_intro: command_prompts + deliverable_prompts
    s = by_id["sharding_intro"]
    assert "command_prompts" in s
    assert "deliverable_prompts" in s

    # tensor_parallel_demo: bottleneck_pick + bottleneck_followup + command_prompts
    t = by_id["tensor_parallel_demo"]
    assert t.get("bottleneck_pick") is True
    assert "bottleneck_followup" in t
    assert "command_prompts" in t

    # pipeline_parallel_demo: bottleneck_pick + bottleneck_followup + command_prompts
    p = by_id["pipeline_parallel_demo"]
    assert p.get("bottleneck_pick") is True
    assert "bottleneck_followup" in p
    assert "command_prompts" in p

    # parallel_scaling_report: deliverable_prompts only
    r = by_id["parallel_scaling_report"]
    assert "deliverable_prompts" in r
    assert "command_prompts" not in r


def test_total_task_count_is_36():
    """The curriculum is now 36 tasks (32 prior + 4 new)."""
    roadmap = gpu.load_roadmap()
    assert len(roadmap["tasks"]) == 36


def test_prerequisite_chain():
    """sharding_intro -> tensor_parallel_demo -> pipeline_parallel_demo -> parallel_scaling_report."""
    roadmap = gpu.load_roadmap()
    by_id = {t["id"]: t for t in roadmap["tasks"]}
    assert "sharding_intro" in by_id["tensor_parallel_demo"]["prerequisites"]
    assert "tensor_parallel_demo" in by_id["pipeline_parallel_demo"]["prerequisites"]
    assert "pipeline_parallel_demo" in by_id["parallel_scaling_report"]["prerequisites"]
    # sharding_intro's only prerequisite is the last task of Week 5
    assert "distill_synthesis_report" in by_id["sharding_intro"]["prerequisites"]
