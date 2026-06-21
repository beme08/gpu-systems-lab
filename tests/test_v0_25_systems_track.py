"""Tests for v0.25 — Week 7 multi-GPU serving.

Validates that the 4 new tasks land on the systems track, that the
prereq chain extends cleanly past v0.24, that the milestone
gpu_serving_track grows from 6 to 10 tasks, and that the 2 new skills
land in the skills list.
"""
import json
from pathlib import Path

import pytest

ROADMAP = Path(__file__).resolve().parent.parent / "roadmap.json"


@pytest.fixture(scope="module")
def roadmap():
    return json.loads(ROADMAP.read_text())


def _task(roadmap, tid):
    return next((t for t in roadmap["tasks"] if t["id"] == tid), None)


def test_v0_25_tasks_present(roadmap):
    for tid in [
        "vllm_tp_attention",
        "sglang_radix_cache",
        "disagg_prefill_decode",
        "serving_synthesis_report_v2",
    ]:
        assert _task(roadmap, tid) is not None, f"missing v0.25 task {tid}"


def test_v0_25_tasks_on_systems_track(roadmap):
    for tid in [
        "vllm_tp_attention",
        "sglang_radix_cache",
        "disagg_prefill_decode",
        "serving_synthesis_report_v2",
    ]:
        t = _task(roadmap, tid)
        assert t["track"] == "systems", f"{tid} is on track {t['track']}, expected systems"


def test_v0_25_tasks_have_teaching_surfaces(roadmap):
    surf_keys = [
        "reality_check",
        "bottleneck_followup",
        "teaching_prompts",
        "deliverable_prompts",
        "command_prompts",
    ]
    for tid in [
        "vllm_tp_attention",
        "sglang_radix_cache",
        "disagg_prefill_decode",
        "serving_synthesis_report_v2",
    ]:
        t = _task(roadmap, tid)
        assert any(t.get(k) for k in surf_keys), f"{tid} has no teaching surface"


def test_v0_25_prereq_chain(roadmap):
    # Chain must close: vllm_tp_attention <- serving_systems_report,
    # sglang_radix_cache <- vllm_tp_attention, etc.
    vllm = _task(roadmap, "vllm_tp_attention")
    sglang = _task(roadmap, "sglang_radix_cache")
    disagg = _task(roadmap, "disagg_prefill_decode")
    synth = _task(roadmap, "serving_synthesis_report_v2")
    assert "serving_systems_report" in vllm["prerequisites"]
    assert "vllm_tp_attention" in sglang["prerequisites"]
    assert "sglang_radix_cache" in disagg["prerequisites"]
    assert "disagg_prefill_decode" in synth["prerequisites"]


def test_v0_25_milestone_grew(roadmap):
    ms_tasks = roadmap["milestones"]["gpu_serving_track"]["tasks"]
    assert len(ms_tasks) == 10, f"gpu_serving_track has {len(ms_tasks)} tasks, expected 10"
    for tid in [
        "vllm_tp_attention",
        "sglang_radix_cache",
        "disagg_prefill_decode",
        "serving_synthesis_report_v2",
    ]:
        assert tid in ms_tasks, f"{tid} not in gpu_serving_track milestone"


def test_v0_25_new_skills(roadmap):
    skill_ids = {s["id"] for s in roadmap["skills"]}
    assert "vllm_serving" in skill_ids
    assert "disaggregated_inference" in skill_ids
    assert len(roadmap["skills"]) == 14


def test_v0_25_total_task_count(roadmap):
    # v0.24 = 36 tasks, +4 v0.25 = 40
    assert len(roadmap["tasks"]) == 40


def test_systems_track_count(roadmap):
    n = sum(1 for t in roadmap["tasks"] if t.get("track") == "systems")
    assert n == 10, f"systems track has {n} tasks, expected 10"
