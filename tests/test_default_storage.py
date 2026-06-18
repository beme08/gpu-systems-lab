"""Tests for default_storage + load_storage backfill (v0.22)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import gpu


def test_default_storage_has_all_v0_21_keys():
    """default_storage returns the v0.21 shape with all 11 keys."""
    roadmap = gpu.load_roadmap()
    s = gpu.default_storage(roadmap)
    assert "started" in s
    assert s["started"] is False
    assert s["completed"] == []
    assert "skills" in s
    # 9 skills in the current roadmap
    assert len(s["skills"]) == len(roadmap.get("skills", []))
    for sid in [sk["id"] for sk in roadmap["skills"]]:
        assert s["skills"][sid] == 0
    # v0.8+
    assert s["bottlenecks"] == {}
    assert s["reality"] == {}
    assert s["benchmarks"] == {}
    # v0.10
    assert s["welcomed"] is False
    # v0.11
    assert s["completed_at"] == {}
    # v0.16
    assert s["teaching"] == {}
    # v0.20
    assert s["llm_cache"] == {}
    # v0.21
    assert s["bottleneck_followup"] == {}


def test_default_storage_is_independent_per_call():
    """Two default_storage() calls don't share mutable state."""
    roadmap = gpu.load_roadmap()
    a = gpu.default_storage(roadmap)
    b = gpu.default_storage(roadmap)
    a["completed"].append("cuda_installed")
    a["bottlenecks"]["x"] = "memory_bound"
    assert b["completed"] == []
    assert b["bottlenecks"] == {}


def test_load_storage_missing_file_returns_default(tmp_storage: Path):
    """Missing storage.json -> default_storage."""
    assert not tmp_storage.exists()
    s = gpu.load_storage()
    assert s["started"] is False
    assert s["completed"] == []
    assert s["bottleneck_followup"] == {}


def test_load_storage_corrupt_json_returns_default(tmp_storage: Path):
    """Corrupt JSON -> default_storage (no exception)."""
    tmp_storage.write_text("not valid json {{{")
    s = gpu.load_storage()
    assert s["started"] is False
    assert s["completed"] == []


def test_load_storage_backfills_legacy(tmp_storage: Path):
    """A v0.7-era storage.json (no new keys) gets every new key backfilled."""
    legacy = {
        "started": True,
        "completed": ["bandwidth_test"],
        "skills": {"memory_hierarchy": 5},
        "bottlenecks": {"bandwidth_test": "compute_bound"},
        "reality": {},
        "benchmarks": {},
    }
    tmp_storage.write_text(json.dumps(legacy))
    s = gpu.load_storage()
    # legacy data preserved
    assert s["completed"] == ["bandwidth_test"]
    assert s["bottlenecks"]["bandwidth_test"] == "compute_bound"
    # new keys backfilled
    assert s["welcomed"] is False
    assert s["completed_at"] == {}
    assert s["teaching"] == {}
    assert s["llm_cache"] == {}
    assert s["bottleneck_followup"] == {}
    # skills backfill: any new skill id in roadmap is added at 0
    roadmap = gpu.load_roadmap()
    for sk in roadmap["skills"]:
        assert sk["id"] in s["skills"]


def test_load_storage_backfills_individual_legacy_keys(tmp_storage: Path):
    """Each new key is backfilled independently (v0.10/11/16/20/21)."""
    # v0.10-era: has welcomed but no completed_at/teaching/llm_cache/bottleneck_followup
    legacy = {
        "started": True,
        "completed": [],
        "skills": {},
        "bottlenecks": {},
        "reality": {},
        "benchmarks": {},
        "welcomed": True,
    }
    tmp_storage.write_text(json.dumps(legacy))
    s = gpu.load_storage()
    assert s["welcomed"] is True
    assert s["completed_at"] == {}
    assert s["teaching"] == {}
    assert s["llm_cache"] == {}
    assert s["bottleneck_followup"] == {}
