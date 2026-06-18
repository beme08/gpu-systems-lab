"""Tests for storage.json round-trip stability (write -> read -> write)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import gpu


def test_write_read_write_is_stable(tmp_storage: Path):
    """A round-trip (write -> read -> write) produces a stable file."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    data["started"] = True
    data["completed"].append("bandwidth_test")
    data["bottlenecks"]["bandwidth_test"] = "compute_bound"
    data["reality"]["vector_add"] = "memory bound"
    data["welcomed"] = True
    data["completed_at"]["bandwidth_test"] = "2026-01-15T10:00:00+00:00"
    gpu.save_storage(data)
    first = tmp_storage.read_text()
    data2 = gpu.load_storage()
    gpu.save_storage(data2)
    second = tmp_storage.read_text()
    # Both serializations should match exactly
    assert first == second


def test_roundtrip_preserves_all_keys(tmp_storage: Path):
    """A round-trip preserves every key in the v0.21 schema."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    # Set every key
    data["teaching"]["vector_add"] = [{"question": "q", "answer": "a", "feedback": "f", "asked_at": "x"}]
    data["llm_cache"]["vector_add|0|a"] = {"model": "gpt-4o-mini", "text": "ok", "cached_at": "x"}
    data["bottleneck_followup"]["bandwidth_test"] = {
        "question": "q", "answer": "a", "feedback": "f",
        "misconception_hit": False, "asked_at": "x",
    }
    gpu.save_storage(data)
    data2 = gpu.load_storage()
    assert "teaching" in data2 and data2["teaching"]["vector_add"][0]["question"] == "q"
    assert "llm_cache" in data2 and "vector_add|0|a" in data2["llm_cache"]
    assert "bottleneck_followup" in data2 and data2["bottleneck_followup"]["bandwidth_test"]["question"] == "q"


def test_completed_list_order_preserved(tmp_storage: Path):
    """The completed list preserves insertion order (not alphabetical)."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    data["completed"] = ["c", "a", "b"]
    gpu.save_storage(data)
    data2 = gpu.load_storage()
    assert data2["completed"] == ["c", "a", "b"]


def test_unicode_in_storage_preserved(tmp_storage: Path):
    """Unicode characters in storage fields (answers, paths) survive a round-trip."""
    roadmap = gpu.load_roadmap()
    data = gpu.default_storage(roadmap)
    data["reality"]["vector_add"] = "mémory bøund — it's the memory bus"
    data["benchmarks"]["vector_add"] = {"path": "labs/测试/output.md", "summary": "测试结果", "recorded_at": "2026-01-15T10:00:00+00:00"}
    gpu.save_storage(data)
    data2 = gpu.load_storage()
    assert data2["reality"]["vector_add"] == "mémory bøund — it's the memory bus"
    assert data2["benchmarks"]["vector_add"]["path"] == "labs/测试/output.md"
    assert data2["benchmarks"]["vector_add"]["summary"] == "测试结果"
