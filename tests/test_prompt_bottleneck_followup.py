"""Tests for prompt_bottleneck_followup (v0.21)."""
from __future__ import annotations

import pytest

import gpu


_BF = {
    "question": "Why is bandwidthTest's main number device-to-device?",
    "expected_answers": [{"pattern": r"pcie|host|overhead", "response": "PCIe matters at transfer time."}],
    "common_misconceptions": [{"pattern": r"many core|parallel.*fast", "response": "Not quite - it's the bus."}],
    "expected_keywords": ["memory", "bandwidth"],
    "follow_up_if_match": "Right.",
    "follow_up_if_miss": "Hint: device-memory bus.",
}


def test_no_followup_field_is_noop(tmp_storage, mock_console, make_prompt):
    """A task without `bottleneck_followup` is a silent no-op."""
    task = {"id": "t", "title": "T"}
    data = gpu.load_storage()
    gpu.prompt_bottleneck_followup(task, data, "t")
    assert "bottleneck_followup" not in data or not data["bottleneck_followup"]
    assert mock_console == []


def test_skip_short_circuits(tmp_storage, mock_console, make_prompt):
    """Typing 'skip' at the prompt -> no record, no console output beyond the panel."""
    task = {"id": "t", "title": "T", "bottleneck_followup": _BF}
    data = gpu.load_storage()
    make_prompt("skip")
    gpu.prompt_bottleneck_followup(task, data, "t")
    assert "bottleneck_followup" not in data or not data["bottleneck_followup"]


def test_records_feedback_on_answer(tmp_storage, mock_console, make_prompt):
    """A non-skip answer records the feedback + question + answer + asked_at."""
    task = {"id": "t", "title": "T", "bottleneck_followup": _BF}
    data = gpu.load_storage()
    make_prompt("PCIe matters but device-to-device is the real bottleneck")
    gpu.prompt_bottleneck_followup(task, data, "t")
    rec = data["bottleneck_followup"]["t"]
    assert rec["answer"].startswith("PCIe matters")
    assert rec["feedback"] == "PCIe matters at transfer time."
    assert "asked_at" in rec
    assert rec["misconception_hit"] is False


def test_misconception_hit_recorded(tmp_storage, mock_console, make_prompt):
    """An answer matching common_misconceptions sets misconception_hit=True."""
    task = {"id": "t", "title": "T", "bottleneck_followup": _BF}
    data = gpu.load_storage()
    make_prompt("the GPU has many cores so it is parallel and fast")
    gpu.prompt_bottleneck_followup(task, data, "t")
    rec = data["bottleneck_followup"]["t"]
    assert rec["misconception_hit"] is True
    assert "Not quite" in rec["feedback"]


def test_rerun_detection_skips_silently(tmp_storage, mock_console, make_prompt):
    """A pre-existing record short-circuits the prompt (no console, no re-record)."""
    task = {"id": "t", "title": "T", "bottleneck_followup": _BF}
    data = {
        "bottleneck_followup": {
            "t": {"question": "old", "answer": "old", "feedback": "old", "misconception_hit": False, "asked_at": "x"}
        }
    }
    # If the prompt is NOT short-circuited, the make_prompt queue exhaustion
    # would raise StopIteration. No make_prompt call -> the fixture's prompt
    # is never installed, so any prompt() call would use the real typer.
    gpu.prompt_bottleneck_followup(task, data, "t")
    # The old record is unchanged
    assert data["bottleneck_followup"]["t"]["answer"] == "old"


def test_empty_answer_treated_as_skip(tmp_storage, mock_console, make_prompt):
    """An empty-string answer is treated as skip (no record)."""
    task = {"id": "t", "title": "T", "bottleneck_followup": _BF}
    data = gpu.load_storage()
    make_prompt("")
    gpu.prompt_bottleneck_followup(task, data, "t")
    assert "bottleneck_followup" not in data or not data["bottleneck_followup"]
