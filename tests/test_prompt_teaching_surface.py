"""Tests for _prompt_teaching_surface (v0.23 deliverable + command_walkthrough)."""
from __future__ import annotations

import pytest

import gpu


def _deliverable_prompt():
    return {
        "question": "What are the 4 sections of your report?",
        "expected_answers": [{"pattern": r"hardware|observation", "response": "Right - 4 sections."}],
        "common_misconceptions": [{"pattern": r"tl.?dr|summary", "response": "Not quite."}],
        "expected_keywords": ["hardware", "observation"],
        "follow_up_if_match": "Good.",
        "follow_up_if_miss": "Hint: 4 sections.",
    }


def _command_prompt():
    return {
        "question": "What does each command do?",
        "expected_answers": [{"pattern": r"driver|compiler", "response": "Right."}],
        "expected_keywords": ["driver"],
        "follow_up_if_match": "Good.",
        "follow_up_if_miss": "Hint: driver + compiler.",
    }


def _task(field):
    if field == "deliverable_prompts":
        return {"id": "t", "title": "T", "deliverable_prompts": _deliverable_prompt()}
    return {"id": "t", "title": "T", "command_prompts": _command_prompt()}


# ---- Field-routing (the two surfaces use different field/storage combos) ---

def test_deliverable_field_routed_to_deliverable_storage(tmp_storage, mock_console, make_prompt):
    """A deliverable_prompts field stores under data['deliverable']."""
    data = gpu.load_storage()
    make_prompt("hardware section, observation section, bottleneck, next steps")
    gpu._prompt_teaching_surface(
        _task("deliverable_prompts"), data, "t",
        field="deliverable_prompts", storage_key="deliverable",
        title="Deliverable", border_style="cyan", feedback_style="cyan",
    )
    assert "t" in data["deliverable"]
    assert "t" not in data.get("command_walkthrough", {})


def test_command_field_routed_to_command_walkthrough_storage(tmp_storage, mock_console, make_prompt):
    """A command_prompts field stores under data['command_walkthrough']."""
    data = gpu.load_storage()
    make_prompt("driver check, then compiler check, then python binding check")
    gpu._prompt_teaching_surface(
        _task("command_prompts"), data, "t",
        field="command_prompts", storage_key="command_walkthrough",
        title="Command Walkthrough", border_style="magenta", feedback_style="magenta",
    )
    assert "t" in data["command_walkthrough"]
    assert "t" not in data.get("deliverable", {})


# ---- No-op cases -----------------------------------------------------------

def test_no_field_is_noop(tmp_storage, mock_console, make_prompt):
    """A task without the requested field is a silent no-op."""
    task = {"id": "t", "title": "T"}  # no field
    data = gpu.load_storage()
    gpu._prompt_teaching_surface(
        task, data, "t",
        field="deliverable_prompts", storage_key="deliverable",
        title="Deliverable", border_style="cyan", feedback_style="cyan",
    )
    assert "deliverable" not in data or not data["deliverable"]


def test_skip_short_circuits(tmp_storage, mock_console, make_prompt):
    """Typing 'skip' at the prompt -> no record, no console output beyond panel."""
    data = gpu.load_storage()
    make_prompt("skip")
    gpu._prompt_teaching_surface(
        _task("deliverable_prompts"), data, "t",
        field="deliverable_prompts", storage_key="deliverable",
        title="Deliverable", border_style="cyan", feedback_style="cyan",
    )
    assert "deliverable" not in data or not data["deliverable"]


def test_empty_answer_treated_as_skip(tmp_storage, mock_console, make_prompt):
    """An empty-string answer is treated as skip."""
    data = gpu.load_storage()
    make_prompt("")
    gpu._prompt_teaching_surface(
        _task("command_prompts"), data, "t",
        field="command_prompts", storage_key="command_walkthrough",
        title="Command Walkthrough", border_style="magenta", feedback_style="magenta",
    )
    assert "command_walkthrough" not in data or not data["command_walkthrough"]


def test_rerun_detection_skips_silently(tmp_storage, mock_console, make_prompt):
    """A pre-existing record short-circuits the prompt (no re-record)."""
    data = {
        "deliverable": {
            "t": {"question": "old", "answer": "old", "feedback": "old", "asked_at": "x"}
        }
    }
    gpu._prompt_teaching_surface(
        _task("deliverable_prompts"), data, "t",
        field="deliverable_prompts", storage_key="deliverable",
        title="Deliverable", border_style="cyan", feedback_style="cyan",
    )
    assert data["deliverable"]["t"]["answer"] == "old"


# ---- Record shape -----------------------------------------------------------

def test_records_question_answer_feedback_asked_at(tmp_storage, mock_console, make_prompt):
    """A non-skip answer records all 4 fields (no misconception_hit at this layer)."""
    data = gpu.load_storage()
    make_prompt("hardware section, observation, bottleneck, next steps")
    gpu._prompt_teaching_surface(
        _task("deliverable_prompts"), data, "t",
        field="deliverable_prompts", storage_key="deliverable",
        title="Deliverable", border_style="cyan", feedback_style="cyan",
    )
    rec = data["deliverable"]["t"]
    assert rec["question"].startswith("What are the 4 sections")
    assert rec["answer"].startswith("hardware section")
    assert "feedback" in rec
    assert "asked_at" in rec
    # No misconception_hit key at the v0.23 surface (the misconception
    # flag is a v0.21 bottleneck-followup concept; v0.23 surfaces don't
    # need it because the answer is about content, not classification).
    assert "misconception_hit" not in rec


def test_layered_feedback_fires(tmp_storage, mock_console, make_prompt):
    """The misconception regex fires and returns the 'not quite' response."""
    data = gpu.load_storage()
    # Trigger the misconception pattern: "tl;dr summary"
    make_prompt("It's a quick tl;dr summary of my findings")
    gpu._prompt_teaching_surface(
        _task("deliverable_prompts"), data, "t",
        field="deliverable_prompts", storage_key="deliverable",
        title="Deliverable", border_style="cyan", feedback_style="cyan",
    )
    rec = data["deliverable"]["t"]
    assert rec["feedback"].startswith("Not quite")
