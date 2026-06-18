"""Tests for give_teaching_feedback (v0.16 + v0.18 layered cascade).

The 4-layer order is critical:
  1. common_misconceptions (regex)
  2. expected_answers (regex)
  3. expected_keywords (substring, count >= 2)
  4. follow_up_if_miss / "Consider: <keywords>"

These tests pin the priority order and the v0.18 word-boundary regression
class (sm in "is small" must NOT match the bare substring "sm").
"""
from __future__ import annotations

import pytest

import gpu


def _prompt(*, misconceptions=None, expected=None, keywords=None, match=None, miss=None):
    # Distinguish "not provided" (None -> default) from "explicitly empty"
    # (use as-is). The give_teaching_feedback layer-4 line uses an
    # empty follow_up_if_miss to fall through to the auto-built
    # 'Consider: <keywords>.' line.
    return {
        "question": "test?",
        "common_misconceptions": misconceptions or [],
        "expected_answers": expected or [],
        "expected_keywords": keywords or [],
        "follow_up_if_match": match if match is not None else "Good match.",
        "follow_up_if_miss": miss if miss is not None else "Consider: relevant keyword.",
    }


# ---- Layer 1: common_misconceptions wins over expected_answers ------------

def test_misconception_fires_before_expected():
    """A misconception match wins even if an expected_answers would also match."""
    p = _prompt(
        misconceptions=[{"pattern": "gpu.*fast", "response": "Not quite - it's the memory bus."}],
        expected=[{"pattern": "fast", "response": "Right - the bus is fast."}],
    )
    out = gpu.give_teaching_feedback("The GPU is fast", p)
    assert out == "Not quite - it's the memory bus."


def test_misconception_pattern_does_not_match_skips_layer():
    """If no misconception matches, the function continues to layer 2."""
    p = _prompt(
        misconceptions=[{"pattern": "completely wrong phrase", "response": "Not quite."}],
        expected=[{"pattern": "memory", "response": "Right - memory bound."}],
    )
    out = gpu.give_teaching_feedback("it is memory bound", p)
    assert out == "Right - memory bound."


# ---- Layer 2: expected_answers (regex, in order) -------------------------

def test_expected_answers_first_match_wins():
    """Multiple expected_answers; the first matching one is returned."""
    p = _prompt(
        expected=[
            {"pattern": "memory", "response": "A: memory."},
            {"pattern": "bandwidth", "response": "B: bandwidth."},
        ],
    )
    out = gpu.give_teaching_feedback("memory and bandwidth both apply", p)
    assert out == "A: memory."


def test_expected_answers_no_match_falls_through():
    """No expected_answers match -> falls through to layer 3 (keywords)."""
    p = _prompt(
        expected=[{"pattern": "specific phrase", "response": "A."}],
        keywords=["memory", "bandwidth", "flop"],
    )
    out = gpu.give_teaching_feedback("it touches memory and bandwidth", p)
    assert out == p["follow_up_if_match"]


# ---- Layer 3: expected_keywords (substring, count >= 2) ------------------

def test_keywords_two_or_more_matches_fires_layer_3():
    """Two or more keyword substring matches -> follow_up_if_match."""
    p = _prompt(keywords=["memory", "bandwidth", "flop"])
    out = gpu.give_teaching_feedback("memory and bandwidth both matter", p)
    assert out == p["follow_up_if_match"]


def test_keywords_one_match_falls_through_to_miss():
    """Only one keyword hit (< 2) is not enough; falls through to miss."""
    p = _prompt(keywords=["memory", "bandwidth", "flop"])
    out = gpu.give_teaching_feedback("it touches memory", p)
    assert out == p["follow_up_if_miss"]


def test_keywords_case_insensitive():
    """Keyword matching is case-insensitive (answer is lowercased internally)."""
    p = _prompt(keywords=["memory", "bandwidth", "flop"])
    out = gpu.give_teaching_feedback("MEMORY and BANDWIDTH both matter", p)
    assert out == p["follow_up_if_match"]


def test_keywords_empty_list_falls_through_to_miss():
    """No keywords at all -> miss fallback."""
    p = _prompt(keywords=[])
    out = gpu.give_teaching_feedback("anything goes", p)
    assert out == p["follow_up_if_miss"]


# ---- Layer 4: miss fallback ----------------------------------------------

def test_miss_fallback_uses_follow_up_if_miss():
    """When no layer matches, follow_up_if_miss is returned."""
    p = _prompt(
        expected=[{"pattern": "specific phrase", "response": "A."}],
        keywords=["memory", "bandwidth"],
        miss="Try memory + bandwidth.",
    )
    out = gpu.give_teaching_feedback("nothing matches", p)
    assert out == "Try memory + bandwidth."


def test_miss_fallback_no_follow_up_uses_consider_line():
    """If follow_up_if_miss is empty, a 'Consider: <kws>.' line is auto-built."""
    p = _prompt(
        expected=[{"pattern": "specific", "response": "A."}],
        keywords=["memory", "bandwidth", "flop", "sm"],
        miss="",  # empty -> falls through to the auto-built line
    )
    out = gpu.give_teaching_feedback("nothing matches", p)
    assert out == "Consider: memory, bandwidth, flop."


# ---- Word-boundary anchor regression (v0.18/v0.19 bug class) --------------

def test_word_boundary_anchors_prevent_false_positives():
    """Word-boundary anchors must not match the bare substring.

    Regression: a pattern of 'sm' would match 'is small' (the 'sm' inside
    'small'). Word boundaries \\bsm\\b fix this.
    """
    p_bare = _prompt(
        expected=[{"pattern": r"\bsm\b", "response": "sm match."}],
    )
    p_anchored = _prompt(
        expected=[{"pattern": r"\bsm\b", "response": "sm match."}],
    )
    # Bare substring 'sm' (no anchors) would match "is small" -- but the
    # v0.18 fix uses word boundaries, so it must NOT match.
    assert gpu.give_teaching_feedback("is small", p_anchored) != "sm match."
    # The same prompt with a real 'sm' token should match.
    assert gpu.give_teaching_feedback("the sm is busy", p_anchored) == "sm match."


# ---- Defensive: empty answer --------------------------------------------

def test_empty_answer_falls_through_to_miss():
    """An empty answer doesn't match anything; miss fallback fires."""
    p = _prompt(
        expected=[{"pattern": "memory", "response": "A."}],
        keywords=["memory", "bandwidth"],
        miss="miss line",
    )
    out = gpu.give_teaching_feedback("", p)
    assert out == "miss line"


def test_none_answer_does_not_crash():
    """A None answer is treated as empty (defensive lower())."""
    p = _prompt(miss="miss line")
    out = gpu.give_teaching_feedback(None, p)
    assert out == "miss line"


# ---- Malformed prompt shape ---------------------------------------------

def test_misconception_missing_response_returns_default():
    """A misconception entry with no 'response' returns 'Not quite.'"""
    p = _prompt(misconceptions=[{"pattern": "memory"}])  # no response key
    out = gpu.give_teaching_feedback("memory", p)
    assert out == "Not quite."
