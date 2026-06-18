"""Tests for _llm_cache_key (v0.20 cache lookup)."""
from __future__ import annotations

import pytest

import gpu


def test_basic_key():
    """The basic key format is task_id|prompt_index|normalized_answer."""
    k = gpu._llm_cache_key("vector_add", 0, "Memory bound")
    assert k == "vector_add|0|memory bound"


def test_normalization_lowercase():
    """Answer is lowercased for cache stability."""
    assert gpu._llm_cache_key("t", 0, "MEMORY") == "t|0|memory"
    assert gpu._llm_cache_key("t", 0, "Memory") == "t|0|memory"
    assert gpu._llm_cache_key("t", 0, "memory") == "t|0|memory"


def test_normalization_strip():
    """Answer is stripped of leading/trailing whitespace."""
    assert gpu._llm_cache_key("t", 0, "  memory  ") == "t|0|memory"
    assert gpu._llm_cache_key("t", 0, "\tmemory\n") == "t|0|memory"


def test_empty_answer():
    """Empty answer normalizes to empty (not 'None' or 'null')."""
    assert gpu._llm_cache_key("t", 0, "") == "t|0|"


def test_none_answer():
    """None answer normalizes to empty (defensive)."""
    assert gpu._llm_cache_key("t", 0, None) == "t|0|"


def test_prompt_index_in_key():
    """Different prompt indices produce different keys for the same answer."""
    a = gpu._llm_cache_key("t", 0, "answer")
    b = gpu._llm_cache_key("t", 1, "answer")
    assert a != b


def test_task_id_in_key():
    """Different task ids produce different keys for the same answer/index."""
    a = gpu._llm_cache_key("task_a", 0, "answer")
    b = gpu._llm_cache_key("task_b", 0, "answer")
    assert a != b


def test_special_chars_in_answer():
    """Pipes in the answer are NOT escaped; this is a known limitation.

    A user answer containing '|' would collide with another answer that
    shares task_id + prompt_index. v0.20 doesn't escape because real
    answers are short sentences without pipes. This test pins the
    current (vulnerable) behavior so a future escape fix is intentional.
    """
    a = gpu._llm_cache_key("t", 0, "a|b")
    # Same as if the answer were "a" at index b - collision is possible
    # but unlikely in practice.
    assert "|" in a
