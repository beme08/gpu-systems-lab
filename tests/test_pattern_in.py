"""Tests for _pattern_in (case-insensitive regex helper, v0.18)."""
from __future__ import annotations

import pytest

import gpu


def test_case_insensitive_match():
    """A pattern matches the answer regardless of case."""
    assert gpu._pattern_in("memory", "MEMORY bound") is True
    assert gpu._pattern_in("memory", "memory bound") is True
    assert gpu._pattern_in("memory", "Memory Bound") is True


def test_no_match():
    """A pattern that does not match returns False."""
    assert gpu._pattern_in("compute", "memory bound") is False


def test_empty_pattern_returns_true():
    """An empty pattern is a zero-width regex match (returns True).

    This is standard Python regex behavior. In practice, roadmap.json
    never has empty patterns, so this is just a contract pin.
    """
    assert gpu._pattern_in("", "any text") is True


def test_malformed_pattern_returns_false():
    """A malformed regex (e.g. unclosed bracket) returns False, no exception."""
    # An unclosed character class is invalid
    assert gpu._pattern_in("[unclosed", "any text") is False


def test_special_regex_chars():
    """Regex special chars work: alternation, anchors, word boundaries."""
    assert gpu._pattern_in(r"memory.*bound", "it is memory bound") is True
    assert gpu._pattern_in(r"^\s*kernel", "kernel launch overhead") is True
    # Word boundary prevents "is small" matching the bare substring "sm"
    # (the v0.18/v0.19 bug class). It also prevents "sm" inside "sm_count"
    # because _ is a word char (no boundary between m and _).
    assert gpu._pattern_in(r"\bsm\b", "is small") is False
    assert gpu._pattern_in(r"\bsm\b", "sm_count is high") is False
    # But it does match a standalone "sm" word
    assert gpu._pattern_in(r"\bsm\b", "the sm is the bottleneck") is True


def test_alternation():
    """Pipe alternation works."""
    assert gpu._pattern_in("memory|bandwidth", "bandwidth is the issue") is True
    assert gpu._pattern_in("memory|bandwidth", "compute is the issue") is False


def test_multi_line_answer():
    """Patterns match across the whole string (not just the first line)."""
    assert gpu._pattern_in("kv.*cache", "first line\nsecond line mentions kv cache") is True


def test_re_uses_caller_lower():
    """_pattern_in is case-insensitive on the answer; the pattern is treated as regex.

    The v0.16 caller passes the answer lowercased; _pattern_in applies
    IGNORECASE so the pattern doesn't need to be lowercased. This test
    pins the contract.
    """
    # Caller is expected to lowercase; this test ensures IGNORECASE handles
    # any case variation in the pattern itself.
    assert gpu._pattern_in("Memory", "memory bound") is True
    assert gpu._pattern_in("MEMORY", "memory bound") is True
