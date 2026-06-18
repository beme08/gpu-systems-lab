"""Tests for score math (v0.8)."""
from __future__ import annotations

import pytest

import gpu


def test_task_raw_score_basic():
    """A task's raw score is impact * depth * reproducibility."""
    t = {"id": "x", "score": {"impact": 3, "depth": 2, "reproducibility": 5}}
    assert gpu._task_raw_score(t) == 30


def test_task_raw_score_missing_score_returns_zero():
    """A task with no `score` field returns 0."""
    assert gpu._task_raw_score({"id": "x"}) == 0


def test_task_raw_score_missing_keys_returns_zero():
    """A task with a partial `score` dict returns 0 (no partial credit)."""
    assert gpu._task_raw_score({"id": "x", "score": {"impact": 3}}) == 0
    assert gpu._task_raw_score({"id": "x", "score": {"impact": 3, "depth": 2}}) == 0


def test_task_raw_score_non_dict_returns_zero():
    """A non-dict `score` field returns 0 (defensive)."""
    assert gpu._task_raw_score({"id": "x", "score": "high"}) == 0
    assert gpu._task_raw_score({"id": "x", "score": 5}) == 0


def test_track_score_returns_percentage():
    """A track's score is a 0-100 percentage of completed raw score / possible raw score."""
    tasks = [
        {"id": "a", "score": {"impact": 3, "depth": 2, "reproducibility": 5}},  # 30
        {"id": "b", "score": {"impact": 4, "depth": 3, "reproducibility": 4}},  # 48
        {"id": "c", "score": {"impact": 2, "depth": 2, "reproducibility": 2}},  # 8
    ]
    # Possible: 30+48+8 = 86. Earned (a,c): 38. 38/86 = 44% (rounded)
    assert gpu._track_score(tasks, ["a", "c"]) == 44


def test_track_score_full_completion():
    """All tasks completed -> 100."""
    tasks = [
        {"id": "a", "score": {"impact": 3, "depth": 2, "reproducibility": 5}},
        {"id": "b", "score": {"impact": 4, "depth": 3, "reproducibility": 4}},
    ]
    assert gpu._track_score(tasks, ["a", "b"]) == 100


def test_track_score_no_completion():
    """No tasks completed -> 0."""
    tasks = [{"id": "a", "score": {"impact": 3, "depth": 2, "reproducibility": 5}}]
    assert gpu._track_score(tasks, []) == 0


def test_track_score_empty_track():
    """An empty track (no tasks) returns 0 (no division by zero)."""
    assert gpu._track_score([], []) == 0
    assert gpu._track_score([], ["any"]) == 0


def test_track_score_division_by_zero_protected():
    """A track where every task has score=0 returns 0 (no NaN)."""
    tasks = [{"id": "a", "score": {"impact": 0, "depth": 0, "reproducibility": 0}}]
    assert gpu._track_score(tasks, ["a"]) == 0


def test_track_score_capped_at_100():
    """A track score never exceeds 100 (sanity check on the percentage math)."""
    # Construct a track where the percent math could overshoot
    tasks = [
        {"id": "a", "score": {"impact": 10, "depth": 10, "reproducibility": 10}},  # 1000
    ]
    assert gpu._track_score(tasks, ["a"]) == 100
