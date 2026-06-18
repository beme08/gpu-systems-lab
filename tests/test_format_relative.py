"""Tests for _format_relative (v0.11 'Xh ago' renderer)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import gpu


def _ago(**kwargs) -> datetime:
    """Return a datetime `kwargs` ago (default: 0 seconds = now)."""
    return datetime.now(timezone.utc) - timedelta(**kwargs)


def test_under_a_minute():
    """A delta under 60s is 'just now'."""
    assert gpu._format_relative(_ago(seconds=0), datetime.now(timezone.utc)) == "just now"
    assert gpu._format_relative(_ago(seconds=30), datetime.now(timezone.utc)) == "just now"
    assert gpu._format_relative(_ago(seconds=59), datetime.now(timezone.utc)) == "just now"


def test_minutes():
    """A delta of 1-59 minutes is 'Xm ago'."""
    assert gpu._format_relative(_ago(minutes=1), datetime.now(timezone.utc)) == "1m ago"
    assert gpu._format_relative(_ago(minutes=15), datetime.now(timezone.utc)) == "15m ago"
    assert gpu._format_relative(_ago(minutes=59), datetime.now(timezone.utc)) == "59m ago"


def test_hours():
    """A delta of 1-23 hours is 'Xh ago'."""
    assert gpu._format_relative(_ago(hours=1), datetime.now(timezone.utc)) == "1h ago"
    assert gpu._format_relative(_ago(hours=5), datetime.now(timezone.utc)) == "5h ago"
    assert gpu._format_relative(_ago(hours=23), datetime.now(timezone.utc)) == "23h ago"


def test_days():
    """A delta of 1+ days is 'Xd ago'."""
    assert gpu._format_relative(_ago(days=1), datetime.now(timezone.utc)) == "1d ago"
    assert gpu._format_relative(_ago(days=7), datetime.now(timezone.utc)) == "7d ago"
    assert gpu._format_relative(_ago(days=365), datetime.now(timezone.utc)) == "365d ago"


def test_naive_datetime_handled():
    """A naive datetime (no tz) doesn't crash; comparison is approximate."""
    naive = (datetime.now() - timedelta(hours=2))
    out = gpu._format_relative(naive, datetime.now())
    # The exact format may differ slightly due to tz handling, but it
    # should be in the 'h ago' family.
    assert "h ago" in out or "m ago" in out
