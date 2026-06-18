"""Tests for expand_bench_path (v0.8, light validation per Q4)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import gpu


def test_tilde_expansion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """~ expands to $HOME (no shell call)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    out = gpu.expand_bench_path("~/notes/foo.md", repo_root=tmp_path)
    assert out == str(tmp_path / "notes" / "foo.md")


def test_repo_relative_resolution(tmp_path: Path):
    """A relative path resolves against the repo root, not cwd."""
    out = gpu.expand_bench_path("labs/vector_add/result.txt", repo_root=tmp_path)
    assert out == str(tmp_path / "labs" / "vector_add" / "result.txt")


def test_absolute_path_returned_as_is(tmp_path: Path):
    """An absolute path is returned unchanged (no re-rooting)."""
    abs_path = str(tmp_path / "absolute.md")
    out = gpu.expand_bench_path(abs_path, repo_root=tmp_path)
    assert out == abs_path


def test_empty_string_resolves_to_repo_root(tmp_path: Path):
    """An empty string is a relative path; resolves to the repo root itself."""
    out = gpu.expand_bench_path("", repo_root=tmp_path)
    assert out == str(tmp_path.resolve())


def test_does_not_check_existence(tmp_path: Path):
    """Per Q4: expand_bench_path does NOT stat the file. Missing -> returned."""
    out = gpu.expand_bench_path("does/not/exist.md", repo_root=tmp_path)
    assert out == str(tmp_path / "does" / "not" / "exist.md")


def test_path_with_spaces(tmp_path: Path):
    """Spaces in path are preserved (no shell parsing)."""
    out = gpu.expand_bench_path("notes/My Hardware Report.md", repo_root=tmp_path)
    assert out == str(tmp_path / "notes" / "My Hardware Report.md")


def test_dot_segments(tmp_path: Path):
    """Paths with ./ and ../ segments resolve against the repo root."""
    out = gpu.expand_bench_path("./notes/hardware.md", repo_root=tmp_path)
    assert out.endswith("notes/hardware.md")
    # The leading "./" should be normalized away
    assert "/./" not in out
