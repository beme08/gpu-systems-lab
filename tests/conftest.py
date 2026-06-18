"""Shared pytest fixtures for gpu-systems-lab tests (v0.22).

These fixtures:
  - redirect STORAGE_PATH to a per-test tmp file
  - capture console.print calls
  - mock urllib.request.urlopen for ask_llm_feedback tests
  - inject canned answers into typer.prompt
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

import gpu


@pytest.fixture
def tmp_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect gpu.STORAGE_PATH to a tmp file. Returns the path."""
    p = tmp_path / "storage.json"
    monkeypatch.setattr(gpu, "STORAGE_PATH", p)
    return p


@pytest.fixture
def mock_console(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Replace gpu.console.print with a recorder. Returns the list of calls."""
    printed: List[Any] = []
    monkeypatch.setattr(gpu.console, "print", lambda *a, **kw: printed.append((a, kw)))
    return printed


@pytest.fixture
def make_urlopen(monkeypatch: pytest.MonkeyPatch):
    """Factory: install a fake urllib.request.urlopen.

    Usage:
        def f(req, timeout=None): return io.BytesIO(b'...')
        make_urlopen(f)
    """
    def _install(fn: Callable[..., Any]) -> None:
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fn)
    return _install


@pytest.fixture
def make_prompt(monkeypatch: pytest.MonkeyPatch):
    """Factory: install a fake typer.prompt.

    Usage:
        make_prompt("answer1", "answer2")  # raises StopIteration after exhaustion
        make_prompt(["a", "b"])             # explicit list
    """
    import typer

    def _install(*answers: Any) -> None:
        if len(answers) == 1 and isinstance(answers[0], (list, tuple)):
            queue = list(answers[0])
        else:
            queue = list(answers)
        def _fake_prompt(*a: Any, **kw: Any) -> str:
            if not queue:
                raise StopIteration("typer.prompt called more times than expected")
            return queue.pop(0)
        monkeypatch.setattr(typer, "prompt", _fake_prompt)
    return _install
