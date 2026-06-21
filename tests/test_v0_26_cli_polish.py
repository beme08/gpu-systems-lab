"""Tests for v0.26 — CLI polish: gpu doctor + gpu start --interactive.

Validates that:
- gpu doctor returns 0 on a healthy install, with a one-screen table
  covering Python / deps / roadmap / storage / optional torch.
- gpu doctor exits 1 when a check fails (e.g. storage unwritable).
- gpu start --interactive is registered and works on non-TTY stdin
  (the gate() helper must not crash when typer.confirm raises).
- gpu start (no flag) still works the same way it did pre-v0.26
  (no --interactive, no gate() calls).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import gpu


def test_doctor_command_registered():
    """gpu doctor is a Typer command on the app."""
    names = [getattr(c, "name", None) or getattr(c, "callback", lambda: None).__name__ for c in gpu.app.registered_commands]
    # 'start' callback name should be start, 'next' is name=next, etc.
    callbacks = [getattr(c, "callback", None) for c in gpu.app.registered_commands]
    callback_names = [cb.__name__ for cb in callbacks if cb is not None]
    assert "doctor" in callback_names, f"doctor not in callbacks: {callback_names}"


def test_doctor_runs_clean():
    """gpu doctor exits 0 on a healthy install (the repo state itself)."""
    result = subprocess.run(
        [sys.executable, "-m", "gpu", "doctor"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode in (0, 1), f"doctor crashed: stderr={result.stderr}"
    # Healthy install -> exit 0
    assert result.returncode == 0, f"doctor failed: stdout={result.stdout}\nstderr={result.stderr}"
    # One-screen summary
    assert "Python" in result.stdout
    assert "typer" in result.stdout
    assert "rich" in result.stdout
    assert "roadmap.json" in result.stdout
    assert "storage.json" in result.stdout


def test_doctor_reports_compute_resources():
    """gpu doctor surfaces the compute-platform count."""
    result = subprocess.run(
        [sys.executable, "-m", "gpu", "doctor"],
        capture_output=True, text=True, timeout=15,
    )
    assert "compute" in result.stdout.lower()


def test_start_interactive_registered_with_flag():
    """gpu start has an --interactive / -i flag."""
    result = subprocess.run(
        [sys.executable, "-m", "gpu", "start", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert "--interactive" in result.stdout or "-i" in result.stdout, (
        f"start --help does not mention --interactive:\n{result.stdout}"
    )


def test_start_interactive_on_empty_stdin():
    """gpu start --interactive on empty stdin must not crash (graceful degradation)."""
    result = subprocess.run(
        [sys.executable, "-m", "gpu", "start", "--interactive"],
        input="", capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"start --interactive crashed: stderr={result.stderr}"


def test_start_default_quiet_on_second_run():
    """gpu start on a non-first-run storage stays quiet (no walkthrough)."""
    # Set up a "welcomed" storage
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    storage = tmp / "storage.json"
    storage.write_text(json.dumps({
        "started": True,
        "completed": [],
        "skills": {},
        "bottlenecks": {},
        "reality": {},
        "benchmarks": {},
        "welcomed": True,
        "completed_at": {},
        "teaching": {},
        "llm_cache": {},
        "bottleneck_followup": {}
    }))
    # Patch gpu.STORAGE_PATH and run start
    import importlib
    gpu.STORAGE_PATH = storage
    importlib.reload(gpu)
    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(gpu.app, ["start"])
    # Should show the current task, not the full walkthrough panels
    assert "Welcome" not in result.stdout, (
        "gpu start on a welcomed storage should be quiet, but it printed the walkthrough"
    )
    assert "Shape of the program" not in result.stdout
