"""Tests for ask_llm_feedback (v0.17 + v0.18 + v0.20).

Covers the model allowlist, env var, cache lookup/store, bypass,
streaming, and graceful failure modes. All network calls are mocked.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

import pytest

import gpu


# ---- Helpers ----------------------------------------------------------------

def _make_nonstream_response(content: str = "LLM response") -> io.BytesIO:
    return io.BytesIO(json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode())


def _make_stream_response(*chunks: str) -> io.BytesIO:
    sse = "".join(
        'data: ' + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n"
        for c in chunks
    ) + "data: [DONE]\n"
    return io.BytesIO(sse.encode())


# ---- Model selection --------------------------------------------------------

def test_model_default(make_urlopen):
    """No env var, no arg -> default model gpt-4o-mini."""
    captured: Dict[str, Any] = {}
    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _make_nonstream_response()
    make_urlopen(fake)
    out = gpu.ask_llm_feedback("ans", "q?", "k", cache={}, task_id="t", prompt_index=0)
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert out == "LLM response"


def test_model_explicit_arg_wins(make_urlopen):
    """An explicit model arg wins over the env var."""
    captured: Dict[str, Any] = {}
    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _make_nonstream_response()
    make_urlopen(fake)
    os.environ["GPU_LLM_MODEL"] = "gpt-4o"
    gpu.ask_llm_feedback("ans", "q?", "k", model="gpt-4.1", cache={}, task_id="t", prompt_index=0)
    assert captured["body"]["model"] == "gpt-4.1"
    del os.environ["GPU_LLM_MODEL"]


def test_model_env_var(make_urlopen):
    """GPU_LLM_MODEL env var selects a model when no arg is given."""
    captured: Dict[str, Any] = {}
    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _make_nonstream_response()
    make_urlopen(fake)
    os.environ["GPU_LLM_MODEL"] = "gpt-4o"
    gpu.ask_llm_feedback("ans", "q?", "k", cache={}, task_id="t", prompt_index=0)
    assert captured["body"]["model"] == "gpt-4o"
    del os.environ["GPU_LLM_MODEL"]


def test_model_invalid_falls_back_to_default(make_urlopen):
    """An unknown model name falls back to gpt-4o-mini silently."""
    captured: Dict[str, Any] = {}
    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _make_nonstream_response()
    make_urlopen(fake)
    os.environ["GPU_LLM_MODEL"] = "gpt-99-turbo"
    gpu.ask_llm_feedback("ans", "q?", "k", cache={}, task_id="t", prompt_index=0)
    assert captured["body"]["model"] == "gpt-4o-mini"
    del os.environ["GPU_LLM_MODEL"]


# ---- Cache hit / miss / bypass ---------------------------------------------

def test_cache_miss_writes(make_urlopen):
    """A cache miss calls urlopen and writes the result to the cache."""
    cache: Dict[str, Any] = {}
    def fake(req, timeout=None):
        return _make_nonstream_response("fresh answer")
    make_urlopen(fake)
    out = gpu.ask_llm_feedback("ans", "q?", "k", cache=cache, task_id="t", prompt_index=0)
    assert out == "fresh answer"
    assert "t|0|ans" in cache
    assert cache["t|0|ans"]["text"] == "fresh answer"
    assert cache["t|0|ans"]["model"] == "gpt-4o-mini"


def test_cache_hit_short_circuits(make_urlopen):
    """A cache hit returns the cached text without calling urlopen."""
    called = {"n": 0}
    def fake(req, timeout=None):
        called["n"] += 1
        return _make_nonstream_response("fresh")
    make_urlopen(fake)
    cache = {"t|0|ans": {"model": "gpt-4o-mini", "text": "cached", "cached_at": "x"}}
    out = gpu.ask_llm_feedback("ans", "q?", "k", cache=cache, task_id="t", prompt_index=0)
    assert out == "cached"
    assert called["n"] == 0


def test_no_cache_env_bypasses(make_urlopen):
    """GPU_LLM_NO_CACHE=1 disables both cache lookup and write."""
    called = {"n": 0}
    def fake(req, timeout=None):
        called["n"] += 1
        return _make_nonstream_response("fresh")
    make_urlopen(fake)
    os.environ["GPU_LLM_NO_CACHE"] = "1"
    cache = {"t|0|ans": {"model": "gpt-4o-mini", "text": "cached", "cached_at": "x"}}
    out = gpu.ask_llm_feedback("ans", "q?", "k", cache=cache, task_id="t", prompt_index=0)
    assert out == "fresh"  # not "cached"
    assert called["n"] == 1
    assert "t|0|ans" not in cache or cache["t|0|ans"]["text"] == "cached"  # not overwritten
    del os.environ["GPU_LLM_NO_CACHE"]


# ---- Streaming --------------------------------------------------------------

def test_streaming_assembles_tokens(make_urlopen):
    """Streaming SSE deltas are assembled into the final text."""
    def fake(req, timeout=None):
        return _make_stream_response("Hello", " world", "!")
    make_urlopen(fake)
    tokens: list = []
    def cb(t: str) -> None:
        tokens.append(t)
    out = gpu.ask_llm_feedback(
        "ans", "q?", "k",
        stream_cb=cb, cache={}, task_id="t", prompt_index=0,
    )
    assert tokens == ["Hello", " world", "!"]
    assert out == "Hello world!"


def test_streaming_sends_stream_true(make_urlopen):
    """When stream_cb is provided, the request body has stream=True."""
    captured: Dict[str, Any] = {}
    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _make_stream_response("ok")
    make_urlopen(fake)
    gpu.ask_llm_feedback(
        "ans", "q?", "k",
        stream_cb=lambda t: None, cache={}, task_id="t", prompt_index=0,
    )
    assert captured["body"]["stream"] is True


def test_non_streaming_sends_stream_false(make_urlopen):
    """When stream_cb is None, the request body has stream=False."""
    captured: Dict[str, Any] = {}
    def fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _make_nonstream_response()
    make_urlopen(fake)
    gpu.ask_llm_feedback("ans", "q?", "k", cache={}, task_id="t", prompt_index=0)
    assert captured["body"]["stream"] is False


# ---- Failure modes ----------------------------------------------------------

def test_no_api_key_returns_none():
    """Empty api_key short-circuits to None, no urlopen."""
    out = gpu.ask_llm_feedback("ans", "q?", "", cache={}, task_id="t", prompt_index=0)
    assert out is None


def test_network_error_returns_none(make_urlopen):
    """A urlopen failure returns None silently (caller falls back)."""
    def fake(req, timeout=None):
        raise ConnectionError("network down")
    make_urlopen(fake)
    out = gpu.ask_llm_feedback("ans", "q?", "k", cache={}, task_id="t", prompt_index=0)
    assert out is None


def test_empty_response_returns_none(make_urlopen):
    """A response with no content returns None."""
    def fake(req, timeout=None):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": ""}}]}).encode())
    make_urlopen(fake)
    out = gpu.ask_llm_feedback("ans", "q?", "k", cache={}, task_id="t", prompt_index=0)
    assert out is None
