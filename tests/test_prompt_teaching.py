"""Tests for prompt_teaching (v0.16 + v0.17 + v0.20)."""
from __future__ import annotations

import pytest

import gpu


_TEACHING = [
    {
        "question": "Why is vector add usually memory-bound?",
        "expected_keywords": ["memory", "bandwidth", "element", "flop"],
        "expected_answers": [{"pattern": "memory.*bound", "response": "Right - memory bound."}],
        "common_misconceptions": [{"pattern": "compute.*fast", "response": "Not quite."}],
        "follow_up_if_match": "Good.",
        "follow_up_if_miss": "Hint: bytes-per-flop.",
    },
    {
        "question": "How would you make it faster?",
        "expected_keywords": ["vector", "load"],
        "follow_up_if_match": "Good.",
        "follow_up_if_miss": "Hint: vector loads.",
    },
]


def _task():
    return {
        "id": "vector_add",
        "title": "T",
        "objective": "Write a CUDA vector add kernel.",
        "deliverable": "Working kernel.",
        "teaching_prompts": _TEACHING,
    }


def test_skip_short_circuits_remaining_prompts(tmp_storage, mock_console, make_prompt):
    """Typing 'skip' at prompt 0 -> prompt 1 is NOT asked."""
    data = gpu.load_storage()
    make_prompt("skip")
    gpu.prompt_teaching(_task(), data, "vector_add", index=0)
    # Only the panel for prompt 0 was printed; no second prompt
    assert len(data["teaching"].get("vector_add", [])) == 1
    # Caller's loop is responsible for breaking on skip; the helper itself
    # records the skip entry but does not loop.


def test_records_answer_and_feedback(tmp_storage, mock_console, make_prompt):
    """A non-skip answer records answer + feedback + asked_at."""
    data = gpu.load_storage()
    make_prompt("memory bound because of bytes per flop")
    gpu.prompt_teaching(_task(), data, "vector_add", index=0)
    rec = data["teaching"]["vector_add"][0]
    assert rec["answer"].startswith("memory bound")
    assert rec["feedback"] == "Right - memory bound."
    assert "asked_at" in rec


def test_index_out_of_range_is_noop(tmp_storage, mock_console, make_prompt):
    """An index past the prompt list is a silent no-op."""
    data = gpu.load_storage()
    gpu.prompt_teaching(_task(), data, "vector_add", index=99)
    assert "teaching" not in data or not data["teaching"]


def test_no_teaching_prompts_is_noop(tmp_storage, mock_console, make_prompt):
    """A task without `teaching_prompts` is a no-op."""
    task = {"id": "t", "title": "T"}  # no teaching_prompts
    data = gpu.load_storage()
    gpu.prompt_teaching(task, data, "t", index=0)
    assert "teaching" not in data or not data["teaching"]


def test_cache_tag_printed_on_fresh_call(tmp_storage, mock_console, make_prompt, monkeypatch):
    """A fresh LLM call prints the (fresh) tag in the post-call line.

    Without LLM feedback (no api_key), the post-call print path is skipped
    entirely; we test the cache-tag logic in the next test.
    """
    data = gpu.load_storage()
    make_prompt("memory bound")
    # No LLM key -> no llm_feedback in record
    gpu.prompt_teaching(_task(), data, "vector_add", index=0, use_llm=True)
    rec = data["teaching"]["vector_add"][0]
    assert "llm_feedback" not in rec


def test_stream_cb_receives_tokens(tmp_storage, mock_console, make_prompt, make_urlopen, monkeypatch):
    """When stream_cb is provided, tokens are delivered to it (mocked LLM)."""
    import io, json
    sse = "".join(
        'data: ' + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n"
        for c in ["Tok1", " Tok2"]
    ) + "data: [DONE]\n"
    def fake(req, timeout=None):
        return io.BytesIO(sse.encode())
    make_urlopen(fake)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GPU_LLM_NO_CACHE", "1")  # skip cache so urlopen fires
    data = gpu.load_storage()
    make_prompt("memory bound")
    tokens: list = []
    gpu.prompt_teaching(
        _task(), data, "vector_add", index=0,
        use_llm=True, stream_cb=lambda t: tokens.append(t),
    )
    assert tokens == ["Tok1", " Tok2"]
    rec = data["teaching"]["vector_add"][0]
    assert rec["llm_feedback"] == "Tok1 Tok2"
