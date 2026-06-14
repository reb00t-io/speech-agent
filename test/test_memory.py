"""Tests for src/memory.py — persistent long-term memory layer."""
import os

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")

import src.memory as memory  # noqa: E402


def _fresh(tmp_path):
    memory._model = None
    memory.DATA_DIR = str(tmp_path / "mem")
    memory.ENABLED = True


def test_disabled_is_noop(tmp_path, monkeypatch):
    _fresh(tmp_path)
    monkeypatch.setattr(memory, "ENABLED", False)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert memory.inject(msgs) is msgs
    assert memory.recall_messages() == []
    memory.observe("u", "a")  # no-op, must not raise


def test_inject_inserts_after_system_and_dedupes(monkeypatch):
    monkeypatch.setattr(memory, "recall_messages", lambda: [
        {"role": "assistant", "content": "User likes oat milk."},
        {"role": "user", "content": "hi"},  # already in the live request → deduped
    ])
    msgs = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "hi"}]
    out = memory.inject(msgs)
    assert out is not msgs  # caller's list not mutated
    assert out[0] == {"role": "system", "content": "prompt"}
    assert out[1]["role"] == "system" and "Recalled memory" in out[1]["content"]  # header
    assert out[2] == {"role": "assistant", "content": "User likes oat milk."}
    contents = [m["content"] for m in out]
    assert contents.count("hi") == 1  # the duplicate recalled "hi" was dropped


def test_as_text_handles_multimodal_and_last_user():
    assert memory._as_text("hello") == "hello"
    assert memory._as_text([{"type": "text", "text": "a"}, {"type": "image_url"}]) == "a"
    msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]}]
    assert memory.last_user_text(msgs) == "second"


def _recall_blob():
    return "\n".join(memory._as_text(m.get("content")) for m in memory.recall_messages())


def test_recall_reads_all_sections(tmp_path):
    _fresh(tmp_path)
    ctx = memory._get().context
    ctx.long_term_factual.append("memory", "The user is named Marko and likes coffee.")
    ctx.working.append("user", "my dog is Pixel")  # recent turn from another session
    blob = _recall_blob()
    assert "Marko" in blob and "coffee" in blob and "Pixel" in blob


def test_persists_across_reload(tmp_path):
    _fresh(tmp_path)
    memory._get().context.long_term_factual.append("memory", "Remembered fact 42.")
    memory._get().context.working.append("user", "favourite tea is genmaicha")
    # simulate a restart: drop the in-memory model, rebuild from the same data_dir
    memory._model = None
    blob = _recall_blob()
    assert "Remembered fact 42." in blob and "genmaicha" in blob
