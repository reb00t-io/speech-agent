"""Tests for src/memory.py — persistent long-term memory layer."""
import os

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")

import src.memory as memory  # noqa: E402


def _fresh(tmp_path):
    memory.reset_model()
    memory.DATA_DIR = str(tmp_path / "mem")
    memory.ENABLED = True
    # These tests exercise the Context (inject/recall sections), not the Qdrant
    # retrieval store; keep it off so the in-process reload test can reopen the
    # same data_dir without contending on the local Qdrant directory lock.
    memory.STORE_ENABLED = False
    memory.ORG_ENABLED = False


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


def test_recall_tool_schema_gating(monkeypatch):
    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "STORE_ENABLED", True)
    schema = memory.recall_tool_schema()
    assert schema and schema["function"]["name"] == "recall"

    monkeypatch.setattr(memory, "STORE_ENABLED", False)
    assert memory.recall_tool_schema() is None
    monkeypatch.setattr(memory, "STORE_ENABLED", True)
    monkeypatch.setattr(memory, "ENABLED", False)
    assert memory.recall_tool_schema() is None


def test_recall_executes_query_against_store(monkeypatch):
    from memorizer.store.qdrant_store import MemoryHit

    class FakeStore:
        def search(self, query, *, limit=5, member_id=None, role=None):
            return [MemoryHit(short_id="m1", kind="episode", scope="agent",
                              text=f"hit for {query}")]

    class FakeModel:
        memory = FakeStore()
        org_memory = None
        member_id = None
        role = None
        recall_limit = 5

    monkeypatch.setattr(memory, "ENABLED", True)
    monkeypatch.setattr(memory, "_model", FakeModel())
    out = memory.recall({"query": "coffee"})
    assert "hit for coffee" in out and "[m1]" in out


def test_recall_disabled_returns_unavailable(monkeypatch):
    monkeypatch.setattr(memory, "ENABLED", False)
    assert memory.recall({"query": "x"}) == "Memory is not available."


def test_persists_across_reload(tmp_path):
    _fresh(tmp_path)
    memory._get().context.long_term_factual.append("memory", "Remembered fact 42.")
    memory._get().context.working.append("user", "favourite tea is genmaicha")
    # simulate a restart: drop the in-memory model, rebuild from the same data_dir
    memory.reset_model()
    blob = _recall_blob()
    assert "Remembered fact 42." in blob and "genmaicha" in blob
