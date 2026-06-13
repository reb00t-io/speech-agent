"""Tests for the generic tool-plugin seam (plugins.py + executor/schemas)."""
import os
import textwrap

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")

import src.plugins as plugins  # noqa: E402
from src.plugins import ToolRegistry, load_plugins  # noqa: E402
from src.tool_executor import execute_tool_call  # noqa: E402
from src.tool_schemas import ALL_TOOLS, get_tools  # noqa: E402


def _schema(name):
    return {
        "type": "function",
        "function": {"name": name, "description": "x", "parameters": {"type": "object", "properties": {}}},
    }


def test_builtin_tools_are_generic_only():
    names = {t["function"]["name"] for t in ALL_TOOLS}
    assert names == {"web_search", "fetch_url", "python", "bash", "get_logs", "publish_document"}
    # No domain/home tools baked into the core agent any more.
    assert not (names & {"home_capabilities", "control_device", "run_scene", "control_audio", "list_devices"})


def test_registry_add_and_lookup():
    reg = ToolRegistry()

    async def handler(session, args):
        return {"ok": True, "args": args}

    reg.add_tool(_schema("demo"), handler)
    assert reg.handler_for("demo") is handler
    assert reg.schemas[0]["function"]["name"] == "demo"


def test_registry_rejects_duplicate_and_bad_schema():
    reg = ToolRegistry()

    async def handler(session, args):
        return {}

    reg.add_tool(_schema("demo"), handler)
    try:
        reg.add_tool(_schema("demo"), handler)
        assert False, "expected duplicate error"
    except ValueError:
        pass
    try:
        reg.add_tool({"type": "function"}, handler)
        assert False, "expected missing-name error"
    except ValueError:
        pass


async def test_get_tools_merges_plugin_schemas(monkeypatch):
    reg = ToolRegistry()

    async def handler(session, args):
        return {}

    reg.add_tool(_schema("plugged"), handler)
    monkeypatch.setattr(plugins, "REGISTRY", reg)
    names = {t["function"]["name"] for t in get_tools()}
    assert "plugged" in names
    assert "web_search" in names  # builtins still present


async def test_execute_tool_call_dispatches_to_plugin(monkeypatch):
    reg = ToolRegistry()
    seen = {}

    async def handler(session, args):
        seen.update(args)
        return {"ok": True}

    reg.add_tool(_schema("plugged"), handler)
    monkeypatch.setattr(plugins, "REGISTRY", reg)

    out = await execute_tool_call(
        None, {"function": {"name": "plugged", "arguments": '{"a": 1}'}}
    )
    assert out == {"ok": True}
    assert seen == {"a": 1}


async def test_execute_tool_call_unknown_tool():
    out = await execute_tool_call(
        None, {"function": {"name": "nope", "arguments": "{}"}}
    )
    assert "error" in out and "Unknown tool" in out["error"]


def test_load_plugins_from_file(tmp_path, monkeypatch):
    plugin = tmp_path / "myplugin.py"
    plugin.write_text(textwrap.dedent(
        '''
        SCHEMA = {"type": "function", "function": {"name": "frob", "description": "x",
                  "parameters": {"type": "object", "properties": {}}}}

        async def _frob(session, args):
            return {"frobbed": True}

        def register(registry):
            registry.add_tool(SCHEMA, _frob)
        '''
    ))
    reg = ToolRegistry()
    load_plugins(env_value=str(plugin), registry=reg)
    assert reg.handler_for("frob") is not None
    assert reg.schemas[0]["function"]["name"] == "frob"


def test_load_plugins_empty_is_noop():
    reg = ToolRegistry()
    load_plugins(env_value="", registry=reg)
    assert reg.schemas == []
