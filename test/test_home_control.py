"""Tests for home-hub control tools (agent → hub API)."""
import os

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ["HUB_API_URL"] = "http://hub:31060"
os.environ["AGENT_API_KEY"] = "agent-secret"

import src.home_control as hc  # noqa: E402
from src.tool_executor import execute_tool_call  # noqa: E402


class FakeResp:
    def __init__(self, status, data):
        self.status = status
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self, content_type=None):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    async def text(self):
        return "rawbody"


class FakeSession:
    """Duck-typed aiohttp.ClientSession for the hub calls."""

    def __init__(self, responder):
        self.calls = []
        self._responder = responder

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json})
        return self._responder(method, url, json)


def _ok(data):
    return FakeSession(lambda m, u, j: FakeResp(200, data))


async def test_control_device_posts_action_with_params_and_auth():
    sess = _ok({"ok": True, "id": "wz-gross", "action": "position"})
    out = await hc.control_device(sess, "wz-gross", "position", {"value": 30})
    assert out["ok"] is True
    call = sess.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "http://hub:31060/api/devices/wz-gross/position"
    assert call["json"] == {"value": 30}
    assert call["headers"]["Authorization"] == "Bearer agent-secret"


async def test_control_device_requires_id_and_action():
    sess = _ok({})
    assert "error" in await hc.control_device(sess, "", "open", {})
    assert sess.calls == []


async def test_list_devices_and_capabilities_and_status_get():
    sess = _ok({"devices": [{"id": "wc"}]})
    assert (await hc.list_devices(sess))["devices"][0]["id"] == "wc"
    assert sess.calls[0]["method"] == "GET"
    assert sess.calls[0]["url"].endswith("/api/devices")

    sess = _ok({"device_types": {}})
    await hc.home_capabilities(sess)
    assert sess.calls[0]["url"].endswith("/api/manifest")

    sess = _ok({"presence": {}})
    await hc.home_status(sess)
    assert sess.calls[0]["url"].endswith("/api/home/status")


async def test_run_scene_and_audio():
    sess = _ok({"ok": True})
    await hc.run_scene(sess, "night")
    assert sess.calls[0]["url"].endswith("/api/scenes/night")

    sess = _ok({"ok": True, "playing": "Flux FM"})
    await hc.control_audio(sess, "play", station="Flux FM")
    assert sess.calls[0]["url"].endswith("/api/audio/play")
    assert sess.calls[0]["json"] == {"station": "Flux FM"}

    sess = _ok({"ok": True})
    await hc.control_audio(sess, "volume", volume=0.4)
    assert sess.calls[0]["json"] == {"volume": 0.4}

    assert "error" in await hc.control_audio(_ok({}), "explode")


async def test_http_error_is_reported():
    sess = FakeSession(lambda m, u, j: FakeResp(401, {"error": "api key required"}))
    out = await hc.control_device(sess, "wc", "open", {})
    assert "error" in out and "401" in out["error"]


async def test_network_failure_is_caught():
    def boom(m, u, j):
        raise OSError("connection refused")

    sess = FakeSession(boom)
    out = await hc.list_devices(sess)
    assert "error" in out and "unreachable" in out["error"]


async def test_execute_tool_call_routes_home_tools():
    sess = _ok({"ok": True, "id": "wc", "action": "close"})
    tool_call = {"function": {"name": "control_device",
                              "arguments": '{"device_id": "wc", "action": "close"}'}}
    out = await execute_tool_call(sess, tool_call)
    assert out["ok"] is True
    assert sess.calls[0]["url"].endswith("/api/devices/wc/close")
