"""Home-hub control — lets the agent operate the house through the hub API.

The agent is the brain; the hub (netmon) is its hands. These helpers call the
hub's component API (``/api/manifest``, ``/api/devices``, ``/api/scenes``,
``/api/audio``, ``/api/home/status``) using the agent's service credential
(``AGENT_API_KEY``). See netmon's docs/AGENT_ARCHITECTURE.md.

Everything goes over HTTP through the same API the web UI uses — one contract,
one auth point, fully auditable.
"""
from __future__ import annotations

import os
from typing import Any

import aiohttp

HUB_API_URL = os.environ.get("HUB_API_URL", "http://192.168.178.45:31060").rstrip("/")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _headers(json_body: bool = False) -> dict[str, str]:
    h: dict[str, str] = {}
    if AGENT_API_KEY:
        h["Authorization"] = f"Bearer {AGENT_API_KEY}"
    if json_body:
        h["Content-Type"] = "application/json"
    return h


async def _request(
    session: aiohttp.ClientSession, method: str, path: str, body: Any = None
) -> dict[str, Any]:
    url = HUB_API_URL + path
    try:
        async with session.request(
            method, url, headers=_headers(body is not None),
            json=body if body is not None else None, timeout=_TIMEOUT,
        ) as resp:
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = {"raw": (await resp.text())[:500]}
            if resp.status >= 400:
                return {"error": f"hub {method} {path} -> {resp.status}", "detail": data}
            return data if isinstance(data, dict) else {"result": data}
    except Exception as exc:  # network / hub down
        return {"error": f"hub unreachable ({HUB_API_URL}): {type(exc).__name__}: {exc}"}


# --- tool entry points -------------------------------------------------------

async def home_capabilities(session: aiohttp.ClientSession) -> dict[str, Any]:
    """What the house can do: device types/instances+actions, scenes, stations."""
    return await _request(session, "GET", "/api/manifest")


async def home_status(session: aiohttp.ClientSession) -> dict[str, Any]:
    """Who's home + internet/network health + anomalies."""
    return await _request(session, "GET", "/api/home/status")


async def list_devices(session: aiohttp.ClientSession) -> dict[str, Any]:
    """All devices with live status (position/state)."""
    return await _request(session, "GET", "/api/devices")


async def control_device(
    session: aiohttp.ClientSession, device_id: str, action: str, params: dict | None = None
) -> dict[str, Any]:
    """Run an action (open/close/stop/position/on/off/…) on a device by id."""
    if not device_id or not action:
        return {"error": "device_id and action are required"}
    return await _request(session, "POST", f"/api/devices/{device_id}/{action}", params or {})


async def run_scene(session: aiohttp.ClientSession, name: str) -> dict[str, Any]:
    """Run a named scene (morning/evening/night/leaving/deter)."""
    if not name:
        return {"error": "scene name is required"}
    return await _request(session, "POST", f"/api/scenes/{name}", {})


async def control_audio(
    session: aiohttp.ClientSession, action: str,
    station: str | None = None, volume: float | None = None,
) -> dict[str, Any]:
    """Audio: play a station, stop, or set volume (0..1)."""
    if action == "play":
        return await _request(session, "POST", "/api/audio/play", {"station": station})
    if action == "stop":
        return await _request(session, "POST", "/api/audio/stop", {})
    if action == "volume":
        return await _request(session, "POST", "/api/audio/volume", {"volume": volume})
    return {"error": f"unknown audio action {action!r} (use play|stop|volume)"}
