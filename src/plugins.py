"""Optional tool plugins — the seam that keeps this agent generic.

The core agent ships only domain-agnostic tools (web search, fetch, python,
bash, get_logs, publish_document). A *deployment* injects its own domain tools
(e.g. a smart-home controller) without forking the agent, via the
``AGENT_PLUGINS`` env var: a comma-separated list of Python module names or
file paths. Each plugin exposes a module-level ``register(registry)`` function
and calls ``registry.add_tool(schema, handler)`` for every tool it contributes.

A tool handler has the signature::

    async def handler(session: aiohttp.ClientSession, args: dict) -> dict

where ``args`` is the already-parsed JSON arguments object and the return value
is the JSON-serialisable tool result. ``session`` is the shared
``aiohttp.ClientSession`` the executor uses for outbound calls.

With ``AGENT_PLUGINS`` unset the registry is empty and the agent runs fully
standalone — no domain coupling at all.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Any, dict], Awaitable[dict]]


class ToolRegistry:
    """Holds the extra tool schemas + handlers contributed by plugins."""

    def __init__(self) -> None:
        self._schemas: list[dict] = []
        self._handlers: dict[str, ToolHandler] = {}

    def add_tool(self, schema: dict, handler: ToolHandler) -> None:
        try:
            name = schema["function"]["name"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"tool schema missing function.name: {schema!r}") from exc
        if name in self._handlers:
            raise ValueError(f"duplicate tool registered: {name!r}")
        self._schemas.append(schema)
        self._handlers[name] = handler

    @property
    def schemas(self) -> list[dict]:
        return list(self._schemas)

    @property
    def handlers(self) -> dict[str, ToolHandler]:
        return dict(self._handlers)

    def handler_for(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)


# The process-wide registry. Populated once at startup by load_plugins().
REGISTRY = ToolRegistry()


def _load_one(spec: str, registry: ToolRegistry) -> None:
    spec = spec.strip()
    if not spec:
        return
    if spec.endswith(".py") or "/" in spec or os.sep in spec:
        path = Path(spec).expanduser()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load plugin from path: {spec}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)
    register = getattr(module, "register", None)
    if not callable(register):
        raise AttributeError(f"plugin {spec!r} has no callable register(registry)")
    register(registry)
    logger.info("loaded agent plugin: %s", spec)


def load_plugins(env_value: str | None = None, registry: ToolRegistry | None = None) -> ToolRegistry:
    """Load every plugin named in ``AGENT_PLUGINS`` into ``registry``.

    Returns the registry. Plugin failures are logged and re-raised — a
    misconfigured deployment should fail loudly, not silently lose its tools.
    """
    registry = registry if registry is not None else REGISTRY
    raw = env_value if env_value is not None else os.environ.get("AGENT_PLUGINS", "")
    for spec in raw.split(","):
        if spec.strip():
            _load_one(spec, registry)
    return registry
