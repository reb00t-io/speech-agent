from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import signal
import sys
import tempfile
from typing import Any

import aiohttp

try:
    from .runtime_logs import get_backend_logs, normalize_log_limit
    from .web_tools import fetch_url, normalize_max_chars, normalize_max_results, web_search
except ImportError:
    from runtime_logs import get_backend_logs, normalize_log_limit
    from web_tools import fetch_url, normalize_max_chars, normalize_max_results, web_search

DEFAULT_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 12000
KILL_GRACE_SECONDS = 15

logger = logging.getLogger(__name__)


def normalize_timeout_seconds(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(value, (int, float)):
        return max(1, min(120, int(value)))
    return DEFAULT_TIMEOUT_SECONDS


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _decode_and_truncate(stdout_b: bytes, stderr_b: bytes) -> tuple[str, str, bool, bool]:
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    stdout, stdout_truncated = _truncate(stdout)
    stderr, stderr_truncated = _truncate(stderr)
    return stdout, stderr, stdout_truncated, stderr_truncated


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


async def _collect_output_and_reap(process: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
    try:
        return await asyncio.wait_for(process.communicate(), timeout=KILL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        try:
            await asyncio.wait_for(process.wait(), timeout=KILL_GRACE_SECONDS)
        except asyncio.TimeoutError:
            pass
        return b"", b""


async def run_bash(command: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    command = (command or "").strip()
    if not command:
        return {"error": "Missing required argument: command"}

    timeout_seconds = normalize_timeout_seconds(timeout_seconds)
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        _kill_process_group(process)
        stdout_b, stderr_b = await _collect_output_and_reap(process)
        stderr_b += f"\n[killed after {timeout_seconds}s timeout]".encode()

    stdout, stderr, stdout_truncated, stderr_truncated = _decode_and_truncate(stdout_b, stderr_b)
    return {
        "command": command,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


_SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "en_US.UTF-8"}
_SANDBOX_MEM_BYTES = 256 * 1024 * 1024  # 256 MB


def _sandbox_limits() -> None:
    """Called in the child process before exec — drops env already handled by env=, apply memory cap."""
    resource.setrlimit(resource.RLIMIT_AS, (_SANDBOX_MEM_BYTES, _SANDBOX_MEM_BYTES))


async def run_python(code: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    code = (code or "").strip()
    if not code:
        return {"error": "Missing required argument: code"}

    timeout_seconds = normalize_timeout_seconds(timeout_seconds)
    with tempfile.TemporaryDirectory() as tmpdir:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env=_SANDBOX_ENV,
            cwd=tmpdir,
            preexec_fn=_sandbox_limits,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_process_group(process)
            stdout_b, stderr_b = await _collect_output_and_reap(process)
            stderr_b += f"\n[killed after {timeout_seconds}s timeout]".encode()

    stdout, stderr, stdout_truncated, stderr_truncated = _decode_and_truncate(stdout_b, stderr_b)
    return {
        "code": code,
        "python_executable": sys.executable,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


async def execute_tool_call(session: aiohttp.ClientSession, tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function") or {}
    fn_name = fn.get("name")
    raw_arguments = fn.get("arguments") or "{}"
    try:
        fn_args = json.loads(raw_arguments)
        if not isinstance(fn_args, dict):
            raise ValueError("Tool arguments must be a JSON object")
    except Exception as exc:
        return {"error": f"Invalid tool arguments for {fn_name}: {exc}"}

    fn_args.pop("user_message", None)

    if fn_name == "web_search":
        query = str(fn_args.get("query") or "").strip()
        if not query:
            return {"error": "Missing required argument: query"}
        return await web_search(session, query, normalize_max_results(fn_args.get("max_results", 5)))

    if fn_name == "fetch_url":
        url = str(fn_args.get("url") or "").strip()
        if not url:
            return {"error": "Missing required argument: url"}
        return await fetch_url(session, url, normalize_max_chars(fn_args.get("max_chars", 8000)))

    if fn_name == "bash":
        command = str(fn_args.get("command") or "").strip()
        if not command:
            return {"error": "Missing required argument: command"}
        return await run_bash(command, normalize_timeout_seconds(fn_args.get("timeout_seconds", 20)))

    if fn_name == "python":
        code = str(fn_args.get("code") or "").strip()
        if not code:
            return {"error": "Missing required argument: code"}
        return await run_python(code, normalize_timeout_seconds(fn_args.get("timeout_seconds", 20)))

    if fn_name == "get_logs":
        system = str(fn_args.get("system") or "").strip()
        if system == "backend":
            limit = normalize_log_limit(fn_args.get("limit", 50))
            logger.info("get_logs requested for backend logs (limit=%s)", limit)
            return get_backend_logs(limit)
        if system == "frontend":
            logger.info("get_logs requested for frontend logs via client bridge")
            return {"error": "Frontend logs must be requested through the client bridge"}
        logger.warning("get_logs requested with invalid system=%r", system)
        return {"error": "Missing or invalid required argument: system"}

    return {"error": f"Unknown tool: {fn_name}"}
