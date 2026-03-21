#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.widgets import Box, Frame, Label, RadioList


def run_gh_json(args: list[str]) -> Any:
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh command failed")
    return json.loads(proc.stdout)


def list_running_actions() -> list[dict[str, Any]]:
    return run_gh_json(
        [
            "run",
            "list",
            "--status",
            "in_progress",
            "--json",
            "databaseId,workflowName,displayTitle,headBranch,event,createdAt,url",
            "--limit",
            "30",
        ]
    )


def list_recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    return run_gh_json(
        [
            "run",
            "list",
            "--json",
            "databaseId,workflowName,displayTitle,status,conclusion,headBranch,createdAt",
            "--limit",
            str(limit),
        ]
    )


def run_state(run: dict[str, Any]) -> str:
    status = str(run.get("status") or "").lower()
    conclusion = str(run.get("conclusion") or "").lower()
    if status != "completed":
        return "running"
    if conclusion in {"success", "neutral", "skipped"}:
        return "completed"
    return "failed"


def run_state_emoji(state: str) -> str:
    if state == "running":
        return "⏳"
    if state == "completed":
        return "✅"
    return "❌"


def format_run_when(created_at: str | None) -> str:
    if not created_at:
        return "<no time>  "
    try:
        run_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return created_at
    now_local = datetime.now().astimezone()
    delta = now_local - run_dt
    if delta.total_seconds() >= 0 and delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() // 60)
        pad = " " if minutes < 10 else ""
        return f" {pad}-{minutes}m"
    if run_dt.date() == now_local.date():
        return run_dt.strftime("%H:%M")
    return run_dt.strftime("%Y-%m-%d")


def select_run(runs: list[dict[str, Any]]) -> int | None:
    values: list[tuple[int, str]] = []
    for item in runs:
        run_id = int(item["databaseId"])
        workflow = item.get("workflowName") or "<unknown workflow>"
        title = item.get("displayTitle") or "<no title>"
        branch = item.get("headBranch") or "<no branch>"
        created = item.get("createdAt") or "<no time>"
        label = f"{workflow} | {title} | {branch} | {created}"
        values.append((run_id, label))

    radio = RadioList(values=values)
    bindings = KeyBindings()

    @bindings.add("enter", eager=True)
    @bindings.add("c-m", eager=True)
    def _accept(event) -> None:  # type: ignore[no-untyped-def]
        event.app.exit(result=radio.current_value)

    @bindings.add("escape")
    def _cancel(event) -> None:  # type: ignore[no-untyped-def]
        event.app.exit(result=None)

    root = HSplit(
        [
            Label(text="Running GitHub Actions\nSelect a run and press Enter. Press Esc to cancel."),
            Box(body=Frame(radio), padding_top=1),
        ]
    )
    app = Application(layout=Layout(root), key_bindings=bindings, full_screen=False)
    return app.run()


def show_run_details(run_id: int) -> int:
    proc = subprocess.run(["gh", "run", "view", str(run_id)], check=False)
    return proc.returncode


def main() -> int:
    if shutil.which("gh") is None:
        print("Error: gh CLI is not installed or not on PATH.", file=sys.stderr)
        return 1

    try:
        recent_runs = list_recent_runs(limit=10)
        runs = list_running_actions()
    except Exception as exc:
        print(f"Error listing running actions: {exc}", file=sys.stderr)
        return 1

    print("Last 10 runs:")
    for item in recent_runs:
        workflow = item.get("workflowName") or "<unknown workflow>"
        title = item.get("displayTitle") or "<no title>"
        state = run_state(item)
        emoji = run_state_emoji(state)
        when = format_run_when(item.get("createdAt"))
        print(f"{when} {emoji} {workflow} | {title}")
    print()

    print(f"Running actions: {len(runs)}")
    for item in runs:
        run_id = item.get("databaseId")
        workflow = item.get("workflowName") or "<unknown workflow>"
        title = item.get("displayTitle") or "<no title>"
        print(f"- {run_id}: {workflow} | {title}")

    if not runs:
        return 0

    selected = select_run(runs)
    if selected is None:
        print("No run selected.")
        return 0

    print(f"\nShowing details for run {selected}...\n")
    return show_run_details(selected)


if __name__ == "__main__":
    raise SystemExit(main())
