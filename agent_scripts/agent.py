#!/usr/bin/env python3
"""Run an agent CLI (Claude Code or Codex) with a prompt."""

from __future__ import annotations

import argparse
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class Agent(ABC):
    @abstractmethod
    def run(self, repo_dir: Path, prompt: str) -> None:
        raise NotImplementedError


class CodexAgent(Agent):
    def __init__(
        self,
        *,
        output_last_message: str | None = None,
        full_auto: bool = False,
    ) -> None:
        self.output_last_message = output_last_message
        self.full_auto = full_auto

    def run(self, repo_dir: Path, prompt: str) -> None:
        cmd = ["codex", "exec", "-C", str(repo_dir), "-"]
        if self.output_last_message:
            cmd.extend(["--output-last-message", self.output_last_message])
        if self.full_auto:
            cmd.append("--full-auto")
        subprocess.run(cmd, input=prompt, text=True, check=True)


class ClaudeAgent(Agent):
    def __init__(
        self,
        *,
        append_system_prompt_file: str | None = None,
        allowed_tools: str = "Read,Edit,Bash",
        no_allowed_tools: bool = False,
    ) -> None:
        self.append_system_prompt_file = append_system_prompt_file
        self.allowed_tools = allowed_tools
        self.no_allowed_tools = no_allowed_tools

    def run(self, repo_dir: Path, prompt: str) -> None:
        cmd = ["claude", "-p"]

        if self.append_system_prompt_file:
            system_path = Path(self.append_system_prompt_file).expanduser().resolve()
            if system_path.exists():
                cmd.extend(["--append-system-prompt-file", str(system_path)])

        if not self.no_allowed_tools:
            cmd.extend(["--allowedTools", self.allowed_tools])

        cmd.append(prompt)
        subprocess.run(cmd, cwd=repo_dir, check=True)


def get_agent(
    tool: str,
    *,
    output_last_message: str | None = None,
    full_auto: bool = False,
    append_system_prompt_file: str | None = None,
    allowed_tools: str = "Read,Edit,Bash",
    no_allowed_tools: bool = False,
) -> Agent:
    if tool == "codex":
        return CodexAgent(
            output_last_message=output_last_message,
            full_auto=full_auto,
        )

    if tool == "claude":
        return ClaudeAgent(
            append_system_prompt_file=append_system_prompt_file,
            allowed_tools=allowed_tools,
            no_allowed_tools=no_allowed_tools,
        )

    raise ValueError(f"Unknown tool: {tool}")


def parse_agent_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agent CLI (Claude Code or Codex) with a prompt."
    )
    parser.add_argument(
        "tool",
        choices=["claude", "codex"],
        help="Which CLI to use.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Target repo path (default: current directory).",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt string to send to the agent.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a prompt file.",
    )

    codex = parser.add_argument_group("codex")
    codex.add_argument(
        "--output-last-message",
        default=None,
        help="Optional path for Codex last message output.",
    )
    codex.add_argument(
        "--full-auto",
        action="store_true",
        help="Enable Codex full-auto preset.",
    )

    claude = parser.add_argument_group("claude")
    claude.add_argument(
        "--append-system-prompt-file",
        default=".automation/claude-system.txt",
        help="System prompt additions file (optional).",
    )
    claude.add_argument(
        "--allowed-tools",
        default="Read,Edit,Bash",
        help="Allowed tools for headless mode.",
    )
    claude.add_argument(
        "--no-allowed-tools",
        action="store_true",
        help="Disable --allowedTools flag.",
    )

    args = parser.parse_args(argv)
    if (args.prompt and args.prompt_file) or (not args.prompt and not args.prompt_file):
        parser.error("Provide exactly one of --prompt or --prompt-file.")
    return args


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    return prompt_path.read_text(encoding="utf-8").strip()


def run_from_cli(argv: list[str] | None = None) -> None:
    args = parse_agent_args(argv)
    repo_dir = Path(args.repo).expanduser().resolve()
    prompt = load_prompt(args)
    agent = get_agent(
        args.tool,
        output_last_message=args.output_last_message,
        full_auto=args.full_auto,
        append_system_prompt_file=args.append_system_prompt_file,
        allowed_tools=args.allowed_tools,
        no_allowed_tools=args.no_allowed_tools,
    )
    agent.run(repo_dir, prompt)


def main() -> int:
    run_from_cli(sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
