"""XME hook installer — xme hook install."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path


def _python() -> str:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        c = Path(venv) / "bin" / "python"
        if c.exists():
            return str(c)
    return sys.executable


def install(repo_path: str, *, dry_run: bool = False) -> dict[str, list[str]]:
    repo = Path(repo_path).resolve()
    python = _python()
    cmd_base = f"{python} -m xme.hooks.handler --repo-path {repo}"
    written: dict[str, list[str]] = {"kiro": [], "claude": [], "git": []}

    # Kiro hooks
    kiro_dir = repo / ".kiro" / "hooks"
    if not dry_run:
        kiro_dir.mkdir(parents=True, exist_ok=True)

    kiro_hooks = [
        ("xme-session-end.json", {
            "name": "XME Session End",
            "version": "1.0.0",
            "description": "Persist memory at end of every Kiro session.",
            "when": {"type": "agentStop"},
            "then": {"type": "runCommand",
                     "command": f"{cmd_base} --event agentStop", "timeout": 15},
        }),
        ("xme-record-turn.json", {
            "name": "XME Record Prompt",
            "version": "1.0.0",
            "description": "Record user prompts in XME episodic journal.",
            "when": {"type": "promptSubmit"},
            "then": {"type": "runCommand",
                     "command": f"{cmd_base} --event promptSubmit", "timeout": 5},
        }),
        ("xme-record-tool.json", {
            "name": "XME Record Tool",
            "version": "1.0.0",
            "description": "Record tool calls in XME episodic journal.",
            "when": {"type": "postToolUse", "toolTypes": ["read", "write", "shell", "web"]},
            "then": {"type": "runCommand",
                     "command": f"{cmd_base} --event postToolUse", "timeout": 5},
        }),
    ]
    for fname, content in kiro_hooks:
        f = kiro_dir / fname
        if not dry_run:
            f.write_text(json.dumps(content, indent=2))
        written["kiro"].append(str(f))

    # Claude Code settings
    claude_dir = repo / ".claude"
    claude_settings = claude_dir / "settings.json"
    if not dry_run:
        claude_dir.mkdir(exist_ok=True)

    xme_claude = {
        "Stop": [{"matcher": "", "hooks": [{"type": "command",
            "command": f"{cmd_base} --event Stop"}]}],
        "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command",
            "command": f"{cmd_base} --event UserPromptSubmit"}]}],
        "PostToolUse": [{"matcher": "", "hooks": [{"type": "command",
            "command": f"{cmd_base} --event PostToolUse"}]}],
    }
    existing: dict = {}
    if claude_settings.exists():
        try:
            existing = json.loads(claude_settings.read_text())
        except Exception:
            existing = {}
    existing_hooks = existing.get("hooks", {})
    for event, entries in xme_claude.items():
        if event not in existing_hooks:
            existing_hooks[event] = entries
        else:
            cmds = [h["command"] for b in existing_hooks[event] for h in b.get("hooks", [])]
            if not any("xme.hooks.handler" in c for c in cmds):
                existing_hooks[event].extend(entries)
    existing["hooks"] = existing_hooks
    if not dry_run:
        claude_settings.write_text(json.dumps(existing, indent=2))
    written["claude"].append(str(claude_settings))

    # Git post-commit hook
    git_hook = repo / ".git" / "hooks" / "post-commit"
    git_content = f"""#!/bin/sh
# XME: capture commit context
{python} -m xme.hooks.handler --event post-commit --repo-path {repo} &
"""
    if not dry_run and (repo / ".git").exists():
        git_hook.write_text(git_content)
        git_hook.chmod(0o755)
    written["git"].append(str(git_hook))

    return written


def uninstall(repo_path: str) -> dict[str, list[str]]:
    repo = Path(repo_path).resolve()
    removed: dict[str, list[str]] = {"kiro": [], "claude": [], "git": []}
    for name in ("xme-session-end.json", "xme-record-turn.json", "xme-record-tool.json"):
        f = repo / ".kiro" / "hooks" / name
        if f.exists():
            f.unlink()
            removed["kiro"].append(str(f))
    claude_settings = repo / ".claude" / "settings.json"
    if claude_settings.exists():
        try:
            data = json.loads(claude_settings.read_text())
            hooks = data.get("hooks", {})
            for event in list(hooks.keys()):
                hooks[event] = [b for b in hooks[event] if not any(
                    "xme.hooks.handler" in h.get("command", "")
                    for h in b.get("hooks", []))]
                if not hooks[event]:
                    del hooks[event]
            data["hooks"] = hooks
            claude_settings.write_text(json.dumps(data, indent=2))
            removed["claude"].append(str(claude_settings))
        except Exception:
            pass
    return removed
