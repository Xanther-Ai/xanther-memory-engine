"""XME hook handler — CLI entrypoint for Kiro/Claude Code hooks.

Design constraint: must complete in < 2 seconds.
All hooks write buffer files only — zero imports of xme/xce packages.
A background process or the next `xme start` call drains the buffer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="")
    parser.add_argument("--repo-path", default="")
    parser.add_argument("--content", default="")
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--outcome", default="unknown")
    args = parser.parse_args()

    stdin_data: dict = {}
    if not sys.stdin.isatty():
        try:
            raw = sys.stdin.read(4096)
            if raw.strip():
                stdin_data = json.loads(raw)
        except Exception:
            pass

    event = (args.event
             or stdin_data.get("event", "")
             or stdin_data.get("hook_event_name", ""))
    repo_path = (args.repo_path
                 or stdin_data.get("repo_path", "")
                 or stdin_data.get("cwd", "")
                 or _git_root())

    if not event or not repo_path or not Path(repo_path).is_dir():
        sys.exit(0)

    buf_dir = Path(repo_path) / ".xanther" / "turns"
    buf_dir.mkdir(parents=True, exist_ok=True)

    event_n = event.lower().replace("-", "_").replace(" ", "_")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fname = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}.json"

    try:
        if event_n in ("promptsubmit", "userpromptsubmit"):
            content = (args.content
                       or stdin_data.get("content", "")
                       or stdin_data.get("prompt", ""))
            if content:
                _write(buf_dir / fname, {
                    "type": "turn", "role": "user",
                    "content": content[:2000], "ts": ts,
                })

        elif event_n in ("posttooluse", "post_tool_use"):
            tool = (args.tool_name
                    or stdin_data.get("tool_name", "")
                    or stdin_data.get("tool", ""))
            if tool and not tool.startswith("xme_") and not tool.startswith("xce_"):
                result = str(stdin_data.get("output", ""))[:200]
                _write(buf_dir / fname, {
                    "type": "turn", "role": "tool",
                    "content": f"[{tool}] {result}", "ts": ts,
                })

        elif event_n in ("agentstop", "stop"):
            # Write a session-end marker — background drain picks this up
            _write(buf_dir / fname, {
                "type": "session_end",
                "summary": args.summary or stdin_data.get("summary", ""),
                "outcome": args.outcome or stdin_data.get("outcome", "unknown"),
                "user_id": stdin_data.get("user_id", os.environ.get("USER", "unknown")),
                "ts": ts,
            })

        elif event_n == "post_commit":
            import subprocess
            try:
                r = subprocess.run(
                    ["git", "-C", repo_path, "log", "-1", "--pretty=%B"],
                    capture_output=True, text=True, timeout=2,
                )
                msg = r.stdout.strip()
                if msg:
                    _write(buf_dir / fname, {
                        "type": "turn", "role": "note",
                        "content": f"[git commit] {msg}", "ts": ts,
                    })
            except Exception:
                pass

    except Exception:
        pass

    sys.exit(0)


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _git_root() -> str:
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return os.getcwd()


if __name__ == "__main__":
    main()
