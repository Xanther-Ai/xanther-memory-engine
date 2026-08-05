"""XME CLI — xme <command>."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xme", description="Xanther Memory Engine")
    sub = p.add_subparsers(dest="command")

    # start
    st = sub.add_parser("start", help="Initialize XME for a project")
    st.add_argument("project_id", help="Project identifier")
    st.add_argument("--path", default=".", help="Repo path (default: cwd)")
    st.add_argument("--fallback", action="store_true", help="SQLite-only mode")

    # add
    add = sub.add_parser("add", help="Add content to memory (UPSERT)")
    add.add_argument("project_id")
    add.add_argument("user_id")
    add.add_argument("content")
    add.add_argument("--type", dest="fact_type", default=None,
                     choices=["decision","attempt","preference","convention","entity"])

    # search
    srch = sub.add_parser("search", help="Search memory")
    srch.add_argument("project_id")
    srch.add_argument("query")
    srch.add_argument("--limit", type=int, default=10)
    srch.add_argument("--layer", choices=["episodic","facts","context"])

    # facts
    facts = sub.add_parser("facts", help="List facts for a project")
    facts.add_argument("project_id")
    facts.add_argument("--type", dest="fact_type", default=None)
    facts.add_argument("--limit", type=int, default=20)

    # stats
    stats = sub.add_parser("stats", help="Show memory stats")
    stats.add_argument("project_id")

    # export
    exp = sub.add_parser("export", help="Export memory")
    exp.add_argument("project_id")
    exp.add_argument("--format", choices=["obsidian","wiki","graphify"], default="obsidian")
    exp.add_argument("--output", default=None)

    # dashboard
    dash = sub.add_parser("dashboard", help="Start the memory dashboard (port 8001)")
    dash.add_argument("--port", type=int, default=8001)

    # hook
    hook = sub.add_parser("hook", help="Manage IDE hooks")
    hook_sub = hook.add_subparsers(dest="hook_command")
    hi = hook_sub.add_parser("install")
    hi.add_argument("path", nargs="?", default=".")
    hi.add_argument("--dry-run", action="store_true")
    hu = hook_sub.add_parser("uninstall")
    hu.add_argument("path", nargs="?", default=".")

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if not args.command:
        p.print_help()
        sys.exit(1)

    if args.command == "start":
        asyncio.run(_cmd_start(args))
    elif args.command == "add":
        asyncio.run(_cmd_add(args))
    elif args.command == "search":
        asyncio.run(_cmd_search(args))
    elif args.command == "facts":
        asyncio.run(_cmd_facts(args))
    elif args.command == "stats":
        asyncio.run(_cmd_stats(args))
    elif args.command == "export":
        asyncio.run(_cmd_export(args))
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "hook":
        _cmd_hook(args)
    else:
        p.print_help()


async def _cmd_start(args) -> None:
    import os
    from xme.config import XMESettings
    from xme.engine import MemoryEngine
    repo = Path(args.path).resolve()
    settings = XMESettings(
        sqlite_path=str(repo / ".xanther" / "xme.db"),
        fallback_mode=args.fallback,
    )
    async with MemoryEngine(settings) as engine:
        stats = await engine.stats(args.project_id)
        print(f"✓ XME initialized for '{args.project_id}'")
        print(f"  DB: {settings.resolved_sqlite_path(str(repo))}")
        print(f"  Facts: {stats['total_facts']}  Episodes: {stats['total_episodes']}")
        print(f"  Fallback mode: {args.fallback}")


async def _cmd_add(args) -> None:
    from xme.engine import get_engine
    engine = await get_engine()
    result = await engine.add(
        content=args.content,
        project_id=args.project_id,
        user_id=args.user_id,
        fact_type=args.fact_type,
        confidence="EXPLICIT",
    )
    print(f"✓ {result.action}: {result.fact_id}")


async def _cmd_search(args) -> None:
    from xme.engine import get_engine
    engine = await get_engine()
    layers = [args.layer] if args.layer else None
    results = await engine.search(args.query, args.project_id, layers=layers, limit=args.limit)
    all_r = results.all_results
    if not all_r:
        print("No results found.")
        return
    print(f"Results for '{args.query}' ({len(all_r)}):\n")
    for r in all_r:
        print(f"  [{r.layer.upper()}] {r.summary[:80]}")
        if r.highlight:
            print(f"    {r.highlight[:100]}")
        print(f"    score={r.score:.3f}  id={r.item_id}")
        print()


async def _cmd_facts(args) -> None:
    from xme.engine import get_engine
    engine = await get_engine()
    facts = await engine.facts.list_facts(
        project_id=args.project_id,
        fact_type=args.fact_type,
        limit=args.limit,
    )
    if not facts:
        print("No facts found.")
        return
    print(f"Facts for '{args.project_id}' ({len(facts)}):\n")
    for f in facts:
        print(f"  [{f.fact_type.upper()}] {f.title}")
        print(f"    {f.content[:100]}")
        print(f"    confidence={f.confidence}  id={f.fact_id[:8]}...")
        print()


async def _cmd_stats(args) -> None:
    from xme.engine import get_engine
    engine = await get_engine()
    s = await engine.stats(args.project_id)
    print(f"XME Stats — {args.project_id}")
    print(f"  Facts:    {s['total_facts']}")
    print(f"  Episodes: {s['total_episodes']}")
    print(f"  Users:    {s['active_users']}  {s['users']}")
    print(f"  Last:     {s['last_activity'] or '—'}")
    if s.get("fact_types"):
        print("  By type:")
        for t, c in s["fact_types"].items():
            print(f"    {t}: {c}")


async def _cmd_export(args) -> None:
    from xme.engine import get_engine
    from xme.export import run_export
    engine = await get_engine()
    path = await run_export(engine, args.project_id, fmt=args.format, output_dir=args.output)
    print(f"✓ Exported to: {path}")


def _cmd_dashboard(args) -> None:
    import uvicorn
    from xme.server.dashboard import create_dashboard_app
    app = create_dashboard_app()
    print(f"Starting XME dashboard on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def _cmd_hook(args) -> None:
    from xme.hooks.installer import install, uninstall
    cmd = getattr(args, "hook_command", None)
    path = getattr(args, "path", ".")
    repo = str(Path(path).resolve())
    if cmd == "install":
        dry = getattr(args, "dry_run", False)
        written = install(repo, dry_run=dry)
        prefix = "[DRY RUN] Would write" if dry else "✓ Installed"
        print(f"\n{prefix} XME hooks:")
        for cat, files in written.items():
            for f in files:
                print(f"  {cat}: {f}")
        if not dry:
            print("\nHooks active. XME will auto-capture every session.")
    elif cmd == "uninstall":
        removed = uninstall(repo)
        print("✓ Removed XME hooks:")
        for files in removed.values():
            for f in files:
                print(f"  {f}")
    else:
        print("Usage: xme hook install|uninstall [path]")


if __name__ == "__main__":
    main()
