"""CLI tool for context-drift-analyzer.

Commands:
    context-drift-analyzer status  — Show session memory status
    context-drift-analyzer reset   — Delete session memory file
    context-drift-analyzer history — Show drift history
    context-drift-analyzer freeze  — Freeze context (prevent modifications)
    context-drift-analyzer unfreeze — Unfreeze context

Usage:
    python -m context_drift_analyzer.cli.main status
    python -m context_drift_analyzer.cli.main status --file .session_memory
    python -m context_drift_analyzer.cli.main reset
    python -m context_drift_analyzer.cli.main history --last 10
"""

from __future__ import annotations

import argparse
import json
import sys

from context_drift_analyzer.persistence.session_memory import SessionMemoryStore


def cmd_status(args: argparse.Namespace) -> None:
    """Show session memory status."""
    store = SessionMemoryStore(path=args.file)

    if not store.exists():
        print(f"No session memory found at: {args.file}")
        return

    data = store.load()
    print(f"Session Memory: {args.file}")
    print(f"  Sessions:       {data.session_count}")
    print(f"  Context frozen: {data.context_frozen}")
    print(f"  Total sessions: {len(data.sessions)}")

    for s in data.sessions:
        sn = s.get("session_number", "?")
        status = s.get("status", "?")
        exchanges = s.get("exchanges", [])
        summary = s.get("summary")
        score = s.get("final_drift_score")
        print(f"\n  Session {sn} [{status}] — {len(exchanges)} exchanges")
        if score is not None:
            print(f"    Final drift: {score:.1f}/100")
        if summary:
            print(f"    Summary: {summary[:80]}")
        if exchanges:
            last_ex = exchanges[-1]
            print(f"    Last exchange: Q: {last_ex.get('user', '')[:50]}")
            print(f"                   A: {last_ex.get('assistant', '')[:50]}")
            print(f"                   Score: {last_ex.get('score', '?')}/100 ({last_ex.get('verdict', '?')})")


def cmd_reset(args: argparse.Namespace) -> None:
    """Delete session memory file."""
    store = SessionMemoryStore(path=args.file)
    if store.delete():
        print(f"Deleted: {args.file}")
    else:
        print(f"No session memory found at: {args.file}")


def cmd_history(args: argparse.Namespace) -> None:
    """Show drift history."""
    store = SessionMemoryStore(path=args.file)

    if not store.exists():
        print(f"No session memory found at: {args.file}")
        return

    data = store.load()

    if not data.sessions:
        print("No drift history recorded.")
        return

    # Flatten all exchanges across sessions for display
    all_exchanges = []
    for s in data.sessions:
        for e in s.get("exchanges", []):
            all_exchanges.append({**e, "session": s.get("session_number", "?")})

    entries = all_exchanges[-args.last:] if args.last else all_exchanges

    if not entries:
        print("No drift history recorded.")
        return

    print(f"{'Exch':>4} {'Sess':>4} {'Score':>7} {'Verdict':<10} {'User Question':<40}")
    print("-" * 80)
    for e in entries:
        print(
            f"{e.get('exchange', '?'):>4} "
            f"{e.get('session', '?'):>4} "
            f"{e.get('score', 0):>7.1f} "
            f"{e.get('verdict', '?'):<10} "
            f"{e.get('user', '')[:40]}"
        )


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze context in session memory."""
    store = SessionMemoryStore(path=args.file)

    if not store.exists():
        print(f"No session memory found at: {args.file}")
        return

    data = store.load()
    data.context_frozen = True
    store.save(data)
    print(f"Context frozen in: {args.file}")


def cmd_unfreeze(args: argparse.Namespace) -> None:
    """Unfreeze context in session memory."""
    store = SessionMemoryStore(path=args.file)

    if not store.exists():
        print(f"No session memory found at: {args.file}")
        return

    data = store.load()
    data.context_frozen = False
    store.save(data)
    print(f"Context unfrozen in: {args.file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="context-drift-analyzer",
        description="Monitor context drift in LLM conversations",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    p_status = subparsers.add_parser("status", help="Show session memory status")
    p_status.add_argument(
        "--file", default=".session_memory", help="Path to .session_memory file"
    )

    # reset
    p_reset = subparsers.add_parser("reset", help="Delete session memory")
    p_reset.add_argument(
        "--file", default=".session_memory", help="Path to .session_memory file"
    )

    # history
    p_history = subparsers.add_parser("history", help="Show drift history")
    p_history.add_argument(
        "--file", default=".session_memory", help="Path to .session_memory file"
    )
    p_history.add_argument(
        "--last", type=int, default=20, help="Show last N entries (default: 20)"
    )

    # freeze
    p_freeze = subparsers.add_parser("freeze", help="Freeze context")
    p_freeze.add_argument(
        "--file", default=".session_memory", help="Path to .session_memory file"
    )

    # unfreeze
    p_unfreeze = subparsers.add_parser("unfreeze", help="Unfreeze context")
    p_unfreeze.add_argument(
        "--file", default=".session_memory", help="Path to .session_memory file"
    )

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "reset": cmd_reset,
        "history": cmd_history,
        "freeze": cmd_freeze,
        "unfreeze": cmd_unfreeze,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
