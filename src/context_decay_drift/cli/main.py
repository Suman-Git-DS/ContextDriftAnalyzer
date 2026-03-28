"""CLI tool for context-decay-drift.

Commands:
    context-decay-drift status  — Show session memory status
    context-decay-drift reset   — Delete session memory file
    context-decay-drift history — Show drift history
    context-decay-drift freeze  — Freeze context (prevent modifications)
    context-decay-drift unfreeze — Unfreeze context

Usage:
    python -m context_decay_drift.cli.main status
    python -m context_decay_drift.cli.main status --file .session_memory
    python -m context_decay_drift.cli.main reset
    python -m context_decay_drift.cli.main history --last 10
"""

from __future__ import annotations

import argparse
import json
import sys

from context_decay_drift.persistence.session_memory import SessionMemoryStore


def cmd_status(args: argparse.Namespace) -> None:
    """Show session memory status."""
    store = SessionMemoryStore(path=args.file)

    if not store.exists():
        print(f"No session memory found at: {args.file}")
        return

    data = store.load()
    print(f"Session Memory: {args.file}")
    print(f"  Sessions:       {data.session_count}")
    print(f"  Total turns:    {data.total_turns}")
    print(f"  Context frozen: {data.context_frozen}")
    print(f"  Summaries:      {len(data.session_summaries)}")
    print(f"  Drift entries:  {len(data.drift_history)}")

    if data.drift_history:
        last = data.drift_history[-1]
        print(f"\n  Latest drift:")
        print(f"    Score:   {last.get('score', 'N/A')}/100")
        print(f"    Verdict: {last.get('verdict', 'N/A')}")
        print(f"    Reason:  {last.get('explanation', 'N/A')}")

    if data.session_summaries:
        print(f"\n  Session summaries:")
        for s in data.session_summaries:
            print(
                f"    Session {s.get('session_number', '?')}: "
                f"{s.get('summary', '')[:80]}..."
            )


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
    entries = data.drift_history[-args.last :] if args.last else data.drift_history

    if not entries:
        print("No drift history recorded.")
        return

    print(f"{'Turn':>5} {'Session':>8} {'Score':>7} {'Verdict':>12}  Explanation")
    print("-" * 80)
    for e in entries:
        print(
            f"{e.get('turn', '?'):>5} "
            f"{e.get('session', '?'):>8} "
            f"{e.get('score', 0):>7.1f} "
            f"{e.get('verdict', '?'):>12}  "
            f"{e.get('explanation', '')[:50]}"
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
        prog="context-decay-drift",
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
