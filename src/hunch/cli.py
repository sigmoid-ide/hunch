"""The `hunch` command line."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .diff import diff_lines
from .render import RAMP, View, can_encode, log_lines, ls_lines, show_lines, visible
from .runner import run_command
from .store import HunchError, Store

EMPTY = "no runs yet. Record one with: hunch run -- python train.py"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = _parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return args.handler(args) or 0
    except HunchError as e:
        print(f"hunch: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hunch",
        description="A zero-config experiment journal: every run keeps its code, config, environment and why.",
    )
    parser.add_argument("--version", action="version", version=f"hunch {__version__}")
    commands = parser.add_subparsers(metavar="COMMAND")

    run = commands.add_parser("run", help="run a command and record it", usage="hunch run [-m WHY] [--from RUN] -- COMMAND ...")
    run.add_argument("-m", "--why", help="why you are running this (asked for if left out in a terminal)")
    run.add_argument("--from", dest="parent", metavar="RUN", help="the run this one builds on (default: the last run on this branch)")
    run.add_argument("command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    run.set_defaults(handler=cmd_run)

    ls = commands.add_parser("ls", help="list recent runs")
    ls.add_argument("-n", type=int, default=20, help="how many runs to show (default 20)")
    ls.add_argument("--all", action="store_true", help="include runs that ended in under 30 seconds")
    ls.set_defaults(handler=cmd_ls)

    log = commands.add_parser("log", help="show the tree of runs")
    log.add_argument("--all", action="store_true", help="include runs that ended in under 30 seconds")
    log.set_defaults(handler=cmd_log)

    show = commands.add_parser("show", help="show everything about one run")
    show.add_argument("run")
    show.set_defaults(handler=cmd_show)

    diff = commands.add_parser("diff", help="compare two runs")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.add_argument("--code", action="store_true", help="show the full code diff")
    diff.set_defaults(handler=cmd_diff)

    note = commands.add_parser("note", help="add a conclusion, verdict or new why to a run")
    note.add_argument("run")
    note.add_argument("text", nargs="?", help="what you learned from the run")
    verdict = note.add_mutually_exclusive_group()
    verdict.add_argument("--keep", action="store_true", help="mark the run as worth building on")
    verdict.add_argument("--drop", action="store_true", help="mark the run as a dead end")
    note.add_argument("--why", help="replace the run's why")
    note.set_defaults(handler=cmd_note)

    link = commands.add_parser("link", help="fix the run a run came from")
    link.add_argument("run")
    link.add_argument("--from", dest="parent", required=True, metavar="RUN", help="the parent run, or 'none'")
    link.set_defaults(handler=cmd_link)

    settings = commands.add_parser("set", help="change a project setting")
    settings.add_argument("key", choices=["main-metric"])
    settings.add_argument("value", help="a metric name, or 'auto' to pick it automatically")
    settings.set_defaults(handler=cmd_set)
    return parser


def _view(store: Store, runs) -> View:
    return View(runs, store.config().get("main_metric"), can_encode(sys.stdout, RAMP))


def cmd_run(args) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise HunchError("nothing to run. Usage: hunch run [-m WHY] -- COMMAND ...")
    store = Store.find()
    parent = store.resolve(args.parent).id if args.parent else None
    why = args.why
    if why is None and _can_ask():
        try:
            why = input("why? (Enter to skip) ").strip() or None
        except EOFError:
            why = None
    return run_command(store, command, why, parent)


def _can_ask() -> bool:
    """True in an interactive terminal. On POSIX only for a foreground job, so a
    background job is never stopped waiting for input."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if os.name == "posix":
        try:
            return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
        except OSError:
            return False
    return True


def cmd_ls(args) -> int:
    store = Store.find()
    runs = store.runs()
    if not runs:
        print(EMPTY)
        return 0
    shown = visible(runs, args.all)
    for line in ls_lines(list(reversed(shown))[: args.n], _view(store, runs)):
        print(line)
    _print_hidden(len(runs) - len(shown))
    return 0


def cmd_log(args) -> int:
    store = Store.find()
    runs = store.runs()
    if not runs:
        print(EMPTY)
        return 0
    shown = visible(runs, args.all)
    for line in log_lines(shown, _view(store, runs)):
        print(line)
    _print_hidden(len(runs) - len(shown))
    return 0


def _print_hidden(count: int) -> None:
    if count:
        print(f"  {count} trivial run{'s' if count != 1 else ''} hidden (--all to show)")


def cmd_show(args) -> int:
    store = Store.find()
    runs = store.runs()
    for line in show_lines(store.resolve(args.run, runs), _view(store, runs), store.root):
        print(line)
    return 0


def cmd_diff(args) -> int:
    store = Store.find()
    runs = store.runs()
    a, b = store.resolve(args.a, runs), store.resolve(args.b, runs)
    for line in diff_lines(a, b, _view(store, runs), store.root, args.code):
        print(line)
    return 0


def cmd_note(args) -> int:
    if not (args.text or args.keep or args.drop or args.why):
        raise HunchError("nothing to add. Give a conclusion, --keep, --drop or --why")
    store = Store.find()
    run = store.resolve(args.run)
    if args.why:
        store.annotate(run, "why", args.why)
    if args.text:
        store.annotate(run, "note", args.text)
    if args.keep or args.drop:
        store.annotate(run, "verdict", "keep" if args.keep else "drop")
    print(f"hunch: noted on {run.id}")
    return 0


def cmd_link(args) -> int:
    store = Store.find()
    runs = store.runs()
    run = store.resolve(args.run, runs)
    if args.parent.lower() == "none":
        store.annotate(run, "parent", None)
        print(f"hunch: {run.id} now has no parent")
        return 0
    parent = store.resolve(args.parent, runs)
    if parent.id == run.id:
        raise HunchError("a run cannot be its own parent")
    if parent.started >= run.started:
        raise HunchError(f"{parent.id} started after {run.id}, so it cannot be its parent")
    store.annotate(run, "parent", parent.id)
    print(f"hunch: {run.id} now comes from {parent.id}")
    return 0


def cmd_set(args) -> int:
    store = Store.find()
    value = None if args.value == "auto" else args.value
    store.set_config("main_metric", value)
    print(f"hunch: main metric is {'picked automatically' if value is None else value}")
    return 0
