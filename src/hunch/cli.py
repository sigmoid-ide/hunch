"""The `hunch` command line."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__, milestones
from .diff import diff_lines
from .render import (
    RAMP, View, can_encode, cell_lines, log_lines, ls_lines, milestone_lines, milestone_show_lines, run_counts, show_lines, visible,
)
from .runner import run_command
from .store import HunchError, Store

EMPTY = "no runs yet. Record one with: hunch run -- python train.py"
WHY_ENV = "HUNCH_WHY"


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
    run.add_argument("-m", "--why", help="why you are running this (default: $HUNCH_WHY, else asked for in a terminal)")
    run.add_argument("--from", dest="parent", metavar="RUN", help="the run this one builds on (default: the last run on this branch)")
    run.add_argument("command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    run.set_defaults(handler=cmd_run)

    log = commands.add_parser("log", help="show milestones, or the tree of runs")
    log.add_argument("milestone", nargs="?", help="a milestone (M3) to show its runs")
    log.add_argument("--runs", action="store_true", help="the tree of all runs, even when there are milestones")
    log.add_argument("--oneline", action="store_true", help="recent runs as a flat list, newest first")
    log.add_argument("-n", type=int, default=20, help="with --oneline, how many runs to show (default 20)")
    log.add_argument("--all", action="store_true", help="include quick crashes (runs that failed in under 30 seconds)")
    log.set_defaults(handler=cmd_log)

    show = commands.add_parser("show", help="show everything about one run or milestone (M3)")
    show.add_argument("run")
    show.add_argument("--cells", action="store_true", help="for a notebook run, the cells it ran, in order")
    show.set_defaults(handler=cmd_show)

    milestone = commands.add_parser("milestone", help="save a milestone: the runs since the previous one, summed up")
    milestone.add_argument("title", help="what this milestone is, for example \"warmup + cosine beats baseline\"")
    milestone.add_argument("-m", "--note", help="your own words, next to the summary Hunch writes")
    milestone.add_argument("--best", metavar="RUN", help="the run that best shows it (default: picked by the main metric)")
    milestone.set_defaults(handler=cmd_milestone)

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

    config = commands.add_parser("config", help="show or change a project setting")
    config.add_argument("key", choices=["main-metric"])
    config.add_argument("value", nargs="?", help="the new value; leave out to show the current one")
    config.add_argument("--unset", action="store_true", help="go back to the default")
    config.set_defaults(handler=cmd_config)
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
    if why is None and WHY_ENV in os.environ:
        # Editors pass the why here, so it never has to survive a shell's quoting.
        why = os.environ[WHY_ENV].strip() or None
    elif why is None and _can_ask():
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


def cmd_log(args) -> int:
    store = Store.find()
    runs = store.runs()
    saved = milestones.milestones(store)
    if not runs and not saved:
        print(EMPTY)
        return 0
    view = _view(store, runs)
    if args.milestone:
        members = set(milestones.resolve(store, args.milestone, saved).runs)
        runs = [run for run in runs if run.id in members]
    elif saved and not (args.runs or args.oneline or args.all):
        for line in milestone_lines(saved, milestones.numbers(saved), runs, view):
            print(line)
        return 0
    shown = visible(runs, args.all)
    lines = ls_lines(list(reversed(shown))[: args.n], view) if args.oneline else log_lines(shown, view)
    for line in lines:
        print(line)
    _print_hidden(len(runs) - len(shown))
    return 0


def _print_hidden(count: int) -> None:
    if count:
        print(f"  {count} quick crash{'es' if count != 1 else ''} hidden (--all to show)")


def cmd_show(args) -> int:
    store = Store.find()
    runs = store.runs()
    if milestones.is_milestone_ref(args.run):
        saved = milestones.milestones(store)
        milestone = milestones.resolve(store, args.run, saved)
        lines = milestone_show_lines(milestone, milestones.numbers(saved)[milestone.id], runs, _view(store, runs), store.root)
    elif args.cells:
        lines = cell_lines(store.resolve(args.run, runs))
    else:
        lines = show_lines(store.resolve(args.run, runs), _view(store, runs), store.root)
    for line in lines:
        print(line)
    return 0


def cmd_milestone(args) -> int:
    store = Store.find()
    milestone = milestones.create(store, args.title, args.note, args.best)
    number = milestones.numbers(milestones.milestones(store))[milestone.id]
    print(f"hunch: saved {number}, covering {run_counts([r for r in store.runs() if r.id in set(milestone.runs)])}")
    print(f"  {milestone.summary}")
    return 0


def cmd_diff(args) -> int:
    store = Store.find()
    runs = store.runs()
    a, b = store.resolve(args.a, runs), store.resolve(args.b, runs)
    for line in diff_lines(a, b, _view(store, runs), store.dir / "objects", args.code):
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


def cmd_config(args) -> int:
    """Like `git config`: no value shows the setting, a value sets it, --unset clears it."""
    store = Store.find()
    if args.unset:
        store.set_config("main_metric", None)
    elif args.value is not None:
        store.set_config("main_metric", args.value)
    print(store.config().get("main_metric") or "(auto: the first metric with 'val' in its name, else the last one)")
    return 0
