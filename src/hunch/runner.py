"""`hunch run`: launches a command and records it as a run."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .capture import guess_python, start_run
from .store import Heartbeat, HunchError, Store, finish_run

RUN_DIR_ENV = "HUNCH_RUN_DIR"
DRAIN_SECONDS = 5  # how long to keep reading output that a leftover background process holds open


def run_command(store: Store, command: list[str], why: str | None, parent: str | None) -> int:
    """Runs `command`, shows its output live while saving it, and returns its exit code."""
    executable = shutil.which(command[0])
    if executable is None:
        raise HunchError(f"command not found: {command[0]}")
    run_dir = start_run(
        store,
        command=command,
        cwd=Path.cwd(),
        why=why,
        parent=parent,
        source="wrapper",
        python=guess_python(executable),
    )

    env = dict(os.environ)
    env[RUN_DIR_ENV] = str(run_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")  # the output is piped; keep it live
    env.setdefault("PYTHONIOENCODING", "utf-8")  # a piped Windows Python would otherwise fail on non-cp1252 text
    heartbeat = Heartbeat(run_dir / "heartbeat")
    heartbeat.start()
    started = time.monotonic()
    log = _Log(run_dir / "output.log")
    try:
        proc = subprocess.Popen([executable, *command[1:]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except OSError as e:
        log.close()
        heartbeat.stop()
        finish_run(run_dir, status="crashed", exit_code=None, duration=time.monotonic() - started)
        raise HunchError(f"could not start {command[0]}: {e}") from e
    pumps = [
        threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, log), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, log), daemon=True),
    ]
    for pump in pumps:
        pump.start()

    stopped = False

    def forward(signum, frame):
        nonlocal stopped
        stopped = True
        proc.send_signal(signum)

    def survive(signum, frame):
        pass  # sent to the whole process group (SLURM, a closed terminal), so the command got it too

    previous = {}
    if os.name == "posix":
        handlers = {signal.SIGTERM: forward, signal.SIGHUP: survive, signal.SIGUSR1: survive, signal.SIGUSR2: survive}
        previous = {signum: signal.signal(signum, handler) for signum, handler in handlers.items()}

    try:
        interrupts = 0
        while True:
            try:
                code = proc.wait(timeout=0.5)  # a timeout keeps Ctrl+C responsive on Windows
                break
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                # The child got Ctrl+C too. Give it time to clean up; stop it on the second one.
                stopped = True
                interrupts += 1
                if interrupts >= 2:
                    proc.terminate()
        duration = time.monotonic() - started
        deadline = time.monotonic() + DRAIN_SECONDS
        for pump in pumps:
            try:
                pump.join(max(0.0, deadline - time.monotonic()))
            except KeyboardInterrupt:
                stopped = True
        log.close()
        heartbeat.stop()
        if code == 0 and not stopped:
            status = "finished"
        elif stopped or code < 0:
            status = "killed"
        else:
            status = "crashed"
        finish_run(run_dir, status=status, exit_code=code, duration=duration)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(f"hunch: run {run_dir.name} {status} after {format_duration(duration)}", file=sys.stderr)
    if code < 0:
        return 128 - code  # POSIX: killed by a signal
    if code > 0x7FFFFFFF:
        return code - (1 << 32)  # Windows status codes such as 0xC000013A (Ctrl+C)
    return code


class _Log:
    """output.log, shared by both pumps. Saving stops at the first write error,
    such as a full disk or quota, so the command never blocks on it."""

    def __init__(self, path: Path):
        self._lock = threading.Lock()
        try:
            self._file = open(path, "ab")
        except OSError:
            self._file = None

    def write(self, chunk: bytes) -> None:
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(chunk)
                self._file.flush()
            except (OSError, ValueError):
                self._close()

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


def _pump(source, target, log: _Log) -> None:
    """Copies a child's output stream to our own stream and to output.log."""
    raw = getattr(target, "buffer", None)
    while True:
        chunk = source.read1(65536)
        if not chunk:
            break
        log.write(chunk)
        try:
            if raw is not None:
                raw.write(chunk)
                raw.flush()
            else:
                target.write(chunk.decode("utf-8", "replace"))
                target.flush()
        except (OSError, ValueError):
            pass  # our own output is gone (closed pipe); keep saving the log


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"
