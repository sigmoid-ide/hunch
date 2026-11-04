# Hunch

A zero-config experiment journal for ML research. Every run keeps its code,
config, environment, metrics and the reason you ran it, as plain files in your
project. Each run also knows which run it came from, so your experiments form a
history you can read back, like `git log` for results.

```
$ hunch log
*    a7f3  Nov 14  warmup + cosine decay  val_loss 2.303                  █▅▃▂▁▁▁▁  keep  "cosine helps a bit"
| x  c21e  Nov 13  warmup 2k, lr 6e-4     crashed, exit 1; val_loss 7.42  ▃▂▂▁▁▄▆█  drop
|/
*    9b40  Nov 12  warmup 1k steps        val_loss 2.323                  █▅▃▂▁▁▁▁  keep
*    3d1c  Nov 10  baseline               val_loss 2.462                  █▅▃▂▁▁▁▁  (no conclusion yet)
```

No server, no account, no network calls, no telemetry, no dependencies.
Hunch is the first standalone piece of Sigmoid IDE, an open-source ML research IDE.

## Install

```
pip install hunch-ml
```

Python 3.11 or newer. The command is `hunch` and the import is `import hunch`.

## Record a run

Put `hunch run` in front of the command you already use:

```
hunch run -m "warmup to fix loss spikes" -- python train.py --lr 3e-4
```

Leave out `-m` in an interactive terminal and Hunch asks why (Enter skips). It
never asks when input is not a terminal, as in cluster jobs, background jobs and
piped scripts. Pass `-m` in shell scripts that launch many runs.

Or record from inside your code. One line records every run, however it is
launched:

```python
import hunch

run = hunch.init(config={"lr": 3e-4}, why="warmup to fix loss spikes")
for step in range(steps):
    ...
    hunch.log({"train_loss": loss, "val_loss": val_loss}, step=step)
```

Use both: inside `hunch run`, `hunch.init()` joins the wrapper's run instead of
starting a second one. The names follow W&B, so `import hunch as wandb` works
for `init`, `log`, `config`, `summary` and `finish`. `config` can be a dict or
an `argparse.Namespace`. Tensors are turned into numbers. In distributed
training only rank 0 records: Hunch reads `RANK`, `OMPI_COMM_WORLD_RANK` and
`PMI_RANK`, and inside `hunch run` also `SLURM_PROCID`.

## Read it back

```
hunch ls                 recent runs, newest first
hunch log                the tree of runs
hunch show a7f3          everything about one run
hunch diff 9b40 a7f3     what changed: code, config, environment and metrics
hunch diff 9b40 a7f3 --code     the full code diff
```

```
$ hunch diff 9b40 a7f3
9b40  Nov 12 14:12  warmup 1k steps
a7f3  Nov 14 09:30  warmup + cosine decay

code     train.py  +6 -1   (--code for the full diff)
config   schedule: const -> cosine
env      no changes
metrics  val_loss 2.323 -> 2.303 (-0.02)
```

Runs are named by the end of their id, usually 4 characters. Runs that ended in
under 30 seconds (typos, quick crashes) are hidden; `--all` shows them.

## Close the loop

```
hunch note a7f3 "cosine helps a bit" --keep     a conclusion and a verdict
hunch note c21e --drop                          a dead end
hunch note a7f3 --why "warmup + cosine decay"   a better why
hunch link a7f3 --from 9b40                     fix the run it came from
hunch set main-metric val/acc                   the metric shown for each run
```

Notes are added, never overwritten, so a run's history stays visible in
`hunch show`.

## What a run records

- The command, arguments and working directory
- The git commit and branch, the uncommitted changes to tracked files, and
  copies of small untracked text files, so the exact code can be rebuilt
- The config passed to `hunch.init()`
- Python version and path, installed packages, torch and CUDA versions, GPU
  model and driver, host, and the SLURM job id
- Metrics, final values, output (with `hunch run`), exit status and duration
- The why, your conclusion and your verdict

## Lineage

Each run's parent is picked for you: the run on the same git branch that ended
most recently, skipping runs still in progress, runs under 30 seconds and runs
you dropped. Branch per idea and you get a tree; stay on one branch and you get
a straight history. Use `hunch run --from <run>` to choose the parent yourself,
and `hunch link` to fix it later.

## Where it lives

Everything is in `.hunch/` at the root of your git repository (outside git, in
the current folder), one folder per run. The
folder ignores itself in git, so your repository is never touched. Runs never
change after they end and have unique ids, so copying them between machines is
a plain `rsync` of folders. The format is documented in
[docs/format.md](docs/format.md), for anyone who wants to read it without Hunch.

## Privacy

Nothing leaves your machine. Only these environment variables are recorded:
`CUDA_VISIBLE_DEVICES`, `CONDA_DEFAULT_ENV`, `VIRTUAL_ENV` and `SLURM_*`.
Credentials are removed from git remote URLs. Command arguments are recorded as
typed, so do not pass secrets as arguments.

## Known limits in this first version

- `hunch run` pipes the command's output to save it, so programs that check for
  a terminal may turn off colors. It sets `PYTHONUNBUFFERED=1` to keep output
  live.
- `hunch run` passes SIGTERM on to the command. It survives SIGHUP, SIGUSR1
  and SIGUSR2 without passing them on, because SLURM and terminals send those
  to the whole process group, so the command already gets them.
- Without `hunch run`, a script that ends with `sys.exit(n)` is recorded as
  finished, and one killed outright shows as lost after 5 minutes. Use
  `hunch run` for exact exit codes.
- No plots beyond sparklines, no sweeps, no checkpoint storage, no importers
  from other trackers yet.

## License

Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute.
