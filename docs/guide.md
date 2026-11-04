# Hunch guide

## Record

```bash
hunch run -m "warmup to fix loss spikes" -- python train.py --lr 3e-4
hunch run --from 9b40 -m "same, longer warmup" -- python train.py
```

- Leave out `-m` in a terminal and Hunch asks why (Enter skips). It never asks in scripts, background jobs or cluster jobs.
- `--from` picks the parent run. Without it, the parent is the last run on the same git branch that ended,
  skipping runs still in progress, runs under 30 seconds and runs you dropped.

In code, `hunch.init()` records the config, metrics and exact interpreter. Inside `hunch run` it joins that run.

```python
import hunch

hunch.init(config=args, why="warmup to fix loss spikes")   # a dict or argparse.Namespace
hunch.log({"val_loss": val_loss}, step=step)               # tensors become numbers
```

The names follow W&B, so `import hunch as wandb` works for `init`, `log`, `config`, `summary` and `finish`.
In distributed training only rank 0 records (`RANK`, `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, and inside
`hunch run` also `SLURM_PROCID`).

## Read

```
hunch log                    the tree of runs
hunch log --oneline -n 50    a flat list, newest first
hunch log --all              include runs under 30 seconds
hunch show a7f3              one run in full
hunch diff 9b40 a7f3         code, config, environment and metric changes
hunch diff 9b40 a7f3 --code  the full code diff
```

Runs are named by the end of their id, usually 4 characters.

## Annotate

```
hunch note a7f3 "cosine helps" --keep       conclusion and verdict
hunch note c21e --drop                      a dead end
hunch note a7f3 --why "warmup + cosine"     a better why
hunch link a7f3 --from 9b40                 fix the parent (--from none for no parent)
hunch config main-metric val_acc            the metric shown for each run (--unset for auto)
```

Notes are added, never overwritten, so `hunch show` keeps their history.

## What a run records

- Command, arguments, working directory, host and SLURM job
- Git commit and branch, uncommitted changes, and small untracked text files (up to 100 KB each, 1 MB in all)
- Config, metrics, final values, output, exit status and duration
- Python, packages, torch and CUDA, GPU model and driver
- Why, conclusion and verdict

Everything is in `.hunch/` at the git root (outside git, the current folder), one folder per run, never
changed after it ends. Copy runs between machines with `rsync`. Every file is specified in [format.md](format.md).

## Privacy

Nothing leaves your machine. Only `CUDA_VISIBLE_DEVICES`, `CONDA_DEFAULT_ENV`, `VIRTUAL_ENV` and `SLURM_*` are
recorded from the environment. Credentials are removed from git remotes and untracked symlinks are skipped.
Arguments are recorded as typed, so don't pass secrets as arguments.

## Known limits

- `hunch run` pipes output, so tools that check for a terminal may drop colors. It sets `PYTHONUNBUFFERED=1` and
  `PYTHONIOENCODING=utf-8` unless you set them.
- SIGTERM is passed on to the command. SIGHUP, SIGUSR1 and SIGUSR2 are not: SLURM and terminals send them to the
  whole process group, so the command gets them anyway.
- Without `hunch run`, `sys.exit(n)` is recorded as finished, and a run killed outright shows as lost after
  5 minutes.
- No plots beyond sparklines, no sweeps, no checkpoint storage, no importers yet.

## Develop

```bash
git clone https://github.com/sigmoid-ide/hunch.git && cd hunch
uv sync && uv run pytest
```

No runtime dependencies, ever. See [CONTRIBUTING.md](../CONTRIBUTING.md).
