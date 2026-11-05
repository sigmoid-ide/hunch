# The Hunch on-disk format, version 2

Hunch keeps everything in plain files, so people, scripts, coding agents and
other tools can read runs without Hunch installed. This document is the
contract: anything written here is safe to rely on.

## Layout

```
<project>/.hunch/
  .gitignore              "*": the folder ignores itself, so your repo is never touched
  config.json             project settings
  objects/ab/cdef...      every saved file, once, as a git blob object
  runs/<run-id>/
    run.json              written at start
    files.json            written at start: the run's code snapshot
    config.json           written at start
    env.json              written at start
    metrics.jsonl         appended while the run is in progress
    summary.json          optional, written while the run is in progress
    output.log            appended while the run is in progress (wrapper only)
    heartbeat             touched every 60 seconds while the run is in progress
    end.json              written once, when the run ends
    notes.jsonl           append-only, at any time
```

In a git repository, `<project>` is the nearest folder with `.hunch/runs`
between the current folder and the repository root, otherwise the root. The
search never leaves the repository. Outside git, it is the current folder.

A folder without `run.json` is not a run yet: `run.json` is written last when a
run starts.

## Rules

- While a run is in progress, the training process may rewrite `run.json`,
  `config.json`, `env.json` and `summary.json` (when it joins a run started by
  `hunch run`). Every rewrite is atomic.
- After `end.json` exists, only `notes.jsonl` changes, and only by appending.
  Corrections (a new why, a conclusion, a verdict, a different parent) are new
  lines, so every change keeps its history.
- Each run writes only inside its own folder. Runs from other machines can be
  copied in as folders; they never conflict.

## Run ids

`YYYYMMDD-HHMM-xxxxxx`: the local start time to the minute, then 6 random hex
characters. Ids sort by time and are unique across machines. Tools show the
shortest unique ending, usually the last 4 characters.

## run.json

| Field | Meaning |
|---|---|
| `format` | Format version, `2` |
| `id` | The run id, same as the folder name |
| `started_at` | ISO 8601 time with offset |
| `source` | `"wrapper"` (started by `hunch run`) or `"init"` (started by `hunch.init()`) |
| `command` | The command as a list of arguments, recorded as typed |
| `cwd` | Working directory |
| `why` | The reason given at start, or `null` |
| `parent` | Id of the run this one came from, or `null` |
| `parent_source` | `"explicit"` (`--from`), `"default"` (picked automatically) or `null` |
| `line` | Line of inquiry: the git branch, or `null` outside git or on a detached HEAD |
| `main_metric` | Metric chosen with `hunch.init(main_metric=...)`, or `null` |
| `labels` | `project`, `name`, `tags` given to `hunch.init()` |
| `git` | See below, or `null` outside a git repository |
| `host` | `hostname`, `os`, `slurm_job_id` |

`git` holds `root`, `commit` (`null` before the first commit), `branch`,
`remote` (credentials removed) and `dirty` (uncommitted changes exist). It is
only context: the code itself is in the snapshot, so git is optional.

## Code snapshot

`files.json` is `{"files": {"<path>": {"sha": ..., "size": ...}}, "bytes": ...,
"skipped": [...], "skipped_count": ..., "scan_truncated": ...}`. Paths are
relative to `<project>` and use `/`. Each `sha` names a file in
`objects/<first 2 characters>/<the rest>`: a git loose blob object, the
zlib-compressed bytes `blob <size>\0<content>`, named by the SHA-1 of those
bytes. The content is the file exactly as it was on disk, line endings included.
Identical files are stored once across all runs.

Which files are saved:
- In a git repository, the files git lists as tracked or untracked-but-not-ignored,
  so `.gitignore` applies. Outside git, every file under `<project>`, skipping
  hidden folders.
- Never: folders named `.git`, `.hunch`, `__pycache__`, `node_modules`,
  `.ipynb_checkpoints`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.tox`,
  `site-packages`, `wandb`, `mlruns` or `lightning_logs`; Python environments
  (folders containing `pyvenv.cfg` or `conda-meta`); secrets (`.env`, `.env.*`,
  `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ed25519*`, `.netrc`,
  `.pypirc`); anything matched by `.hunchignore`.
- `.hunchignore` has one pattern per line (`#` starts a comment). A pattern
  ending in `/` matches folders, one containing `/` matches the whole path, and
  any other matches a file or folder name.
- Symlinks, binary files (a NUL byte in the first 8 KB), files over 1 MB, files
  past a 20 MB total and files past the first 5000 are not saved; they are
  listed in `skipped` (first 1000 names). Without git, scanning stops after
  20000 files and `scan_truncated` is `true`.

## config.json

`{"args": [...], "values": {...}}`. `args` is the full command. `values` is
what the training process passed to `hunch.init(config=...)` or set on
`hunch.config`.

## env.json

| Field | Meaning |
|---|---|
| `source` | `"process"` (recorded by the training process itself) or `"wrapper"` (the wrapper's guess of the interpreter) |
| `python`, `implementation`, `executable`, `prefix`, `platform` | The interpreter |
| `packages` | Installed distributions, `{name: version}` |
| `torch` | `{version, cuda, hip}` read from `torch/version.py`, or `null` |
| `gpus` | From `nvidia-smi`: `[{name, driver, memory}]`, or `null` |
| `env_vars` | Only `CUDA_VISIBLE_DEVICES`, `CONDA_DEFAULT_ENV`, `VIRTUAL_ENV` and `SLURM_*` |
| `error` | Present when the interpreter could not be inspected |

## metrics.jsonl

One JSON object per `log()` call: `_step` (integer), `_time` (Unix seconds),
then the logged values. Keys starting with `_` are reserved. Non-finite floats
are written as the bare tokens `NaN`, `Infinity` and `-Infinity`, as Python's
`json` module does. A last line cut short by a crash is ignored by readers.

## summary.json

Optional overrides for final values, set with `run.summary[...] = value`.

## end.json

| Field | Meaning |
|---|---|
| `status` | `"finished"`, `"crashed"` or `"killed"` |
| `exit_code` | The exit code; negative for a signal; `null` when unknown |
| `ended_at` | ISO 8601 time with offset |
| `duration` | Seconds |
| `steps` | Last `_step`, or `null` |
| `metrics` | Final value of each metric, with `summary.json` applied |
| `metric_keys` | Metric names in the order they first appeared |
| `sparks` | Up to 32 averaged points per numeric metric, for sparklines |

## Status

Taken from `end.json` when it exists. Otherwise the run is `"running"` if
`heartbeat` changed less than 5 minutes ago, and `"lost"` if not (killed
outright, preempted, or the machine went down).

## notes.jsonl

One event per line: `{"at": "<ISO time>", "kind": ..., "value": ...}`.

| Kind | Value |
|---|---|
| `why` | Replaces the why |
| `note` | A conclusion |
| `verdict` | `"keep"` or `"drop"` |
| `parent` | A run id, or `null` for no parent |

The latest event of each kind wins.

## Lineage

When a run starts without `--from`, its parent is the run on the same line
that ended most recently, skipping runs still in progress, trivial runs and
runs with the verdict `drop`. A parent must have started before its child.

Runs whose status is not `"finished"` and that lasted under 30 seconds are
"trivial" (typos, instant crashes): hidden from `hunch log` unless `--all` is
given or another shown run came from them. Quick runs that finished are not
trivial.

## Project config.json

`main_metric`: the metric shown for every run. When unset, each run uses its
own `main_metric`, else its first metric with "val" in the name, else its last
metric.

## Compatibility

Readers must ignore fields they do not know. New fields can be added within a
version; removing or changing the meaning of a field requires a new version.

Version 1 (Hunch 0.1) kept the code as `code.patch` (the uncommitted diff
against `git.commit`) plus copies in `untracked/`, and its `git` field had
`patch_bytes` and `untracked*` fields. Version 2 replaces them with the
snapshot. Hunch still reads version 1 runs, but cannot compare their code.
