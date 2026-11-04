# Contributing to Hunch

## Setup

```
uv sync
uv run pytest
```

Hunch has no runtime dependencies and supports Python 3.11 and newer on
Windows, Linux and macOS. Please keep it that way: anything that runs inside a
training process must install cleanly into any research environment.

## Changes to the on-disk format

[docs/format.md](docs/format.md) is a contract with every tool that reads runs.
Adding a field is fine; removing a field or changing what it means needs a new
format version and a migration note.

## Sign your commits (DCO)

Contributions are accepted under the Apache-2.0 license with a
[Developer Certificate of Origin](https://developercertificate.org/) sign-off.
Add it with `git commit -s`, which appends a line like:

```
Signed-off-by: Your Name <you@example.com>
```
