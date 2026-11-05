<div align="center">
  <br/>
  <img src="https://raw.githubusercontent.com/sigmoid-ide/hunch/main/media/logo.svg" width="140" alt="Hunch logo" />

# Hunch

**Every hunch, tracked.**

<a href="https://github.com/sigmoid-ide/hunch/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sigmoid-ide/hunch/ci.yml?branch=main&style=for-the-badge&label=ci" alt="CI status" /></a>
<a href="https://pypi.org/project/sigmoid-hunch/"><img src="https://img.shields.io/pypi/v/sigmoid-hunch?style=for-the-badge&color=6E56CF" alt="PyPI version" /></a>
<img src="https://img.shields.io/badge/python-3.11%2B-6E56CF?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11 or newer" />
<a href="https://github.com/sigmoid-ide/hunch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-6E56CF?style=for-the-badge" alt="License: Apache-2.0" /></a>

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/sigmoid-ide/hunch/main/media/hunch-log.svg" width="100%" alt="hunch run, then hunch log showing a tree of runs" />
</p>

**A journal for your ML experiments.** Every training run saves its code, config, environment and metrics,
why you ran it, and which run it built on.

```bash
pip install sigmoid-hunch
```

<p>
  <a href="https://github.com/sigmoid-ide/hunch-jetbrains"><img src="https://raw.githubusercontent.com/sigmoid-ide/hunch/main/media/pycharm-button.svg" height="44" alt="Get the PyCharm plugin" /></a>
</p>

## If you know git, you know Hunch

| git | hunch |
|---|---|
| `git commit -m "why"` | `hunch run -m "why" -- python train.py` |
| `git log` | `hunch log` |
| `git show a7f3` | `hunch show a7f3` |
| `git diff 9b40 a7f3` | `hunch diff 9b40 a7f3` |
| `git notes add` | `hunch note a7f3 "helps" --keep` |
| `git tag` | `hunch milestone "warmup beats baseline"` |

**Why not just git?** Twenty runs of the same code with different learning rates are one commit to git.
Hunch keeps all twenty, each with its settings and results.

## In your code

```python
import hunch

hunch.init(config=args, why="warmup to fix loss spikes")
hunch.log({"val_loss": val_loss}, step=step)
```

Same names as W&B: `import hunch as wandb` works.

## In your editor

<a href="https://github.com/sigmoid-ide/hunch-jetbrains"><img src="https://raw.githubusercontent.com/sigmoid-ide/hunch-jetbrains/main/media/marketplace/01-runs.png" width="100%" alt="The Hunch tool window in PyCharm: milestones, the tree of runs with their metric curves and conclusions, and one run's details" /></a>

**[Hunch for PyCharm](https://github.com/sigmoid-ide/hunch-jetbrains)**: your runs as a tree, keep or drop them,
compare any two in the diff viewer, and start runs with one click. In review on JetBrains Marketplace.

**VS Code, Cursor and Windsurf**: an extension is on the way.

## Your disk, your files

No server, no account, no telemetry. Just files in `.hunch/`.

[Guide](https://github.com/sigmoid-ide/hunch/blob/main/docs/guide.md) · [File format](https://github.com/sigmoid-ide/hunch/blob/main/docs/format.md) · [Questions](https://github.com/sigmoid-ide/hunch/discussions/categories/q-a) · [Contributing](https://github.com/sigmoid-ide/hunch/blob/main/CONTRIBUTING.md)

## Share it

Saved you a "what did I run last month?" moment? Tell one person, or post your `hunch log` in
[Show and tell](https://github.com/sigmoid-ide/hunch/discussions/categories/show-and-tell).

<p align="center">
  <a href="https://x.com/intent/post?text=Hunch%3A%20every%20hunch%2C%20tracked.%20Every%20ML%20run%20keeps%20its%20code%2C%20config%2C%20environment%20and%20why.%20pip%20install%20sigmoid-hunch&url=https%3A%2F%2Fgithub.com%2Fsigmoid-ide%2Fhunch"><img src="https://img.shields.io/badge/share_on-X-000000?style=for-the-badge&logo=x" alt="Share on X" /></a>
  <a href="https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fsigmoid-ide%2Fhunch&title=Hunch%3A%20every%20hunch%2C%20tracked.%20An%20experiment%20journal%20for%20ML"><img src="https://img.shields.io/badge/share_on-Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white" alt="Share on Reddit" /></a>
  <a href="https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fsigmoid-ide%2Fhunch&t=Hunch%3A%20every%20hunch%2C%20tracked.%20An%20experiment%20journal%20for%20ML"><img src="https://img.shields.io/badge/share_on-Hacker_News-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" alt="Share on Hacker News" /></a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fsigmoid-ide%2Fhunch"><img src="https://img.shields.io/badge/share_on-LinkedIn-0A66C2?style=for-the-badge" alt="Share on LinkedIn" /></a>
</p>

Tracking a research repo with Hunch? Show it:
[![tracked with Hunch](https://img.shields.io/badge/tracked_with-hunch-6E56CF?style=flat-square)](https://github.com/sigmoid-ide/hunch)

```markdown
[![tracked with Hunch](https://img.shields.io/badge/tracked_with-hunch-6E56CF?style=flat-square)](https://github.com/sigmoid-ide/hunch)
```

<p align="center">Apache-2.0 · Part of Sigmoid IDE · Like it? Give it a star.</p>
