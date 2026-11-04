"""Collects the environment of the Python that runs the training code.

It runs either inside the current process or, from the `hunch run` wrapper,
as `python -c <this file's source>` in the target interpreter. So it must not
import anything from hunch, and it must keep working on old Pythons.
"""

import json
import os
import platform
import re
import sys


def collect():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "packages": _packages(),
        "torch": _torch(),
    }


def _packages():
    try:
        from importlib import metadata
    except ImportError:
        return None
    found = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found[name] = dist.version
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def _torch():
    # Read torch/version.py as text, because importing torch takes seconds.
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    path = os.path.join(os.path.dirname(spec.origin), "version.py")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    def field(name):
        match = re.search(r"^%s\s*(?::[^=\n]*)?=\s*['\"]([^'\"]*)['\"]" % name, text, re.M)
        return match.group(1) if match else None

    return {"version": field("__version__"), "cuda": field("cuda"), "hip": field("hip")}


if __name__ == "__main__":
    sys.stdout.write(json.dumps(collect()))
