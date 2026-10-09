#!/usr/bin/env python3
"""Regenerate every experiments/**/stepNN_*.ipynb from its annotated .py source (jupytext py:percent).

The .py files are the canonical, reviewable source (they carry the [PAPER] comments); the .ipynb files
are generated copies for uploading to Kaggle. Usage:  python tools/sync_notebooks.py [--check]
--check exits non-zero if any notebook is out of date (useful in CI).
"""
import argparse, glob, json, os, sys
import jupytext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "accelerator": "GPU"}


def build(py):
    nb = jupytext.read(py, fmt="py:percent")
    nb.metadata.pop("jupytext", None)
    nb.metadata.update(KAGGLE_META)
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:03d}"  # deterministic ids so regenerated notebooks diff cleanly
    return jupytext.writes(nb, fmt="ipynb")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    stale = 0
    for py in sorted(glob.glob(os.path.join(ROOT, "experiments", "*", "step*.py"))):
        if py.endswith("_toy.py"):
            continue  # plain script, not a notebook
        ipynb = py[:-3] + ".ipynb"; new = build(py)
        old = open(ipynb).read() if os.path.exists(ipynb) else None
        if old != new:
            stale += 1
            if not a.check:
                open(ipynb, "w").write(new)
            print(("STALE " if a.check else "wrote ") + os.path.relpath(ipynb, ROOT))
    if a.check and stale:
        sys.exit(1)


if __name__ == "__main__":
    main()
