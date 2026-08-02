"""
GitHub Actions utility to detect and pass modified Jupyter notebooks to the workflow.

This script identifies files changed in the most recent Git commit. If it
detects any notebook files modified inside the `notebooks/` directory, it extracts
the first one and writes its details directly to the GitHub Actions outputs.

Outputs:
    execute (bool): 'true' if a notebook changed under notebooks/, otherwise 'false'.
    notebook (str, optional): The relative file path of the first changed notebook.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys


def get_changed_files() -> list[str]:
    """Retrive list of changed file between the last two commits."""
    # Local/ACT override
    if notebook := os.getenv("TEST_NOTEBOOK"):
        return [notebook]

    # GitHub Actions
    before = os.getenv("BEFORE_SHA")
    after = os.getenv("AFTER_SHA")

    if before and after:
        return subprocess.check_output(
            ["git", "diff", "--name-only", before, after],
            text=True,
        ).splitlines()

    # Local git fallback
    return subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD^", "HEAD"],
        text=True,
    ).splitlines() # nosec B603

def find_notebooks(files: list[str]) -> list[str]:
    """Filters files down to those in the notebooks/ folder ending in .ipynb."""
    return [
        f
        for f in files
        if f.startswith("notebooks/")
        and f.endswith(".ipynb")
    ]

def write_outputs(notebook: str | None = None) -> None:
    """Writes outputs to GITHUB_OUTPUT environment file, or prints if running locally."""
    output = os.environ["GITHUB_OUTPUT"]

    with open(output, "a") as f:
        if notebook:
            f.write(f"notebook={notebook}\n")

def main() -> int:
    changed = get_changed_files()
    notebooks = find_notebooks(changed)
    write_outputs(notebooks[0])

    return 0
if __name__ == "__main__":
    sys.exit(main())
