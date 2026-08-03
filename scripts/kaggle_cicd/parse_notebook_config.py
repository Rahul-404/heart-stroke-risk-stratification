"""
Parse Kaggle notebook configuration and export GitHub Action outputs.

Responsibility:
- Read notebook
- Extract KAGGLE_CONFIG values
- Export outputs

Does NOT:
- Validate values
- Modify notebook
- Generate metadata
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATTERN = re.compile(
    r"#\s*KAGGLE_CONFIG:\s*([a-zA-Z0-9_]+)\s*=\s*(.+)"
)

def set_output(name: str, value) -> None:
    """Export GitHub Action output."""

    github_output = os.environ["GITHUB_OUTPUT"]

    if isinstance(value, (list, dict)):
        out_value = json.dumps(value)
    elif isinstance(value, bool):
        out_value = str(value).lower()
    else:
        out_value = str(value)

    with open(github_output, "a", encoding="utf-8") as f:
        f.write(f"{name}={out_value}\n")

def parse_value(value: str):
    """Convert string into Python object."""

    value = value.strip()

    try:
        return ast.literal_eval(value)
    except Exception:

        lower = value.lower()

        if lower == "true":
            return True

        if lower == "false":
            return False

        return value

def parse_title_value(value: str):
    """Convert title string into Python object."""

    value = value.strip()

    title = value.lower().replace('_', '-')

    return title

def read_config(notebook_path: Path) -> dict:

    with notebook_path.open(encoding="utf-8") as f:
        notebook = json.load(f)

    config = {}

    for cell in notebook["cells"]:
        if cell["cell_type"] not in ("code", "markdown"):
            continue

        for line in cell["source"]:
            match = CONFIG_PATTERN.match(line)

            if match:
                key = match.group(1)
                value = parse_value(match.group(2))

                config[key] = value
    return config

def get_git_sha():
    """Extracts the commit SHA from GITHUB_SHA env var or fallback to local git."""
    # Check if running inside Github Actions environment
    github_sha = os.getenv("GITHUB_SHA")
    if github_sha:
        return github_sha

    # Fallback to local git CLI if executing locally
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return sha
    except Exception:
        return "unknown"

def create_config(notebook_path: Path) -> dict:

    config = read_config(notebook_path)

    # Generate UTC timestamp and Commit SHA
    executed_at = datetime.now(timezone.utc).isoformat()
    sha = get_git_sha()[:7]

    defaults = {
        "execute": True,
        "title": parse_title_value(notebook_path.stem),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": [],
        "sha": sha,
        "executed_at": executed_at,
    }

    defaults.update(config)

    return defaults

def main():

    notebook = Path(os.environ["NOTEBOOK"])

    config = create_config(notebook)

    for key, value in config.items():
        set_output(key, value)


if __name__ == "__main__":
    sys.exit(main())
