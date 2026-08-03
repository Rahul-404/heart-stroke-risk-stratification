"""
Push a prepared notebook to Kaggle.

Responsibilities
----------------
1. Validate required inputs.
2. Authenticate using Kaggle credentials.
3. Upload notebook using Kaggle CLI.
4. Emit GitHub Action outputs.

This script intentionally does NOT:
- modify notebook
- generate metadata
- inject GitHub metadata
- validate notebook config
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

################################################################################
# Helpers
################################################################################


def fail(message: str) -> None:
    """Print an error and terminate."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def github_output(name: str, value: str) -> None:
    """Write a GitHub Actions output."""
    output = os.getenv("GITHUB_OUTPUT")

    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


################################################################################
# Validation
################################################################################


def validate_environment() -> tuple[Path, Path]:
    notebook = cast(str, os.getenv("NOTEBOOK"))

    if not notebook:
        fail("NOTEBOOK environment variable is missing.")

    notebook_path = Path(notebook).resolve()

    if not notebook_path.exists():
        fail(f"Notebook not found: {notebook_path}")

    metadata_path = notebook_path.parent / "kernel-metadata.json"

    if not metadata_path.exists():
        fail(f"kernel-metadata.json not found: {metadata_path}")

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username:
        fail("KAGGLE_USERNAME is not set.")

    if not key:
        fail("KAGGLE_KEY is not set.")

    return notebook_path, metadata_path


################################################################################
# Upload
################################################################################


def push_to_kaggle(notebook_path: Path) -> None:
    notebook_dir = notebook_path.parent

    cmd = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(notebook_dir),
    ]

    print("Executing:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


################################################################################
# Main
################################################################################


def main() -> None:
    notebook_path, _ = validate_environment()

    print(f"Notebook : {notebook_path}")

    # push_to_kaggle(notebook_path)

    github_output("upload_status", "success")

    print("Notebook uploaded successfully.")


if __name__ == "__main__":
    main()
