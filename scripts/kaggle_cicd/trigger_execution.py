"""
Trigger execution of an uploaded Kaggle notebook.

Responsibilities
----------------
1. Authenticate with Kaggle.
2. Trigger notebook execution.
3. Exit immediately after Kaggle accepts the request.

This is intentionally a "fire-and-forget" script.
"""

from __future__ import annotations

import os
import sys

from kaggle.api.kaggle_api_extended import KaggleApi


def get_env(name: str) -> str:
    """Read a required environment variable."""

    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def trigger_execution(owner: str, notebook: str) -> None:
    """Trigger execution of a Kaggle notebook."""

    api = KaggleApi()
    api.authenticate()

    folder = os.path.dirname(notebook) or "."

    print(f"Triggering notebook: {owner}/{notebook}")

    api.kernels_push_cli(
        folder=folder,
        timeout=None,
        acc=None,
    )

    print("✓ Execution request submitted successfully.")


def main() -> int:
    try:
        owner = get_env("KAGGLE_USERNAME")
        notebook = get_env("NOTEBOOK")

        trigger_execution(owner, notebook)

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
