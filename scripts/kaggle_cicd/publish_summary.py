"""
Publish a GitHub Actions workflow summary for the
Dynamic Kaggle Fire-and-Forget Pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path


def getenv(name: str, default: str = "N/A") -> str:
    """Return an environment variable or a default value."""
    return os.getenv(name, default)


def build_summary() -> str:
    """Build the workflow summary markdown."""

    notebook = getenv("NOTEBOOK")
    slug = getenv("SLUG")
    owner = "rahulshelke98"

    branch = getenv("GITHUB_REF_NAME")
    sha = getenv("GITHUB_SHA")[:7]
    run_number = getenv("GITHUB_RUN_NUMBER")

    kaggle_url = (
        f"https://www.kaggle.com/code/{owner}/{slug}"
        if owner != "N/A" and slug != "N/A"
        else "N/A"
    )

    return f"""# 🚀 Kaggle Notebook Submitted

| Item | Value |
|------|-------|
| Notebook | `{notebook}` |
| Slug | `{slug}` |
| Branch | `{branch}` |
| Commit | `{sha}` |
| Workflow Run | `{run_number}` |

## Submission Status

✅ Notebook uploaded successfully.

✅ Execution request submitted to Kaggle.

This workflow follows a **Fire-and-Forget** execution model.

## Kaggle Notebook

{kaggle_url}

---
Results, metrics, models, and artifacts will become available in DagsHub after notebook execution completes.
"""


def publish_summary(summary: str) -> None:
    """Append the workflow summary to GitHub Actions."""

    summary_file = os.getenv("GITHUB_STEP_SUMMARY")

    if not summary_file:
        print(summary)
        return

    Path(summary_file).write_text(summary, encoding="utf-8")


def main() -> int:
    publish_summary(build_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
