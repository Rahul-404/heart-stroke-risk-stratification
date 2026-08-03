"""
Generate kernel-metadata.json for Kaggle.

This script assumes all configuration has already been:
1. Parsed
2. Validated

Its only responsibility is to generate the metadata file expected by
`kaggle kernels push`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def get_string(name: str) -> str:
    """Read a string environment variable."""
    value = os.environ.get(name, "").strip()

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in ("'", '"')
    ):
        value = value[1:-1]

    return value

def get_bool(name: str) -> bool:
    """Read a boolean environment variable."""
    return os.environ.get(name, "false").lower() == "true"


def get_list(name: str) -> list[str]:
    """
    Read a JSON list from an environment variable.

    Example:
        '["dataset1", "dataset2"]'
    """
    value = os.environ.get(name, "[]")

    if not value.strip():
        return []

    return json.loads(value)


def build_metadata() -> dict:
    """Build the kernel metadata dictionary."""

    notebook = Path(os.environ["NOTEBOOK"])

    return {
        # Required
        "id": f'{os.environ["KAGGLE_USERNAME"]}/{os.environ["TITLE"]}',
        "title": get_string("TITLE"),
        "code_file": notebook.name,
        "language": get_string("LANGUAGE"),
        "kernel_type": get_string("KERNEL_TYPE"),
        "is_private": get_bool("IS_PRIVATE"),

        # Runtime
        "enable_gpu": get_bool("ENABLE_GPU"),
        "enable_tpu": get_bool("ENABLE_TPU"),
        "enable_internet": get_bool("ENABLE_INTERNET"),

        # Optional
        "machine_shape": get_string("MACHINE_SHAPE"),

        # Dependencies
        "dataset_sources": get_list("DATASET_SOURCES"),
        "competition_sources": get_list("COMPETITION_SOURCES"),
        "kernel_sources": get_list("KERNEL_SOURCES"),
        "model_sources": get_list("MODEL_SOURCES"),

        # Metadata
        "keywords": get_list("KEYWORDS"),
    }


def write_metadata(metadata: dict) -> Path:
    """Write kernel-metadata.json next to the notebook."""

    notebook = Path(os.environ["NOTEBOOK"])

    output = notebook.parent / "kernel-metadata.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output


def main() -> None:
    metadata = build_metadata()
    output = write_metadata(metadata)

    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
