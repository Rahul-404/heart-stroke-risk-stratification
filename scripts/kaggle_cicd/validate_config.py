"""
Validate Kaggle notebook configuration before workflow execution.

Responsibility:
- Validate configuration keys
- Validate required fields
- Validate value types
- Validate field values (slug, machine shape, sources, etc.)
- Validate cross-field constraints

Does NOT:
- Parse notebook metadata
- Modify notebook contents
- Export GitHub Action outputs
"""

from __future__ import annotations

import os
import re
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import TypedDict

# Append the repository root path to python's loading index dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.kaggle_cicd.parse_notebook_config import create_config


class Configfield(TypedDict, total=False):
    type: type
    required: bool
    pattern: str
    choices: list[str]
    slug: str

CONFIG_SCHEMA: dict[str, Configfield] = {
    "execute": {
        "type": bool,
        "required": True,
    },
    "slug": {
        "type": str,
        "required": True,
        "pattern": r"^[a-z0-9-]+$",
    },
    "title": {
        "type": str,
        "required": True,
        "pattern": r"^[A-Za-z0-9-]+$",
    },
    "language": {
        "type": str,
        "required": True,
        "choices": ["python", "r"],
    },
    "kernel_type": {
        "type": str,
        "required": True,
        "choices": ["notebook", "script"],
    },
    "is_private": {
        "type": bool,
        "required": True,
    },
    "enable_gpu": {
        "type": bool,
        "required": False,
    },
    "enable_tpu": {
        "type": bool,
        "required": False,
    },
    "enable_internet": {
        "type": bool,
        "required": True,
    },
    "machine_shape": {
        "type": str,
        "required": False,
        "choices": ["NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
    },
    "dataset_sources": {
        "type": list,
        "required": False,
        "pattern": "^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$",
    },
    "competition_sources": {
        "type": list,
        "required": False,
        "pattern": "^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$",
    },
    "kernel_sources": {
        "type": list,
        "required": False,
        "pattern": "^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$",
    },
    "model_sources": {
        "type": list,
        "required": False,
        "pattern": "^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$",
    },
    "keywords": {
        "type": list,
        "required": False,
        # "pattern": ,
    },
    "sha": {
        "type": str,
        "required": False,
    },
    "executed_at": {
        "type": str,
        "required": False,
    },
}

SOURCE_KEYS = (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
)

class ValidationError(Exception):
    pass

def validate_config(config: dict) -> None:
    """Validate that all config keys are recognized."""

    for key in config:
        if key not in CONFIG_SCHEMA:
            suggestion = get_close_matches(key, CONFIG_SCHEMA, n=1, cutoff=0.6)

            if suggestion:
                raise ValidationError(
                    f"Unknown configuration key '{key}'. "
                    f"Did you mean '{suggestion[0]}'?"
                )

            raise ValidationError(
                f"Unknown config key '{key}'"
                "No similar config key was found."
            )

def validate_required_config(config: dict) -> None:
    """Raise an error if one or more required configuration keys are missing."""
    missing = [
        key
        for key, schema in CONFIG_SCHEMA.items()
        if schema["required"] and key not in config
    ]

    if missing:
        raise ValidationError(
            "Missing required configuration key(s): "
            + ", ".join(sorted(missing))
        )

def validate_types(config: dict) -> None:
    for key, value in config.items():
        expected_type = CONFIG_SCHEMA[key]["type"]

        if not isinstance(value, expected_type):
            raise ValidationError(
                f"'{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )

def validate_slug(config: dict) -> None:
    """Validate the notebook slug."""

    key = "slug"
    slug = config.get(key)
    assert isinstance(slug, str)

    pattern = CONFIG_SCHEMA[key]["pattern"]

    if not re.fullmatch(pattern, slug):
        raise ValidationError(
            "Invalid slug "
            f"'{slug}'. Slug must contain only lowercase letters, "
            "numbers, and hyphens."
        )

def validate_sources(config: dict) -> None:
    """Validate Kaggle source references."""

    for key in SOURCE_KEYS:
        sources = config.get(key)

        # Optional field
        if sources is None:
            continue

        # Detect duplicates
        if len(sources) != len(set(sources)):
            raise ValidationError(
                f"Duplicate entries found in '{key}'."
            )

        for source in sources:
            if not re.fullmatch(CONFIG_SCHEMA[key]["pattern"], source):
                raise ValidationError(
                    f"Invalid source '{source}' in '{key}'. "
                    "Expected format: 'owner/resource'."
                )

def validate_machine(config: dict) -> None:
    """Validate machine shape."""

    enable_gpu = config.get("enable_gpu")
    enable_tpu = config.get("enable_tpu")
    machine_shape = config.get("machine_shape")

    # Optional field
    if machine_shape is None:
        return

    valid_choices = CONFIG_SCHEMA["machine_shape"]["choices"]

    if enable_gpu or enable_tpu:
        if machine_shape not in valid_choices:
            raise ValidationError(
                f"Invalid machine_shape '{machine_shape}'. "
                f"Expected one of: {', '.join(valid_choices)}."
            )

def validate(config: dict) -> None:
    """Validate the complete notebook configuration."""

    # Schema validation
    validate_config(config)
    validate_required_config(config)
    validate_types(config)

    # Field validation
    validate_slug(config)
    validate_sources(config)
    validate_machine(config)

    # Cross-field validation
    if config.get("enable_gpu") and config.get("enable_tpu"):
        raise ValidationError(
            "GPU and TPU cannot both be enabled."
        )

def main() -> int:
    notebook = Path(os.environ["NOTEBOOK"])

    config = create_config(notebook)

    validate(config)

    print("Notebook configuration is valid.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
