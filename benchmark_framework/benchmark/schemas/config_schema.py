from __future__ import annotations

from typing import Any

REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "run",
    "task",
    "model",
    "backend",
    "dataset",
    "pipeline",
    "benchmark",
    "profiling",
    "monitoring",
    "reporting",
)

REQUIRED_NESTED_KEYS: dict[str, tuple[str, ...]] = {
    "task": ("type",),
    "model": ("name", "format", "path", "precision", "input_size", "class_names"),
    "backend": ("type", "device"),
    "dataset": ("type", "root_dir", "split", "image_dir", "annotation_dir"),
    "reporting": ("formats",),
}


def validate_config_shape(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            errors.append(f"Missing top-level key: '{key}'")

    for section, keys in REQUIRED_NESTED_KEYS.items():
        if section not in config or not isinstance(config[section], dict):
            continue
        for key in keys:
            if key not in config[section]:
                errors.append(f"Missing key: '{section}.{key}'")

    reporting = config.get("reporting", {})
    formats = reporting.get("formats")
    if formats is not None and not isinstance(formats, list):
        errors.append("'reporting.formats' must be a list")

    task = config.get("task", {})
    task_type = task.get("type")
    if task_type is not None and task_type != "detection":
        errors.append("Only 'detection' task type is supported in MVP skeleton")

    return errors
