"""Utilities for validating TraffCOCO YOLO dataset YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_dataset_yaml(data_yaml: str | Path) -> dict[str, Any]:
    data_yaml = Path(data_yaml).resolve()

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML does not exist: {data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid YOLO dataset YAML.")

    required = {"train", "val", "names"}
    missing = required - data.keys()

    if missing:
        raise ValueError(
            f"Missing required dataset fields: {sorted(missing)}"
        )

    return data


def get_dataset_root(
    data_yaml: str | Path,
    data: dict[str, Any],
) -> Path:
    data_yaml = Path(data_yaml).resolve()
    root = Path(data.get("path", "."))

    if not root.is_absolute():
        root = data_yaml.parent / root

    return root.resolve()


def resolve_split_path(
    data_yaml: str | Path,
    data: dict[str, Any],
    split: str,
) -> Path:
    root = get_dataset_root(data_yaml, data)
    split_path = Path(data[split])

    if not split_path.is_absolute():
        split_path = root / split_path

    return split_path.resolve()


def validate_split(
    data_yaml: str | Path,
    data: dict[str, Any],
    split: str,
) -> None:
    if split not in data:
        raise ValueError(
            f"Dataset does not define a '{split}' split."
        )

    path = resolve_split_path(data_yaml, data, split)

    if not path.exists():
        raise FileNotFoundError(
            f"{split.upper()} path does not exist: {path}"
        )

    if not path.is_dir():
        raise ValueError(
            f"{split.upper()} path is not a directory: {path}"
        )


def validate_dataset(data_yaml: str | Path) -> dict[str, Any]:
    data = load_dataset_yaml(data_yaml)

    for split in ("train", "val"):
        validate_split(data_yaml, data, split)

    if "test" in data:
        validate_split(data_yaml, data, "test")

    names = data["names"]

    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        class_names = [
            names[k]
            for k in sorted(names, key=lambda x: int(x))
        ]
    else:
        raise ValueError("'names' must be a list or dictionary.")

    if not class_names:
        raise ValueError("No classes defined in dataset.")

    print(f"Dataset: {data_yaml}")
    print(f"Classes: {len(class_names)}")

    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    return data
