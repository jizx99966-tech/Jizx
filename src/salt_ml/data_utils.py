from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_YEARLY_COLUMNS = {
    "case_id",
    "year",
    "VolShrinkRatio",
    "TotalVolumn",
    "MaxDisp",
    "MaxAbsUx",
    "MaxAbsUz",
    "MinPrincipal",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def validate_yearly_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_YEARLY_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required yearly columns: {sorted(missing)}")
