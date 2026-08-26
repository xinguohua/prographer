"""YAML configuration loader for ATHENA.

Reads ``configs/athena.yaml`` (or any path passed to :func:`load_config`) and
returns a nested ``dict``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in requirements.txt
    yaml = None  # type: ignore


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "athena.yaml"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required: pip install pyyaml")
    p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
