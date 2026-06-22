"""YAML configuration loader for ATHENA.

Reads ``configs/athena.yaml`` (or any path passed to :func:`load_config`) and
returns a nested ``dict``. Also re-exports the constant names used by older
modules so they keep working unchanged.
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


# Constants reproduced from the YAML so existing call sites keep working.
_cfg = {}
try:
    _cfg = load_config()
except Exception:
    pass

MALICIOUS_WINDOW_MINUTES = int(_cfg.get("snapshot", {}).get("malicious_window_minutes", 10))
TEST_WINDOW_MINUTES = int(_cfg.get("snapshot", {}).get("test_window_minutes", 20))
SNAPSHOT_SIZE = int(_cfg.get("snapshot", {}).get("snapshot_size", 500))
FORGETTING_RATE = float(_cfg.get("snapshot", {}).get("forgetting_rate", 0.2))

SEQUENCE_LENGTH_L = int(_cfg.get("sequence", {}).get("length", 12))
MIN_SEQUENCE_LENGTH = int(_cfg.get("sequence", {}).get("min_length", 3))
SEQUENCE_ADAPT_RATIO = float(_cfg.get("sequence", {}).get("adapt_ratio", 0.5))

DETECTION_THRESHOLD = float(_cfg.get("detection", {}).get("threshold", 0.01))


def get_time_split_config() -> Dict[str, Any]:
    """Return the subset of config consumed by the snapshot-window splitter."""
    return {
        "malicious_window_minutes": MALICIOUS_WINDOW_MINUTES,
        "test_window_minutes":      TEST_WINDOW_MINUTES,
        "sequence_length":          SEQUENCE_LENGTH_L,
        "min_sequence_length":      MIN_SEQUENCE_LENGTH,
        "sequence_adapt_ratio":     SEQUENCE_ADAPT_RATIO,
        "detection_threshold":      DETECTION_THRESHOLD,
        "snapshot_size":            SNAPSHOT_SIZE,
        "forgetting_rate":          FORGETTING_RATE,
    }
