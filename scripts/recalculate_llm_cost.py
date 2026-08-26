"""Recalculate an augmentation manifest's token cost from recorded usage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.llm import recalculate_llm_cost


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "athena.yaml")
    parser.add_argument("--write", action="store_true", help="update llm_usage_summary.cost in place")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    model_key = str(manifest.get("llm", ""))
    model_cfg = (cfg.get("llm", {}).get("models", {}) or {}).get(model_key)
    if not isinstance(model_cfg, dict):
        raise RuntimeError(f"manifest model key is not configured: {model_key}")
    manifest_model = str((manifest.get("llm_config") or {}).get("resolved_model") or "")
    record_models = {
        str(row.get("model")) for row in manifest.get("llm_calls", [])
        if isinstance(row, dict) and row.get("model")
    }
    if manifest_model and record_models and record_models != {manifest_model}:
        raise RuntimeError("manifest resolved_model does not match recorded LLM calls")
    served_model = manifest_model or (
        next(iter(record_models)) if len(record_models) == 1
        else str(model_cfg.get("served_model", model_key))
    )
    result = recalculate_llm_cost(
        list(manifest.get("llm_calls", [])),
        {served_model: model_cfg.get("pricing", {}) or {}},
    )
    if args.write:
        manifest.setdefault("llm_usage_summary", {})["cost"] = result
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
