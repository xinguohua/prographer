"""Run complete ATHENA detection+interpretation on a source-linked E3 interval replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
E3_DATASETS = {"cadets", "theia", "trace", "clearscope"}
CONDITIONS = {"24h", "48h", "72h"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = base / path
    return local if local.exists() else REPO_ROOT / path


def _validate_source_events(path: Path, dataset: str, scene: str | None, condition: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or int(payload.get("schema_version", 0)) != 2
        or payload.get("dataset") != dataset
        or payload.get("scene") != scene
        or payload.get("condition") != condition
    ):
        raise ValueError("source event manifest dataset/scene/condition mismatch")
    boundary = payload.get("attack_event_boundaries") or {}
    boundary_path = Path(str(boundary.get("path") or ""))
    if not boundary_path.is_absolute():
        boundary_path = path.parent / boundary_path
    if not boundary_path.is_file() or _sha256(boundary_path) != boundary.get("sha256"):
        raise ValueError("attack-event boundary source path/hash mismatch")
    gaps = payload.get("replay_gaps")
    if not isinstance(gaps, list) or not gaps:
        raise ValueError("source event manifest lacks verified event gaps")
    plan = payload.get("benign_source_plan") or {}
    plan_path = Path(str(plan.get("path") or ""))
    if not plan_path.is_absolute():
        plan_path = path.parent / plan_path
    if not plan_path.is_file() or _sha256(plan_path) != plan.get("sha256"):
        raise ValueError("benign source-plan path/hash mismatch")
    from scripts.build_benign_injection_manifest import _load_source_plan
    plan_rows = _load_source_plan(plan_path, dataset, condition)
    plan_index = {
        (str(row["host"]), str(row["before_attack_event_id"]), str(row["after_attack_event_id"])): row
        for row in plan_rows
    }
    ranges = {}
    any_reuse = False
    for gap in gaps:
        stream = str(gap.get("host") or "")
        start, end = int(gap.get("source_start_timestamp", 0)), int(gap.get("source_end_timestamp", 0))
        plan_row = plan_index.get((stream, str(gap.get("attack_event_id") or ""),
                                   str(gap.get("next_attack_event_id") or "")))
        if not stream or end <= start:
            raise ValueError("replay gap lacks a valid source stream/time range")
        if plan_row is None or (
            str(gap.get("source_plan_id") or "") != str(plan_row["source_id"])
            or gap.get("reuse_policy") != plan_row.get("reuse_policy")
            or list(gap.get("source_snapshots", [])) != list(plan_row.get("source_snapshots", []))
            or list(gap.get("source_snapshot_sha256", [])) != list(plan_row.get("source_snapshot_sha256", []))
            or start != int(plan_row.get("source_start_timestamp", 0))
            or end != int(plan_row.get("source_end_timestamp", 0))
        ):
            raise ValueError("derived replay gap differs from the hashed benign source plan")
        source_ids = {int(value) for value in gap.get("source_snapshots", [])}
        overlaps = [row for row in ranges.get(stream, [])
                    if not (end <= row[0] or start >= row[1]) or source_ids.intersection(row[3])]
        reused = bool(overlaps)
        if reused and (gap.get("reuse_policy") != "allow" or any(row[2] != "allow" for row in overlaps)):
            raise ValueError("replay source reuse lacks bilateral allow policy")
        if reused != bool(gap.get("source_slice_reused")):
            raise ValueError("replay source reuse audit flag is inconsistent")
        any_reuse = any_reuse or reused
        ranges.setdefault(stream, []).append((start, end, gap.get("reuse_policy"), source_ids))
    if set(plan_index) != {
        (str(gap.get("host") or ""), str(gap.get("attack_event_id") or ""),
         str(gap.get("next_attack_event_id") or "")) for gap in gaps
    } or bool(payload.get("source_snapshot_reuse")) != any_reuse:
        raise ValueError("replay source-plan coverage/reuse summary is inconsistent")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dataset = str(spec.get("dataset") or "")
    scene = str(spec["scene"]) if spec.get("scene") not in (None, "") else None
    condition = str(spec.get("condition") or "")
    if dataset not in E3_DATASETS or condition not in CONDITIONS:
        raise ValueError("replay spec requires an E3 dataset and 24h, 48h, or 72h condition")
    base = args.spec.parent
    config = _resolve(base, str(spec.get("config") or ""))
    checkpoint = _resolve(base, str(spec.get("checkpoint") or ""))
    source_events = _resolve(base, str(spec.get("source_event_manifest") or ""))
    if not config.is_file() or not checkpoint.is_file():
        raise ValueError("replay config or frozen E3 checkpoint is missing")
    source_payload = _validate_source_events(source_events, dataset, scene, condition)
    if _sha256(config) != source_payload.get("config_sha256"):
        raise ValueError("replay config hash does not match the generated event manifest")
    boundary_path = _resolve(base, str(spec.get("attack_event_boundaries") or ""))
    if (
        not boundary_path.is_file()
        or str(boundary_path.resolve()) != str(Path(source_payload["attack_event_boundaries"]["path"]).resolve())
        or _sha256(boundary_path) != source_payload["attack_event_boundaries"]["sha256"]
    ):
        raise ValueError("replay spec does not bind the consumed attack-event boundaries")
    source_plan_path = _resolve(base, str(spec.get("benign_source_plan") or ""))
    if (
        not source_plan_path.is_file()
        or str(source_plan_path.resolve()) != str(Path(source_payload["benign_source_plan"]["path"]).resolve())
        or _sha256(source_plan_path) != source_payload["benign_source_plan"]["sha256"]
    ):
        raise ValueError("replay spec does not bind the consumed benign source plan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detection = args.output_dir / "detection.json"
    interpretation = args.output_dir / "interpretation.json"
    detection_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_detection.py"),
        "--config", str(config), "--dataset", dataset,
        "--execution", "eval-only", "--checkpoint", str(checkpoint),
        "--mode", "complete",
        "--benign-injection-manifest", str(source_events),
        "--output", str(detection),
    ]
    interpretation_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_interpretation.py"),
        "--config", str(config), "--dataset", dataset,
        "--mapping-variant", "full-enhanced", "--detections", str(detection),
        "--benign-injection-manifest", str(source_events),
        "--output", str(interpretation),
    ]
    if scene is not None:
        detection_cmd.extend(["--scene", scene])
        interpretation_cmd.extend(["--scene", scene])
    subprocess.run(detection_cmd, check=True, cwd=REPO_ROOT)
    subprocess.run(interpretation_cmd, check=True, cwd=REPO_ROOT)

    for output in (detection, interpretation):
        if not output.is_file():
            raise RuntimeError(f"complete replay did not produce {output}")
    manifest = {
        "source_id": str(spec.get("source_id") or f"{dataset}-{scene}-{condition}"),
        "source_record": str(args.spec.resolve()),
        "source_hash": _sha256(args.spec),
        "source_corpus": str(spec.get("source_corpus") or "ATHENA-E3-interval-replay"),
        "dataset": dataset,
        "scene": scene,
        "condition": condition,
        "artifacts": {
            "detection": {"path": str(detection.resolve()), "sha256": _sha256(detection)},
            "interpretation": {
                "path": str(interpretation.resolve()), "sha256": _sha256(interpretation),
            },
            "source_event_manifest": {
                "path": str(source_events.resolve()), "sha256": _sha256(source_events),
            },
            "attack_event_boundaries": {
                "path": str(boundary_path.resolve()), "sha256": _sha256(boundary_path),
            },
            "benign_source_plan": {
                "path": str(source_plan_path.resolve()), "sha256": _sha256(source_plan_path),
            },
            "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
            },
        },
    }
    output_manifest = args.output_dir / "manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output_manifest), "condition": condition}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
