"""Import public ground-truth labels into ATHENA's normalized label layout.

The script normalizes two released label sources used by the artifact:

* OrTHRUS DARPA/ATLAS ground-truth CSVs: one malicious entity UUID per row.
* AT03380/optc-labels tasks.zip: analyst-assigned OpTC malicious process/event
  tasks with event and object identifiers.

It does not infer ATT&CK technique labels. Technique JSON files must be added
from an explicit annotation source under data/annotated_labels/*/attack_techniques.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

ORTHRUS_E5_MAP = {
    "E5-CADETS/node_Nginx_Drakon_APT.csv": "darpa_e5/malicious_entities/cadets_nginx_drakon_apt.txt",
    "E5-CADETS/node_Nginx_Drakon_APT_17.csv": "darpa_e5/malicious_entities/cadets_nginx_drakon_apt_17.txt",
    "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv": "darpa_e5/malicious_entities/theia_firefox_drakon_apt_binfmt_elevate_inject.txt",
    "E5-TRACE/node_Trace_Firefox_Drakon.csv": "darpa_e5/malicious_entities/trace_firefox_drakon.txt",
    "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv": "darpa_e5/malicious_entities/clearscope_appstarter_0515.txt",
    "E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv": "darpa_e5/malicious_entities/clearscope_firefox_0517.txt",
    "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv": "darpa_e5/malicious_entities/clearscope_lockwatch_0517.txt",
    "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv": "darpa_e5/malicious_entities/clearscope_tester_0517.txt",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv": "darpa_e5/malicious_entities/fivedirections_bits_0515.txt",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv": "darpa_e5/malicious_entities/fivedirections_copykatz_0509.txt",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv": "darpa_e5/malicious_entities/fivedirections_dns_0517.txt",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv": "darpa_e5/malicious_entities/fivedirections_drakon_0517.txt",
}

ORTHRUS_ATLAS_MAP = {
    "atlasv2_h1/node_h1_m1.csv": "atlas/malicious_entities/h1_m1.txt",
    "atlasv2_h1/node_h1_m2.csv": "atlas/malicious_entities/h1_m2.txt",
    "atlasv2_h1/node_h1_m3.csv": "atlas/malicious_entities/h1_m3.txt",
    "atlasv2_h1/node_h1_m4.csv": "atlas/malicious_entities/h1_m4.txt",
    "atlasv2_h1/node_h1_m5.csv": "atlas/malicious_entities/h1_m5.txt",
    "atlasv2_h1/node_h1_m6.csv": "atlas/malicious_entities/h1_m6.txt",
    "atlasv2_h1/node_h1_s1.csv": "atlas/malicious_entities/h1_s1.txt",
    "atlasv2_h1/node_h1_s2.csv": "atlas/malicious_entities/h1_s2.txt",
    "atlasv2_h1/node_h1_s3.csv": "atlas/malicious_entities/h1_s3.txt",
    "atlasv2_h1/node_h1_s4.csv": "atlas/malicious_entities/h1_s4.txt",
    "atlasv2_edr/atlasv2_edr_m1.csv": "atlas/malicious_entities/edr_m1.txt",
    "atlasv2_edr/atlasv2_edr_m2.csv": "atlas/malicious_entities/edr_m2.txt",
    "atlasv2_edr/atlasv2_edr_m3.csv": "atlas/malicious_entities/edr_m3.txt",
    "atlasv2_edr/atlasv2_edr_m4.csv": "atlas/malicious_entities/edr_m4.txt",
    "atlasv2_edr/atlasv2_edr_m5.csv": "atlas/malicious_entities/edr_m5.txt",
    "atlasv2_edr/atlasv2_edr_m6.csv": "atlas/malicious_entities/edr_m6.txt",
    "atlasv2_edr/atlasv2_edr_s1.csv": "atlas/malicious_entities/edr_s1.txt",
    "atlasv2_edr/atlasv2_edr_s2.csv": "atlas/malicious_entities/edr_s2.txt",
    "atlasv2_edr/atlasv2_edr_s3.csv": "atlas/malicious_entities/edr_s3.txt",
    "atlasv2_edr/atlasv2_edr_s4.csv": "atlas/malicious_entities/edr_s4.txt",
}


def _write_ids(path: Path, values: Iterable[str]) -> int:
    ids = sorted({v.strip() for v in values if v and v.strip()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return len(ids)


def _read_first_csv_column(path: Path) -> list[str]:
    values = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                values.append(row[0].strip())
    return values


def import_orthrus(orthrus_root: Path) -> dict[str, int]:
    counts = {}
    for source_map in (ORTHRUS_E5_MAP, ORTHRUS_ATLAS_MAP):
        for src_rel, dst_rel in source_map.items():
            src = orthrus_root / src_rel
            if not src.exists():
                raise FileNotFoundError(src)
            dst = REPO_ROOT / "data" / "annotated_labels" / dst_rel
            counts[str(dst.relative_to(REPO_ROOT))] = _write_ids(dst, _read_first_csv_column(src))
    return counts


def _host_key(hostname: str) -> str:
    match = re.search(r"(\d+)", hostname or "")
    if not match:
        return "unknown"
    return match.group(1).zfill(4)


def _add_process_tree_ids(node: dict, target: set[str]) -> None:
    for key in ("event_id", "object_id", "parent_object_id"):
        value = node.get(key)
        if value:
            target.add(str(value))
    for child in node.get("children") or []:
        if isinstance(child, dict):
            _add_process_tree_ids(child, target)


def import_optc_tasks(tasks_zip: Path) -> dict[str, int]:
    if not tasks_zip.exists():
        raise FileNotFoundError(tasks_zip)
    with zipfile.ZipFile(tasks_zip) as zf:
        with zf.open("tasks.json") as f:
            tasks = json.load(f)

    per_host: dict[str, set[str]] = {}
    for task in tasks:
        labels = set(task.get("labels") or [])
        if "malicious" not in labels or "invalid" in labels:
            continue
        host = _host_key(task.get("hostname", ""))
        ids = per_host.setdefault(host, set())
        for key in ("event_id", "actor_id", "object_id", "parent_object_id"):
            value = task.get(key)
            if value:
                ids.add(str(value))
        process_table = task.get("proces_table") or task.get("process_table") or {}
        if isinstance(process_table, dict):
            _add_process_tree_ids(process_table, ids)

    counts = {}
    for host, ids in per_host.items():
        dst = REPO_ROOT / "data" / "annotated_labels" / "optc" / "malicious_entities" / f"host_{host}.txt"
        counts[str(dst.relative_to(REPO_ROOT))] = _write_ids(dst, ids)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthrus-root", type=Path, help="path to OrTHRUS Ground_Truth/orthrus directory")
    parser.add_argument("--optc-tasks-zip", type=Path, help="path to AT03380/optc-labels tasks/tasks.zip")
    args = parser.parse_args()

    result = {}
    if args.orthrus_root:
        result["orthrus"] = import_orthrus(args.orthrus_root)
    if args.optc_tasks_zip:
        result["optc_tasks"] = import_optc_tasks(args.optc_tasks_zip)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
