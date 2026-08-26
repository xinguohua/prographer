"""Import PIDSMaker malicious-node ground truth into ATHENA.

PIDSMaker's OrTHRUS CSV rows contain a provenance-node UUID, a Python
dictionary describing the node, and a PIDSMaker database-local index.  Only
the first field is a dataset entity identifier.  This importer never treats
the third field as an event ID and never infers ATT&CK annotations.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable
from uuid import UUID


REPO_ROOT = Path(__file__).resolve().parents[1]
PIDSMaker_REPOSITORY = "https://github.com/ubc-provenance/PIDSMaker.git"
PIDSMaker_COMMIT = "32602734bc9f896be5fc0f03f0a185c967cd6624"
PIDSMaker_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"

E5_SOURCES = {
    "E5-CADETS/node_Nginx_Drakon_APT.csv": (
        "cadets_nginx_drakon_apt.txt", "cadets-nginx-drakon-apt",
        ["2019-05-16 09:31:00", "2019-05-16 10:12:00"],
    ),
    "E5-CADETS/node_Nginx_Drakon_APT_17.csv": (
        "cadets_nginx_drakon_apt_17.txt", "cadets-nginx-drakon-apt-17",
        ["2019-05-17 10:15:00", "2019-05-17 15:33:00"],
    ),
    "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv": (
        "theia_firefox_drakon_apt_binfmt_elevate_inject.txt",
        "theia-firefox-drakon-apt-binfmt-elevate-inject",
        ["2019-05-15 14:47:00", "2019-05-15 15:08:00"],
    ),
    "E5-TRACE/node_Trace_Firefox_Drakon.csv": (
        "trace_firefox_drakon.txt", "trace-firefox-drakon",
        ["2019-05-14 10:17:00", "2019-05-14 11:45:00"],
    ),
    "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv": (
        "clearscope_appstarter_0515.txt", "clearscope-appstarter-0515",
        ["2019-05-15 15:38:00", "2019-05-15 16:19:00"],
    ),
    "E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv": (
        "clearscope_firefox_0517.txt", "clearscope-firefox-0517", None,
    ),
    "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv": (
        "clearscope_lockwatch_0517.txt", "clearscope-lockwatch-0517",
        ["2019-05-17 15:48:00", "2019-05-17 16:01:00"],
    ),
    "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv": (
        "clearscope_tester_0517.txt", "clearscope-tester-0517",
        ["2019-05-17 16:20:00", "2019-05-17 16:28:00"],
    ),
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv": (
        "fivedirections_bits_0515.txt", "fivedirections-bits-0515",
        ["2019-05-15 13:14:00", "2019-05-15 13:35:00"],
    ),
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv": (
        "fivedirections_copykatz_0509.txt", "fivedirections-copykatz-0509",
        ["2019-05-09 13:25:00", "2019-05-09 13:57:00"],
    ),
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv": (
        "fivedirections_dns_0517.txt", "fivedirections-dns-0517",
        ["2019-05-17 12:46:00", "2019-05-17 12:57:00"],
    ),
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv": (
        "fivedirections_drakon_0517.txt", "fivedirections-drakon-0517",
        ["2019-05-17 16:10:00", "2019-05-17 16:16:00"],
    ),
}

# Content at the pinned PIDSMaker commit.  Checking HEAD alone is insufficient:
# a locally modified checkout must never be able to re-sign different labels.
E5_SOURCE_SHA256 = {
    "E5-CADETS/node_Nginx_Drakon_APT.csv": "2e5327b94fef4751c9edc8f530b5ff6b5cea867ef403fff33280d2672fa42c07",
    "E5-CADETS/node_Nginx_Drakon_APT_17.csv": "10c3d8f11aa11f53a1fefa7fb84519e5a7a52be31ba805a696bb92e034949918",
    "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv": "f13984a6cbdb01039605a8aa6ed1ab735bb49258a1cb543ab42bc5f1f6cd8a9b",
    "E5-TRACE/node_Trace_Firefox_Drakon.csv": "1055e300e020c1c5fc149f3a0068e8c1f779775da807ee40f8205ab36d6dedcb",
    "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv": "6f2253b0eb68f15d856d4764b034786e4eb83c5c7457ad0dfb3d7a185de6fef6",
    "E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv": "deece5b65d85b40731de0be60024bf578a2394d1353cb4bcc9c34b5790f0d3c2",
    "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv": "cef24f7ae329961dceb328d9d022fa53623a8905c2a338bcf8c60b3c7e3f562e",
    "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv": "484fd8b9689a827c5c6c8dbcd6876954f37463823a6599f6da46dcfd4d2e8567",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv": "0841d453aff67c8254cc1d4a47265ed5cc0c8d7a79c1708ca0ab2019574b8cd4",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv": "9ba7f64b45aacb670a395356c14bf2a2a992ce9fccffd6dda40065752e6130de",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv": "c4ca6910338afec0555128d15f00c9ff1bce80b44ffebfe39d84ab84f8b640e3",
    "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv": "bedd5adeda44654add866eb54752f87da2e71b8fb711018975dc015aef6e954f",
}

OPTC_SOURCES = {
    "h051/node_h051_0925.csv": {
        "output": "host_0051.txt", "host_id": "H051", "raw_host": "SysClient0051",
        "attack_id": "optc-h051-2019-09-25", "expected_count": 114,
        "window": ["2019-09-25 10:29:00", "2019-09-25 14:25:00"],
        "source_sha256": "ff8af2562c6746b48f81445fa36a5860ebd9a4402fa6b83cd47ddda35bfdeb3b",
    },
    "h201/node_h201_0923.csv": {
        "output": "host_0201.txt", "host_id": "H201", "raw_host": "SysClient0201",
        "attack_id": "optc-h201-2019-09-23", "expected_count": 2905,
        "window": ["2019-09-23 11:23:00", "2019-09-23 13:25:00"],
        "source_sha256": "b43c03d547b8d00b27a7b6faee6efdec4d4dcb47d992419147ce80363ad26d0e",
    },
    "h501/node_h501_0924.csv": {
        "output": "host_0501.txt", "host_id": "H501", "raw_host": "SysClient0501",
        "attack_id": "optc-h501-2019-09-24", "expected_count": 749,
        "window": ["2019-09-24 10:28:00", "2019-09-24 15:29:00"],
        "source_sha256": "16ed69a63742431d35bed1ca3ff27a314a21c3ef673eaaf6a346b1410abd5a42",
    },
}


def _e5_metadata() -> dict:
    return {
        source: {
            "output": values[0], "attack_id": values[1], "window": values[2],
            "source_sha256": E5_SOURCE_SHA256[source],
        }
        for source, values in E5_SOURCES.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _write_ids(path: Path, values: Iterable[str]) -> int:
    ids = sorted({str(value).strip() for value in values if str(value).strip()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return len(ids)


def _joinable_node_id(value: str) -> bool:
    return str(value).strip().lower() != "00000000-0000-0000-0000-000000000000"


def _read_source_rows(path: Path) -> list[dict]:
    rows, seen = [], set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for source_row, row in enumerate(csv.reader(stream), 1):
            if len(row) != 3:
                raise ValueError(f"{path}:{source_row}: expected exactly three CSV fields")
            node_id_raw = row[0].strip()
            try:
                UUID(node_id_raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{source_row}: invalid node UUID {node_id_raw!r}") from exc
            try:
                attributes = ast.literal_eval(row[1])
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"{path}:{source_row}: invalid node-attribute dictionary") from exc
            if not isinstance(attributes, dict):
                raise ValueError(f"{path}:{source_row}: node attributes must be a dictionary")
            try:
                export_index = int(row[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{source_row}: PIDSMaker export index must be an integer") from exc
            canonical = node_id_raw.lower()
            if canonical in seen:
                raise ValueError(f"{path}:{source_row}: duplicate UUID {node_id_raw}")
            seen.add(canonical)
            rows.append({
                "node_id_raw": node_id_raw,
                "node_id_canonical": canonical,
                "node_attributes": attributes,
                "pidsmaker_export_index": export_index,
                "source_row": source_row,
            })
    return rows


def _verify_checkout(root: Path) -> None:
    license_path = root / "LICENSE"
    if not license_path.is_file() or _sha256(license_path) != PIDSMaker_LICENSE_SHA256:
        raise RuntimeError("PIDSMaker LICENSE does not match the pinned checkout")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve PIDSMaker source commit") from exc
    if commit != PIDSMaker_COMMIT:
        raise RuntimeError(f"PIDSMaker checkout must be {PIDSMaker_COMMIT}; found {commit}")


def _import_dataset(dataset: str, sources: dict, orthrus_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, source_manifest, output_paths = [], [], []
    for source_rel, metadata in sources.items():
        source_path = orthrus_root / source_rel
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha256 = _sha256(source_path)
        expected_sha = metadata.get("source_sha256")
        if expected_sha and source_sha256 != expected_sha:
            raise RuntimeError(f"source hash mismatch for {source_rel}")
        rows = _read_source_rows(source_path)
        expected_count = metadata.get("expected_count")
        if expected_count is not None and len(rows) != int(expected_count):
            raise RuntimeError(f"{source_rel}: expected {expected_count} nodes, found {len(rows)}")
        output_path = output_dir / metadata["output"]
        _write_ids(
            output_path,
            (row["node_id_raw"] for row in rows if _joinable_node_id(row["node_id_raw"])),
        )
        output_paths.append(output_path)
        for row in rows:
            record = {
                "record_type": "malicious_entity",
                "dataset": dataset,
                "attack_id": metadata["attack_id"],
                "node_id_raw": row["node_id_raw"],
                "node_id_canonical": row["node_id_canonical"],
                "id_namespace": (
                    "optc.actor_or_object.uuid" if dataset == "optc"
                    else "darpa.cdm.provenance_node.uuid"
                ),
                "node_attributes": row["node_attributes"],
                "pidsmaker_export_index": row["pidsmaker_export_index"],
                "joinable": row["node_id_canonical"] != "00000000-0000-0000-0000-000000000000",
                "attack_window_local": metadata.get("window"),
                "attack_window_timezone": "America/New_York",
                "source_path": f"Ground_Truth/orthrus/{source_rel}",
                "source_row": row["source_row"],
                "source_file_sha256": source_sha256,
                "source_repository": PIDSMaker_REPOSITORY,
                "source_commit": PIDSMaker_COMMIT,
            }
            if metadata.get("host_id"):
                record.update(host_id=metadata["host_id"], raw_host=metadata["raw_host"])
            records.append(record)
        source_manifest.append({
            "source_path": f"Ground_Truth/orthrus/{source_rel}",
            "source_sha256": source_sha256,
            "record_count": len(rows),
            "attack_id": metadata["attack_id"],
            "host_id": metadata.get("host_id"),
            "raw_host": metadata.get("raw_host"),
            "attack_window_local": metadata.get("window"),
            "attack_window_timezone": "America/New_York",
            "output": metadata["output"],
            "output_sha256": _sha256(output_path),
        })
    records.sort(key=lambda row: (
        str(row.get("host_id", "")), row["attack_id"], row["node_id_canonical"], row["source_row"],
    ))
    entities_path = output_dir / "entities.jsonl"
    entities_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "record_type": "pidsmaker_malicious_entity_registry",
        "dataset": dataset,
        "source_repository": PIDSMaker_REPOSITORY,
        "source_commit": PIDSMaker_COMMIT,
        "source_license": {"path": "LICENSE", "sha256": PIDSMaker_LICENSE_SHA256},
        "converter": {"path": "scripts/import_ground_truth_labels.py", "sha256": _sha256(Path(__file__))},
        "sources": source_manifest,
        "entity_records": {"path": "entities.jsonl", "count": len(records), "sha256": _sha256(entities_path)},
        "output_files": [
            {"path": path.name, "sha256": _sha256(path),
             "count": len(path.read_text(encoding="utf-8").splitlines())}
            for path in sorted(output_paths)
        ],
    }
    manifest["aggregate_sha256"] = _canonical_hash(manifest)
    (output_dir / "content_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def import_pidsmaker(
    pidsmaker_root: Path, *, output_root: Path | None = None,
    verify_checkout: bool = True, remove_legacy_optc_hosts: bool = True,
) -> dict:
    root = pidsmaker_root.resolve()
    if verify_checkout:
        _verify_checkout(root)
    orthrus_root = root / "Ground_Truth" / "orthrus"
    if not orthrus_root.is_dir():
        raise FileNotFoundError(orthrus_root)
    output_root = output_root or REPO_ROOT / "data" / "annotated_labels"
    result = {
        "darpa_e5": _import_dataset(
            "darpa_e5", _e5_metadata(), orthrus_root,
            output_root / "darpa_e5" / "malicious_entities",
        ),
        "optc": _import_dataset(
            "optc", OPTC_SOURCES, orthrus_root,
            output_root / "optc" / "malicious_entities",
        ),
    }
    if remove_legacy_optc_hosts:
        optc_dir = output_root / "optc" / "malicious_entities"
        keep = {metadata["output"] for metadata in OPTC_SOURCES.values()}
        removed = []
        for path in sorted(optc_dir.glob("host_*.txt")):
            if path.name not in keep:
                path.unlink()
                removed.append(path.name)
        result["removed_legacy_optc_host_files"] = removed
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pidsmaker-root", type=Path, required=True,
        help="pinned PIDSMaker checkout containing Ground_Truth/orthrus",
    )
    parser.add_argument(
        "--keep-legacy-optc-hosts", action="store_true",
        help="retain non-paper-profile host label files (not used by ATHENA)",
    )
    args = parser.parse_args(argv)
    result = import_pidsmaker(
        args.pidsmaker_root,
        remove_legacy_optc_hosts=not args.keep_legacy_optc_hosts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
