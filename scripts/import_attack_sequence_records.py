"""Import source-linked AttackSeqBench grouped attack-sequence records."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interpretation.global_alignment import load_technique_sequence_records

OFFICIAL_SOURCE_URL = "https://anonymous.4open.science/r/AttackSeqBench"
ARXIV_URL = "https://arxiv.org/abs/2503.03170"
TECHNIQUE_PREFIX = re.compile(r"^(T\d{4}(?:\.\d{3})?)(?:-|$)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_grouped_attackseqs(raw_root: Path, archive_sha256: str) -> list[dict]:
    grouped = raw_root / "question_generation" / "grouped_attackseqs"
    paths = sorted(grouped.glob("*.json"), key=lambda path: int(path.stem))
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups = payload.get("triplet_groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"{path} lacks ordered triplet_groups")
        techniques, tactics, mapping, seen = [], [], [], set()
        for tactic, technique_groups in groups.items():
            if not isinstance(technique_groups, dict):
                raise ValueError(f"{path} tactic {tactic!r} is not an object")
            tactic = str(tactic).strip()
            if tactic and tactic not in tactics:
                tactics.append(tactic)
            for raw_technique in technique_groups:
                match = TECHNIQUE_PREFIX.match(str(raw_technique).strip())
                if not match:
                    raise ValueError(f"{path} has invalid technique key {raw_technique!r}")
                technique = match.group(1).upper()
                if technique not in seen:
                    seen.add(technique)
                    techniques.append(technique)
                    mapping.append({"technique": technique, "tactic": tactic})
        if not techniques:
            raise ValueError(f"{path} contains no ATT&CK techniques")
        relative = path.relative_to(raw_root).as_posix()
        records.append({
            "source_id": f"attackseqbench-grouped-{int(path.stem):04d}",
            "source_corpus": "AttackSeqBench",
            "source_record": relative,
            "source_hash": _sha256(path),
            "techniques": techniques,
            "tactics": tactics,
            "technique_tactics": mapping,
            "metadata": {
                "file_name": str(payload.get("file_name", "")),
                "tactic_label": bool(payload.get("tactic_label", False)),
                "technique_label": bool(payload.get("technique_label", False)),
                # A download-level checksum records this particular retrieval;
                # per-record hashes and the deterministic content manifest are
                # the stable verification contract for the derived corpus.
                "retrieval_archive_sha256": archive_sha256,
                "source_url": OFFICIAL_SOURCE_URL,
                "paper_url": ARXIV_URL,
            },
        })
    return records


def _write_records(records: list[dict], output: Path) -> None:
    if not records:
        raise RuntimeError("source contains no attack-sequence records")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _content_manifest(records: list[dict]) -> dict:
    content = []
    for record in records:
        stable_metadata = {
            key: value for key, value in (record.get("metadata") or {}).items()
            if key != "retrieval_archive_sha256"
        }
        derived = {
            "source_id": str(record["source_id"]),
            "source_corpus": str(record["source_corpus"]),
            "source_record": str(record["source_record"]),
            "source_hash": str(record["source_hash"]).lower(),
            "techniques": list(record.get("techniques") or []),
            "tactics": list(record.get("tactics") or []),
            "technique_tactics": list(record.get("technique_tactics") or []),
            "metadata": stable_metadata,
        }
        canonical_record = json.dumps(
            derived, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        content.append({
            **derived,
            "derived_sha256": hashlib.sha256(canonical_record).hexdigest(),
        })
    canonical = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    retrieval_hashes = sorted({
        str((record.get("metadata") or {}).get("retrieval_archive_sha256") or "")
        for record in records
        if (record.get("metadata") or {}).get("retrieval_archive_sha256")
    })
    return {
        "schema_version": 1,
        "source_corpus": "AttackSeqBench",
        "record_count": len(content),
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "records": content,
        "retrieval": {
            "source_url": OFFICIAL_SOURCE_URL,
            "paper_url": ARXIV_URL,
            "archive_sha256": retrieval_hashes[0] if len(retrieval_hashes) == 1 else None,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-root", type=Path)
    source.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--expected-records", type=int, default=408)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data" / "attack_knowledge" / "attackseqbench" / "verified_sequences.jsonl",
    )
    parser.add_argument("--content-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.raw_root:
        if args.source_archive is None:
            raise ValueError("--raw-root requires --source-archive for archive provenance")
        archive_hash = _sha256(args.source_archive)
        records = convert_grouped_attackseqs(args.raw_root, archive_hash)
    else:
        records = load_technique_sequence_records(str(args.source_jsonl))
    if len(records) != int(args.expected_records):
        raise ValueError(
            f"expected {args.expected_records} AttackSeqBench records, received {len(records)}"
        )
    _write_records(records, args.output)
    manifest_path = args.content_manifest or args.output.with_name("content_manifest.json")
    manifest = _content_manifest(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "records": len(records),
        "content_manifest": str(manifest_path),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "retrieval_archive_sha256": manifest["retrieval"]["archive_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
