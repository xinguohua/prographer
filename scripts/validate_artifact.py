"""Validate released ATHENA artifact structure against the paper-facing claims.

This script checks the presence of prompt templates, curated ATT&CK resources,
released label files, and executable pipeline entry points. It intentionally
does not download raw DARPA/OpTC/ATLAS logs.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _exists(path: Path) -> bool:
    return path.exists() and (not path.is_file() or path.stat().st_size > 0)


def main() -> int:
    checks = []
    required_files = [
        "prompts/edge_mutation.txt",
        "prompts/replacement.txt",
        "prompts/rewriting.txt",
        "prompts/extension.txt",
        "configs/athena.yaml",
        "data/attack_knowledge/attackseqbench/technique_sequences.txt",
        "data/attack_knowledge/mitre_attack/technique_triples_transformed.json",
        "data/attack_knowledge/mitre_attack/technique_to_tactic.json",
        "scripts/run_augmentation.py",
        "scripts/run_detection.py",
        "scripts/run_interpretation.py",
    ]
    for rel in required_files:
        checks.append((rel, _exists(REPO_ROOT / rel)))

    label_root = REPO_ROOT / "data" / "annotated_labels"
    malicious_dirs = sorted(label_root.glob("*/malicious_entities"))
    technique_json = sorted(label_root.glob("*/attack_techniques/*.json"))
    checks.append(("malicious entity label directories", bool(malicious_dirs)))
    checks.append(("normalized attack_techniques JSON files", bool(technique_json)))

    e3_json = sorted((label_root / "darpa_e3" / "attack_techniques").glob("*.json"))
    checks.append(("DARPA E3 normalized technique JSON", bool(e3_json)))

    seq_path = REPO_ROOT / "data" / "attack_knowledge" / "attackseqbench" / "technique_sequences.txt"
    seq_count = 0
    if seq_path.exists():
        seq_count = sum(
            1 for line in seq_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    checks.append(("attack sequence library >= Top-5", seq_count >= 5))

    print(json.dumps({
        "checks": [{"name": name, "passed": passed} for name, passed in checks],
        "attack_sequence_count": seq_count,
        "malicious_label_dirs": [str(p.relative_to(REPO_ROOT)) for p in malicious_dirs],
        "normalized_technique_json": [str(p.relative_to(REPO_ROOT)) for p in technique_json],
    }, indent=2))
    return 0 if all(passed for _name, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
