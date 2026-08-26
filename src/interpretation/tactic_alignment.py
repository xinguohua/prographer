"""Technique → tactic aggregation and tactic-sequence full-match alignment.

The semantic matcher emits parent-level MITRE ATT&CK techniques (e.g. T1071);
the global interpretation stage aggregates them to the coarser ATT&CK tactic
layer (e.g. ``Command and Control``) and aligns the resulting tactic sequence
against the attack-sequence library with multi-stage full-match filtering.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .global_alignment import best_library_match, lcs_length

REPO_ROOT = Path(__file__).resolve().parents[2]

# Parent-technique → ATT&CK tactic. Built from the official MITRE matrix; only
# the techniques referenced in the released annotated labels and the curated
# attack-sequence library are listed here. Sub-techniques (e.g. T1547/004)
# collapse to their parent (T1547).
DEFAULT_TECH_TO_TACTIC: Dict[str, str] = {
    # Initial Access
    "T1078": "Initial Access",
    "T1189": "Initial Access",
    "T1190": "Initial Access",
    "T1133": "Initial Access",
    "T1199": "Initial Access",
    "T1566": "Initial Access",
    # Execution
    "T1059": "Execution",
    "T1106": "Execution",
    "T1129": "Execution",
    "T1203": "Execution",
    "T1204": "Execution",
    # Persistence
    "T1098": "Persistence",
    "T1136": "Persistence",
    "T1505": "Persistence",
    "T1543": "Persistence",
    "T1546": "Persistence",
    "T1547": "Persistence",
    # Privilege Escalation
    "T1055": "Defense Evasion",
    "T1068": "Privilege Escalation",
    "T1134": "Privilege Escalation",
    "T1484": "Privilege Escalation",
    "T1548": "Privilege Escalation",
    # Defense Evasion
    "T1027": "Defense Evasion",
    "T1036": "Defense Evasion",
    "T1070": "Defense Evasion",
    "T1140": "Defense Evasion",
    "T1218": "Defense Evasion",
    "T1222": "Defense Evasion",
    # Credential Access
    "T1003": "Credential Access",
    "T1110": "Credential Access",
    "T1555": "Credential Access",
    # Discovery
    "T1007": "Discovery",
    "T1016": "Discovery",
    "T1018": "Discovery",
    "T1033": "Discovery",
    "T1057": "Discovery",
    "T1082": "Discovery",
    "T1083": "Discovery",
    "T1087": "Discovery",
    # Lateral Movement
    "T1021": "Lateral Movement",
    "T1080": "Lateral Movement",
    "T1210": "Lateral Movement",
    # Collection
    "T1005": "Collection",
    "T1056": "Collection",
    "T1074": "Collection",
    "T1113": "Collection",
    "T1114": "Collection",
    "T1115": "Collection",
    "T1119": "Collection",
    "T1185": "Collection",
    "T1213": "Collection",
    "T1530": "Collection",
    "T1560": "Collection",
    # Command and Control
    "T1071": "Command and Control",
    "T1090": "Command and Control",
    "T1095": "Command and Control",
    "T1102": "Command and Control",
    "T1104": "Command and Control",
    "T1105": "Command and Control",
    "T1132": "Command and Control",
    "T1205": "Command and Control",
    "T1219": "Command and Control",
    "T1573": "Command and Control",
    # Exfiltration
    "T1011": "Exfiltration",
    "T1020": "Exfiltration",
    "T1041": "Exfiltration",
    "T1048": "Exfiltration",
    "T1052": "Exfiltration",
    # Impact
    "T1485": "Impact",
    "T1486": "Impact",
    "T1490": "Impact",
    "T1496": "Impact",
    "T1499": "Impact",
    "T1531": "Impact",
}


_TECH_ROOT_RE = re.compile(r"^(T\d{4})")


def normalize_tech_id(tech_id: str) -> str:
    """Reduce sub-technique IDs like ``T1547/004`` or ``T1547.004`` to the
    parent technique ``T1547``."""
    if not tech_id:
        return ""
    m = _TECH_ROOT_RE.match(tech_id.strip())
    return m.group(1) if m else tech_id.strip()


_MITRE_TACTIC_JSON = (
    REPO_ROOT / "data" / "attack_knowledge" / "mitre_attack" / "technique_to_tactic.json"
)


def _load_mitre_tactic_json() -> Dict[str, str]:
    """Load ``technique_to_tactic.json`` (built from the MITRE STIX bundle).

    The JSON stores ``Dict[tech_id, List[tactic]]`` because some techniques
    span multiple tactics. This compatibility helper selects the first
    deterministically stored tactic and folds sub-techniques to their parent;
    callers that require every valid tactic use :func:`load_tech_to_tactics`.
    """
    mapping: Dict[str, str] = {}
    if not _MITRE_TACTIC_JSON.exists():
        return mapping
    try:
        with _MITRE_TACTIC_JSON.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return mapping
    for raw_id, tactics in payload.items():
        parent = normalize_tech_id(raw_id)
        if not parent or not tactics:
            continue
        primary = tactics[0] if isinstance(tactics, list) else str(tactics)
        if isinstance(primary, str) and primary:
            mapping.setdefault(parent, primary)
    return mapping


def load_tech_to_tactic(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return the single-tactic compatibility mapping used by legacy callers.

    Precedence (later overrides earlier):
    1. Hardcoded :data:`DEFAULT_TECH_TO_TACTIC` compatibility entries.
    2. The released MITRE STIX-derived ``technique_to_tactic.json``.
    3. Legacy per-scene ``attack_techniques/*.json`` reference annotations.
    4. ``extra`` overrides passed by the caller.
    """
    mapping = dict(DEFAULT_TECH_TO_TACTIC)
    for tech, tactic in _load_mitre_tactic_json().items():
        mapping.setdefault(tech, tactic)
    for dataset_dir in (REPO_ROOT / "data" / "annotated_labels").glob("*/attack_techniques"):
        for json_path in dataset_dir.glob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            entities = payload.get("entities") if isinstance(payload, dict) else None
            if not isinstance(entities, dict):
                continue
            for record in entities.values():
                if not isinstance(record, dict):
                    continue
                tech = normalize_tech_id(str(record.get("technique", "")))
                tactic = str(record.get("tactic", "")).strip()
                if tech and tactic:
                    mapping[tech] = tactic
    if extra:
        mapping.update(extra)
    return mapping


def load_tech_to_tactics() -> Dict[str, List[str]]:
    """Return every associated tactic for each parent technique."""
    mapping: Dict[str, List[str]] = {
        technique: [tactic] for technique, tactic in DEFAULT_TECH_TO_TACTIC.items()
    }
    if _MITRE_TACTIC_JSON.exists():
        try:
            with _MITRE_TACTIC_JSON.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            payload = {}
        for raw_id, raw_tactics in payload.items():
            technique = normalize_tech_id(str(raw_id))
            tactics = raw_tactics if isinstance(raw_tactics, list) else [raw_tactics]
            clean = [str(tactic).strip() for tactic in tactics if str(tactic).strip()]
            if technique and clean:
                mapping.setdefault(technique, [])
                for tactic in clean:
                    if tactic not in mapping[technique]:
                        mapping[technique].append(tactic)
    for dataset_dir in (REPO_ROOT / "data" / "annotated_labels").glob("*/attack_techniques"):
        for json_path in dataset_dir.glob("*.json"):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entities = payload.get("entities", {}) if isinstance(payload, dict) else {}
            for record in entities.values() if isinstance(entities, dict) else []:
                if not isinstance(record, dict):
                    continue
                technique = normalize_tech_id(str(record.get("technique", "")))
                tactic = str(record.get("tactic", "")).strip()
                if technique and tactic:
                    mapping.setdefault(technique, [])
                    if tactic not in mapping[technique]:
                        mapping[technique].append(tactic)
    return mapping


def techniques_to_tactics(
    techniques: List[str],
    mapping: Optional[Dict[str, str]] = None,
    collapse_adjacent: bool = True,
) -> List[str]:
    """Convert a technique sequence into the corresponding tactic sequence.

    Sub-technique IDs are folded to their parent; unknown techniques are
    dropped. If ``collapse_adjacent`` (default), consecutive duplicate
    tactics are reduced to a single occurrence.
    """
    mapping = mapping if mapping is not None else load_tech_to_tactic()
    out: List[str] = []
    for raw in techniques:
        tech = normalize_tech_id(raw)
        tactic = mapping.get(tech)
        if not tactic:
            continue
        if collapse_adjacent and out and out[-1] == tactic:
            continue
        out.append(tactic)
    return out


def load_tactic_sequence_library(
    technique_seq_path: Optional[str] = None,
    mapping: Optional[Dict[str, str]] = None,
) -> List[List[str]]:
    """Produce a tactic-level attack-sequence library by mapping each
    technique in the existing technique-sequence library to its tactic and
    collapsing adjacent duplicates."""
    from .global_alignment import load_technique_sequence_library
    tech_lib = load_technique_sequence_library(technique_seq_path)
    mapping = mapping if mapping is not None else load_tech_to_tactic()
    out: List[List[str]] = []
    for seq in tech_lib:
        tactic_seq = techniques_to_tactics(seq, mapping=mapping, collapse_adjacent=True)
        if tactic_seq:
            out.append(tactic_seq)
    return out


def load_tactic_sequence_records(
    technique_seq_path: Optional[str] = None,
    mapping: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Load record-local ordered tactics without losing provenance.

    AttackSeqBench's grouped records are the authoritative source for tactic
    order.  A record-local ``technique_tactics`` mapping is accepted only as a
    fallback for an older source-linked export that does not already carry a
    ``tactics`` list.  The global primary-tactic map is never used to rewrite
    an official record.
    """
    from .global_alignment import load_technique_sequence_records
    records = load_technique_sequence_records(technique_seq_path)
    output = []
    for record in records:
        raw_tactics = record.get("tactics")
        if isinstance(raw_tactics, list) and raw_tactics:
            tactics = [str(value).strip() for value in raw_tactics if str(value).strip()]
        else:
            local = record.get("technique_tactics")
            if not isinstance(local, list) or not local:
                raise ValueError(
                    f"attack-sequence record {record['source_id']} lacks ordered tactics"
                )
            tactics = []
            for item in local:
                if not isinstance(item, dict) or not str(item.get("tactic", "")).strip():
                    raise ValueError(
                        f"attack-sequence record {record['source_id']} has invalid technique_tactics"
                    )
                tactic = str(item["tactic"]).strip()
                if not tactics or tactics[-1] != tactic:
                    tactics.append(tactic)
        if tactics:
            output.append({**record, "tactics": tactics})
    return output


def best_tactic_match(
    predicted_tactics: List[str],
    tactic_library: Optional[List[List[str]]] = None,
    min_ratio: float = 0.60,
):
    """Convenience wrapper around :func:`best_library_match` for tactic
    sequences. Returns ``(best_library_sequence, full_match_score)`` or ``(None,
    best_ratio)`` if no library sequence meets ``min_ratio``."""
    if tactic_library is None:
        tactic_library = load_tactic_sequence_library()
    return best_library_match(predicted_tactics, tactic_library, min_ratio=min_ratio)


__all__ = [
    "DEFAULT_TECH_TO_TACTIC",
    "normalize_tech_id",
    "load_tech_to_tactic",
    "load_tech_to_tactics",
    "techniques_to_tactics",
    "load_tactic_sequence_library",
    "load_tactic_sequence_records",
    "best_tactic_match",
]
