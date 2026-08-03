"""
based on sentence-transformers 's  ATT&CK semanticmapping. 

responsibility: 
1. log: willsnapshotin's uselogis
2. library: from's level3tuplebuildlibrary
3. vectorretrieve: use Sentence-BERT encodeaftercosinesimilarity, retrievemost similar's 

fromsource: process/data/technique_triples_transformed.json
logthen: process/translation_rules.py
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import json
import os
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.interpretation.attack_sequence import (
    TYPE_MAP, EVENT_MAP, LOW_INFO_PREFIXES,
    translate_event, get_process_role,
)


# ============================================================
# ============================================================

def snapshot_to_query(snapshot, *, node_scope: str = "malicious", max_nodes: int = 200) -> str:
    """willsnapshotgraphisquerytext. 

    outputandconsistent: 
    "subject verb object. subject verb object. ..."

    for example: command shell writes shared library. command shell sends network connection.
    """
    nodes = []
    for v in snapshot. vs :
        attrs = v.attributes()
        if node_scope == "malicious":
            try:
                if int(attrs.get("label", 0)) != 1:
                    continue
            except Exception:
                continue
        node_type = str(attrs.get("type") or attrs.get("type_name") or "")
        props = str(attrs.get("properties") or "")
        freq = attrs.get("frequency", 0)
        try:
            freq = int(freq)
        except Exception:
            freq = 0
        nodes.append({"type": node_type, "properties": props, "frequency": freq})

    nodes.sort(key=lambda d: d["frequency"], reverse=True)
    nodes = nodes[:max_nodes]

    triples = []
    seen = set()
    for n in nodes:
        raw_type = n["type"].strip()

        if raw_type != "SUBJECT_PROCESS":
            continue

        proc_name = _extract_process_name(n["properties"])
        subject = get_process_role(proc_name) if proc_name else "process"

        events_raw = n["properties"].strip("{} '\"")
        event_items = [e.strip().strip("'\"") for e in events_raw.split(",") if e.strip()]

        for e in event_items:
            t = translate_event(e)
            if not t or any(t.startswith(p) for p in LOW_INFO_PREFIXES):
                continue
            #  "subject verb object" 's 3tuple
            triple = f"{subject} {t}"
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)

    return ". ".join(triples) if triples else ""


def path_edges_to_query(snapshot, key_path, max_edges: int = 80) -> str:
    """Serialize extracted causal-path edges into ATT&CK matching text."""
    triples = []
    seen = set()
    for src, dst, action in list(key_path)[:max_edges]:
        try:
            src_attrs = snapshot.vs[int(src)].attributes()
            dst_attrs = snapshot.vs[int(dst)].attributes()
        except Exception:
            continue
        src_type = str(src_attrs.get("type") or src_attrs.get("type_name") or "entity")
        dst_type = str(dst_attrs.get("type") or dst_attrs.get("type_name") or "entity")
        src_proc = _extract_process_name(str(src_attrs.get("properties") or ""))
        dst_proc = _extract_process_name(str(dst_attrs.get("properties") or ""))
        subject = get_process_role(src_proc) if src_proc else TYPE_MAP.get(src_type, src_type.lower())
        obj = get_process_role(dst_proc) if dst_proc else TYPE_MAP.get(dst_type, dst_type.lower())
        translated = translate_event(str(action)) or str(action).lower()
        triple = f"{subject} {translated} {obj}".strip()
        if triple and triple not in seen:
            seen.add(triple)
            triples.append(triple)
    return ". ".join(triples)


# ============================================================
# ============================================================

def _load_technique_descriptions(
    json_path: str,
) -> Dict[str, str]:
    """fromconvertafter's 3tuple JSON buildlibrary. 

    each's 3tuple [{subject, verb, object}, ...] concatenateis1segment: 
    "subject verb object. subject verb object. ..."
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = {}
    for tech_id, triples in data.items():
        if not triples:
            continue
        parts = []
        for t in triples:
            parts.append(f"{t['subject']} {t['verb']} {t['object']}")
        descriptions[tech_id] = ". ".join(parts)

    return descriptions


# ============================================================
# ============================================================

class TechniqueSemanticMapper:
    """ATT&CK semanticmatch. 

    use Sentence-BERT tologqueryandrowencode, 
    viacosinesimilarityretrievemost match's . 
    """

    def __init__(
        self,
        *,
        triples_path: str = os.path.join(
            os.path.dirname(__file__), "data/technique_triples_raw.json"
        ),
        aux_triples_path: str = None,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        top_k: int = 5,
        threshold: float = 0.0,
        aux_weight: float = 0.3,
        **kwargs,
    ) -> None:
        self.triples_path = triples_path
        self.model_name = model_name
        self.top_k = int(max(1, top_k))
        self.threshold = threshold
        self.aux_weight = aux_weight

        self._tech_descs = _load_technique_descriptions(triples_path)
        self._tech_ids = list(self._tech_descs.keys())
        self._tech_texts = [self._tech_descs[tid] for tid in self._tech_ids]

        # load Sentence-BERT modulo
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "needs sentence-transformers: pip install sentence-transformers"
            )

        print(f"[SemMapper] encode {len(self._tech_ids)} ...")
        self._tech_embeddings = self._model.encode(
            self._tech_texts, show_progress_bar=False, normalize_embeddings=True
        )

        self._aux_embeddings = None
        if aux_triples_path and os.path.exists(aux_triples_path):
            aux_descs = _load_technique_descriptions(aux_triples_path)
            # primarylibrary's  tech_ids orderalign
            aux_texts = [aux_descs.get(tid, "") for tid in self._tech_ids]
            print(f"[SemMapper] encode {sum(1 for t in aux_texts if t)} ...")
            self._aux_embeddings = self._model.encode(
                aux_texts, show_progress_bar=False, normalize_embeddings=True
            )

        print(f"[SemMapper] library. ")

    def snapshot_to_query(self, snap) -> str:
        """willsnapshotisquerytext. """
        return snapshot_to_query(snap)

    def _compute_similarities(self, q_emb: np.ndarray) -> np.ndarray:
        """queryandlibrary's similarity. """
        sim_main = np.dot(self._tech_embeddings, q_emb.T).flatten()
        if self._aux_embeddings is not None:
            sim_aux = np.dot(self._aux_embeddings, q_emb.T).flatten()
            w = self.aux_weight
            return (1 - w) * sim_main + w * sim_aux
        return sim_main

    def predict_top(self, query: str) -> Optional[Tuple[str, float]]:
        """returnmost match's  (mitre_id, cosine_similarity). 

        similarity the morelargethe moresimilar (range [-1, 1]) . 
        """
        if not query or not query.strip():
            return None

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )

        similarities = self._compute_similarities(q_emb)

        # take top_k
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        best_idx = top_indices[0]
        best_score = float(similarities[best_idx])

        if best_score < self.threshold:
            return None

        mitre_id = self._tech_ids[best_idx]
        return mitre_id, best_score

    def predict_top_k(self, query: str) -> List[Tuple[str, float]]:
        """return top_k most match's  (mitre_id, cosine_similarity). """
        if not query or not query.strip():
            return []

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        similarities = self._compute_similarities(q_emb)
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.threshold:
                results.append((self._tech_ids[idx], score))

        return results

    def predict_top_k_detail(self, query: str) -> List[dict]:
        """return top_k most match's detailedinfo, containstext. 
        eachitem: {"tech_id": str, "score": float, "tech_text": str}
        """
        if not query or not query.strip():
            return []

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        similarities = self._compute_similarities(q_emb)
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.threshold:
                results.append({
                    "tech_id": self._tech_ids[idx],
                    "score": score,
                    "tech_text": self._tech_texts[idx],
                })
        return results

    def get_tech_text(self, tech_id: str) -> str:
        """according to ID returntext. """
        if tech_id in self._tech_descs:
            return self._tech_descs[tech_id]
        return ""

    def predict_codes(self, queries: List[str]) -> List[str]:
        """batchquery, return ID list. """
        return [
            (self.predict_top(q) or (None,))[0] or "UNKNOWN"
            for q in queries
        ]

    def predict_codes_batch(self, queries: List[str]) -> List[str]:
        """batchquery (vector) , return ID list. """
        if not queries:
            return []

        q_embs = self._model.encode(
            queries, show_progress_bar=False, normalize_embeddings=True
        )

        similarities = np.dot(q_embs, self._tech_embeddings.T)  # (n_queries, n_techs)

        results = []
        for i in range(len(queries)):
            if not queries[i] or not queries[i].strip():
                results.append("UNKNOWN")
                continue
            best_idx = int(np.argmax(similarities[i]))
            best_score = float(similarities[i, best_idx])
            if best_score < self.threshold:
                results.append("UNKNOWN")
            else:
                results.append(self._tech_ids[best_idx])

        return results


# ============================================================
# ============================================================

def _extract_process_name(properties: str) -> str:
    """fromnode properties intakeprocess. """
    props = properties.strip()
    m = re.search(r"'name'\s*:\s*'([^']+)'", props)
    if m:
        return m.group(1)
    m = re.search(r'"name"\s*:\s*"([^"]+)"', props)
    if m:
        return m.group(1)
    if props and not props.startswith("{") and len(props) < 50:
        return props
    return ""
