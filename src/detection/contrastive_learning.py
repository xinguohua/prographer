"""ATHENA adaptive contrastive encoder (Proof §IV.D).

This is the sole training path: standard GIN message passing, one final-layer
GRU update per node and snapshot, complete benign r-hop anchors, and HCL over
the train-only original/synthetic attack corpus.
"""
from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ._encoder_base import GraphEmbedderBase
from .gin_encoder import GINEncoder
from .temporal_encoder import TemporalNodeEncoder


def benign_anchor_indices(graph) -> List[int]:
    """Return exactly the benign node centers allowed by Proof §IV-D1."""
    labels = graph.vs["label"] if "label" in graph.vs.attributes() else [0] * graph.vcount()
    return [index for index, label in enumerate(labels) if int(label or 0) == 0]


def malicious_anchor_indices(graph) -> List[int]:
    labels = graph.vs["label"] if "label" in graph.vs.attributes() else [0] * graph.vcount()
    return [index for index, label in enumerate(labels) if int(label or 0) == 1]


def _stable_w2v_hash(value) -> int:
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "little")


class ATHENAEncoder(GraphEmbedderBase):
    """Paper-aligned ATHENA adaptive contrastive encoder."""

    _default_path = "athena_encoder.pth"

    def __init__(
        self,
        snapshots,
        features=None,
        mapp=None,
        use_temporal: bool = False,
        prop_feat_dim: int = 128,
        enc_hidden_dim: int = 64,
        enc_out_dim: int = 256,
        gin_layers: int = 3,
        dropout: float = 0.1,
        num_epochs: int = 3,
        batch_size: int = 64,
        lr: float = 1e-3,
        temperature: float = 0.07,
        r_hop: int = 4,
        train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
        test_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
        model_path: Optional[str] = None,
        anomaly_alpha: float = 1.0,
        w2v_window: int = 5,
        w2v_min_count: int = 1,
        w2v_sg: int = 1,
        w2v_epochs: int = 20,
        w2v_pretrained_path: Optional[str] = None,
        grad_clip_norm: float = 5.0,
        seed: int = 42,
    ):
        super().__init__(snapshots, features, mapp)
        self.snapshots = snapshots
        self.use_temporal = bool(use_temporal)
        self.prop_feat_dim = int(prop_feat_dim)
        self.enc_hidden_dim = int(enc_hidden_dim)
        self.enc_out_dim = int(enc_out_dim)
        self.gin_layers = int(gin_layers)
        self.dropout = float(dropout)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.r_hop = int(r_hop)
        self.model_path = model_path or self._default_path
        self.anomaly_alpha = float(anomaly_alpha)
        self.w2v_window = int(w2v_window)
        self.w2v_min_count = int(w2v_min_count)
        self.w2v_sg = int(w2v_sg)
        self.w2v_epochs = int(w2v_epochs)
        self.w2v_pretrained_path = w2v_pretrained_path
        self.grad_clip_norm = float(grad_clip_norm)
        self.seed = int(seed)
        self._w2v_model = None
        self._prop_cache: Dict[str, np.ndarray] = {}
        self.mutation_map: Dict[int, list] = {}

        self.train_snapshot_indices = self._resolve_snapshot_indices(train_indices, "train_indices")
        if test_indices is None:
            train_set = set(self.train_snapshot_indices)
            test_indices = [index for index in range(len(self.snapshots)) if index not in train_set]
        self.test_snapshot_indices = self._resolve_snapshot_indices(test_indices, "test_indices")
        overlap = set(self.train_snapshot_indices) & set(self.test_snapshot_indices)
        if overlap:
            raise ValueError(f"train_indices and test_indices overlap: {sorted(overlap)}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = self.prop_feat_dim if self.prop_feat_dim > 0 else 1
        self.encoder = GINEncoder(
            input_dim,
            self.enc_hidden_dim,
            self.enc_out_dim,
            num_layers=self.gin_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.temporal = TemporalNodeEncoder(self.enc_out_dim).to(self.device)
        parameters = list(self.encoder.parameters())
        if self.use_temporal:
            parameters += list(self.temporal.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=1e-4)
        self.snapshot_node_embeddings: List[Dict[str, np.ndarray]] = []
        self._attack_corpus: List[Tuple[int, int]] = []

    def train(self):
        if not self.train_snapshot_indices:
            raise RuntimeError("no valid training snapshots")
        self._ensure_w2v_model()
        self._attack_corpus = [
            (snapshot_id, center)
            for snapshot_id in self.train_snapshot_indices
            for center in malicious_anchor_indices(self.snapshots[snapshot_id])
        ]
        if not self._attack_corpus and not self.mutation_map:
            raise RuntimeError("HCL requires a non-empty train-only attack corpus")

        for epoch in range(self.num_epochs):
            value = self._train_one_epoch()
            print(
                f"[ATHENA] epoch={epoch + 1}/{self.num_epochs} "
                f"steps={1 if value is not None else 0} "
                f"mean_loss={0.0 if value is None else value:.6f}"
            )
        self.generate_node_embeddings(use_temporal=self.use_temporal)

    def _train_one_epoch(self) -> Optional[float]:
        """Train on same-snapshot positives with an own-time attack bank.

        A chronological no-graph pass first materializes own-time attack
        embeddings as gradient leaves. Same-snapshot benign losses accumulate
        ``dL/dz_attack`` while releasing each benign graph immediately. The
        attack branch is then replayed at the identical parameter version and
        with identical dropout seeds, and the accumulated embedding gradient
        is propagated into the encoder. Only then is one optimizer step taken.
        """
        attack_memory = self._build_attack_memory()
        if attack_memory is None:
            raise RuntimeError("no train-only attack embeddings available for HCL")
        attack_embeddings = attack_memory.detach().requires_grad_(True)
        eligible = [
            snapshot_id
            for snapshot_id in self._chronological_subset(self.train_snapshot_indices)
            if self.snapshots[snapshot_id] is not None
            and self.snapshots[snapshot_id].vcount() > 0
            and len(benign_anchor_indices(self.snapshots[snapshot_id])) >= 2
        ]
        if not eligible:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_temporal:
            self.temporal.reset()
        losses: List[float] = []
        for snapshot_id in self._chronological_subset(self.train_snapshot_indices):
            graph = self.snapshots[snapshot_id]
            if graph is None or graph.vcount() == 0:
                continue
            prior_table = self.temporal.snapshot() if self.use_temporal else None
            states = self._encode_snapshot(graph, snapshot_id, prior_table=prior_table)
            benign_centers = benign_anchor_indices(graph)
            if len(benign_centers) < 2:
                del states
                continue
            benign = states.index_select(
                0, torch.tensor(benign_centers, dtype=torch.long, device=self.device)
            )
            loss = self._weighted_contrastive_loss(benign, attack_embeddings)
            (loss / len(eligible)).backward()
            losses.append(float(loss.detach().cpu().item()))
            del loss, benign, states
        if attack_embeddings.grad is None:
            raise RuntimeError("HCL attack branch received no embedding gradient")
        self._backprop_attack_memory(attack_embeddings.grad.detach())
        parameters = list(self.encoder.parameters())
        if self.use_temporal:
            parameters += list(self.temporal.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.grad_clip_norm)
        self.optimizer.step()
        return float(np.mean(losses)) if losses else None

    def _build_attack_memory(self) -> Optional[torch.Tensor]:
        """Refresh detached negative representations at their own time points."""
        if self.use_temporal:
            self.temporal.reset()
        attack_parts: List[torch.Tensor] = []
        with torch.no_grad():
            for snapshot_id in self._chronological_subset(self.train_snapshot_indices):
                graph = self.snapshots[snapshot_id]
                if graph is None or graph.vcount() == 0:
                    continue
                prior_table = self.temporal.snapshot() if self.use_temporal else None
                with self._attack_rng(snapshot_id, -1):
                    states = self._encode_snapshot(graph, snapshot_id, prior_table=prior_table)
                centers = malicious_anchor_indices(graph)
                if centers:
                    attack_parts.append(states.index_select(
                        0, torch.tensor(centers, dtype=torch.long, device=self.device)
                    ).detach())
                for mutation_index, mutation_graph in enumerate(
                    self._mutations_for_snapshot(snapshot_id)
                ):
                    with self._attack_rng(snapshot_id, mutation_index):
                        encoded = self._encode_complete_graph(
                            mutation_graph,
                            snapshot_id=snapshot_id,
                            prior_table=prior_table,
                        )
                    if encoded is not None:
                        attack_parts.append(encoded.detach())
                del states
        if not attack_parts:
            return None
        return torch.cat(attack_parts, dim=0).detach()

    def _backprop_attack_memory(self, embedding_gradient: torch.Tensor) -> None:
        """Replay own-time attacks and propagate the bank's accumulated gradient."""
        if self.use_temporal:
            self.temporal.reset()
        cursor = 0
        for snapshot_id in self._chronological_subset(self.train_snapshot_indices):
            graph = self.snapshots[snapshot_id]
            if graph is None or graph.vcount() == 0:
                continue
            prior_table = self.temporal.snapshot() if self.use_temporal else None
            with self._attack_rng(snapshot_id, -1):
                states = self._encode_snapshot(graph, snapshot_id, prior_table=prior_table)
            centers = malicious_anchor_indices(graph)
            if centers:
                selected = states.index_select(
                    0, torch.tensor(centers, dtype=torch.long, device=self.device)
                )
                width = int(selected.size(0))
                torch.autograd.backward(
                    selected,
                    embedding_gradient[cursor:cursor + width],
                )
                cursor += width
            del states
            for mutation_index, mutation_graph in enumerate(
                self._mutations_for_snapshot(snapshot_id)
            ):
                with self._attack_rng(snapshot_id, mutation_index):
                    encoded = self._encode_complete_graph(
                        mutation_graph,
                        snapshot_id=snapshot_id,
                        prior_table=prior_table,
                    )
                if encoded is None:
                    continue
                width = int(encoded.size(0))
                torch.autograd.backward(
                    encoded,
                    embedding_gradient[cursor:cursor + width],
                )
                cursor += width
                del encoded
        if cursor != int(embedding_gradient.size(0)):
            raise RuntimeError(
                f"attack replay gradient mismatch: consumed {cursor}, "
                f"expected {embedding_gradient.size(0)}"
            )

    @contextmanager
    def _attack_rng(self, snapshot_id: int, mutation_index: int):
        """Make attack-bank materialization and replay bitwise reproducible."""
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        seed = (
            0xA7E1A
            + 1_000_003 * int(snapshot_id)
            + 97_003 * (int(mutation_index) + 2)
        ) % (2**31 - 1)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield

    def _mutations_for_snapshot(self, snapshot_id: int) -> List:
        graphs = self.mutation_map.get(int(snapshot_id), [])
        if not graphs:
            return []
        return list(graphs) if isinstance(graphs, list) else [graphs]

    def _encode_snapshot(
        self,
        graph,
        snapshot_id: int,
        *,
        prior_table: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply final-layer GIN then one GRU update and one state commit."""
        self._ensure_vertex_names(graph, snapshot_id)
        features = torch.from_numpy(self._build_node_features(graph)).to(self.device)
        edge_index, edge_features = self._igraph_edges_to_edge_index(graph)
        node_ids = self._temporal_node_ids(graph)
        states = self._encode_nodes(
            features,
            edge_index,
            edge_features,
            node_ids,
            prior_table=prior_table,
            commit_temporal=False,
        )
        if self.use_temporal:
            self.temporal.commit(node_ids, states)
        return states

    def _encode_ego_graph(self, graph, center: int, snapshot_id: int) -> Optional[torch.Tensor]:
        if graph is None or graph.vcount() == 0:
            return None
        vertices = sorted(set(graph.neighborhood(center, order=self.r_hop, mode="all")))
        local_center = vertices.index(int(center))
        return self._encode_complete_graph(
            graph.induced_subgraph(vertices), snapshot_id=snapshot_id, centers=[local_center],
        )

    def _encode_complete_graph(
        self,
        graph,
        snapshot_id: int = -1,
        centers: Optional[List[int]] = None,
        prior_table: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        if graph is None or graph.vcount() == 0:
            return None
        self._ensure_vertex_names(graph, snapshot_id)
        features = torch.from_numpy(self._build_node_features(graph)).to(self.device)
        edge_index, edge_features = self._igraph_edges_to_edge_index(graph)
        node_ids = self._temporal_node_ids(graph)
        states = self._encode_nodes(
            features,
            edge_index,
            edge_features,
            node_ids,
            prior_table=prior_table,
            commit_temporal=False,
        )
        if centers is None:
            centers = []
            if "_athena_anchor" in graph.vs.attributes():
                centers = [
                    index for index, flag in enumerate(graph.vs["_athena_anchor"])
                    if bool(flag) and int(graph.vs[index].attributes().get("label", 0) or 0) == 1
                ]
            if not centers:
                centers = malicious_anchor_indices(graph)
            if not centers and "_athena_replaced_region" in graph.vs.attributes():
                centers = [
                    index for index, flag in enumerate(graph.vs["_athena_replaced_region"])
                    if bool(flag)
                ]
        if not centers:
            return None
        indices = torch.tensor(sorted(set(centers)), dtype=torch.long, device=self.device)
        return states.index_select(0, indices)

    def _weighted_contrastive_loss(
        self,
        benign: torch.Tensor,
        attacks: torch.Tensor,
    ) -> torch.Tensor:
        """Proof Eqs. (5)-(6), with every other benign graph as a positive."""
        benign = F.normalize(benign, dim=-1)
        attacks = F.normalize(attacks, dim=-1)
        losses = []
        total = benign.size(0)
        for start in range(0, total, max(1, self.batch_size)):
            stop = min(total, start + max(1, self.batch_size))
            anchors = benign[start:stop]
            positive_logits = anchors @ benign.T / self.temperature
            self_mask = torch.zeros_like(positive_logits, dtype=torch.bool)
            rows = torch.arange(stop - start, device=self.device)
            self_mask[rows, torch.arange(start, stop, device=self.device)] = True
            positive_logits = positive_logits.masked_fill(self_mask, float("-inf"))
            attack_logits = anchors @ attacks.T / self.temperature
            weights = F.softmax(self.anomaly_alpha * attack_logits, dim=-1)
            weighted_attacks = attacks.size(0) * (weights * torch.exp(attack_logits)).sum(dim=-1)
            positive_exp = torch.exp(positive_logits).masked_fill(self_mask, 0.0)
            denominator = positive_exp.sum(dim=-1) + weighted_attacks
            log_prob = positive_logits - torch.log(denominator.clamp_min(1e-12)).unsqueeze(1)
            positive_mask = (~self_mask).float()
            per_anchor = -(positive_mask * log_prob.masked_fill(self_mask, 0.0)).sum(dim=-1)
            losses.append(per_anchor / positive_mask.sum(dim=-1).clamp_min(1.0))
        return torch.cat(losses).mean()

    def _encode_nodes(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_ids: List[str],
        *,
        prior_table: Optional[Dict[str, torch.Tensor]] = None,
        commit_temporal: bool,
    ) -> torch.Tensor:
        instantaneous = self.encoder(features, edge_index, edge_feat=edge_features)
        if not self.use_temporal:
            return instantaneous
        previous = self.temporal.fetch(node_ids, device=self.device, table=prior_table)
        updated = self.temporal(instantaneous, previous)
        if commit_temporal:
            self.temporal.commit(node_ids, updated)
        return updated

    def generate_node_embeddings(self, use_temporal: bool = False):
        """Generate train and held-out streams without cross-partition writes.

        Training embeddings come from a train-only pass.  Held-out embeddings
        come from a fresh chronological replay where each held-out snapshot can
        observe only preceding snapshots, never a future training snapshot.
        """
        self.encoder.eval()
        self.snapshot_node_embeddings = [{} for _ in self.snapshots]
        with torch.no_grad():
            self._generate_partition_embeddings(
                self._chronological_subset(self.train_snapshot_indices),
                store_ids=set(self.train_snapshot_indices),
                use_temporal=use_temporal,
            )
            if self.test_snapshot_indices:
                replay_ids = self._chronological_subset(
                    list(self.train_snapshot_indices) + list(self.test_snapshot_indices)
                )
                self._generate_partition_embeddings(
                    replay_ids,
                    store_ids=set(self.test_snapshot_indices),
                    use_temporal=use_temporal,
                )
        self.encoder.train()

    def _generate_partition_embeddings(
        self,
        chronological_ids: List[int],
        *,
        store_ids: set[int],
        use_temporal: bool,
    ) -> None:
        if use_temporal:
            self.temporal.reset()
        for snapshot_id in chronological_ids:
            graph = self.snapshots[snapshot_id]
            if graph is None or graph.vcount() == 0:
                continue
            self._ensure_vertex_names(graph, snapshot_id)
            features = torch.from_numpy(self._build_node_features(graph)).to(self.device)
            edge_index, edge_features = self._igraph_edges_to_edge_index(graph)
            output_node_ids = [str(graph.vs[index]["name"]) for index in range(graph.vcount())]
            node_ids = self._temporal_node_ids(graph)
            instantaneous = self.encoder(features, edge_index, edge_feat=edge_features)
            if use_temporal:
                previous = self.temporal.fetch(node_ids, device=self.device)
                states = self.temporal(instantaneous, previous)
                self.temporal.commit(node_ids, states)
            else:
                states = instantaneous
            if snapshot_id in store_ids:
                self.snapshot_node_embeddings[snapshot_id] = {
                    output_node_ids[index]: states[index].cpu().numpy().astype(np.float32)
                    for index in range(len(output_node_ids))
                }

    def _resolve_snapshot_indices(self, indices, name: str) -> List[int]:
        total = len(self.snapshots)
        if total == 0:
            return []
        if indices is None:
            raw = list(range(total))
        elif isinstance(indices, int):
            raw = [indices]
        elif isinstance(indices, tuple) and len(indices) == 2:
            start, end = sorted((int(indices[0]), int(indices[1])))
            raw = list(range(start, end + 1))
        else:
            raw = list(indices)
        valid = list(dict.fromkeys(int(index) for index in raw if 0 <= int(index) < total))
        if not valid:
            raise ValueError(f"{name} does not contain a valid snapshot index")
        return valid

    def _chronological_snapshot_ids(self) -> List[int]:
        return self._chronological_subset(range(len(self.snapshots)))

    def _chronological_subset(self, snapshot_ids: Iterable[int]) -> List[int]:
        def key(snapshot_id: int):
            graph = self.snapshots[snapshot_id]
            timestamps = []
            for sequence in (getattr(graph, "vs", []), getattr(graph, "es", [])):
                for item in sequence:
                    value = item.attributes().get("timestamp")
                    try:
                        timestamps.append(float(value))
                    except (TypeError, ValueError):
                        continue
            return (0, min(timestamps), snapshot_id) if timestamps else (1, snapshot_id)
        return sorted({int(value) for value in snapshot_ids}, key=key)

    def _ensure_vertex_names(self, graph, snapshot_id: int) -> None:
        if "name" not in graph.vs.attributes():
            graph.vs["name"] = [f"snapshot-{snapshot_id}-node-{index}" for index in range(graph.vcount())]
            return
        for index in range(graph.vcount()):
            if graph.vs[index]["name"] in (None, ""):
                graph.vs[index]["name"] = f"snapshot-{snapshot_id}-node-{index}"

    @staticmethod
    def _temporal_node_ids(graph) -> List[str]:
        attribute = (
            "_athena_temporal_id"
            if "_athena_temporal_id" in graph.vs.attributes()
            else "name"
        )
        return [str(graph.vs[index][attribute]) for index in range(graph.vcount())]

    def _tokenize_properties(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_\-./:\\]+", str(text or ""))

    def _collect_w2v_corpus(self) -> List[List[str]]:
        corpus = []
        seen = set()
        for snapshot_id in self.train_snapshot_indices:
            graph = self.snapshots[snapshot_id]
            if graph is None:
                continue
            for vertex in graph.vs:
                value = str(vertex.attributes().get("properties", ""))
                if value in seen:
                    continue
                seen.add(value)
                tokens = self._tokenize_properties(value)
                if tokens:
                    corpus.append(tokens)
        return corpus

    def _ensure_w2v_model(self):
        if self._w2v_model is not None:
            return
        try:
            from gensim.models import Word2Vec
        except Exception as exc:
            raise RuntimeError("ATHENA node features require gensim Word2Vec") from exc
        if self.w2v_pretrained_path and os.path.exists(self.w2v_pretrained_path):
            model = Word2Vec.load(self.w2v_pretrained_path)
            if int(model.wv.vector_size) == self.prop_feat_dim:
                self._w2v_model = model
                return
        corpus = self._collect_w2v_corpus()
        if not corpus:
            raise RuntimeError("Word2Vec training corpus is empty")
        self._w2v_model = Word2Vec(
            corpus,
            vector_size=self.prop_feat_dim,
            window=self.w2v_window,
            min_count=self.w2v_min_count,
            sg=self.w2v_sg,
            workers=1,
            epochs=self.w2v_epochs,
            seed=self.seed,
            hashfxn=_stable_w2v_hash,
        )

    def _build_node_features(self, graph) -> np.ndarray:
        if self.prop_feat_dim <= 0:
            return np.ones((graph.vcount(), 1), dtype=np.float32)
        self._ensure_w2v_model()
        output = np.zeros((graph.vcount(), self.prop_feat_dim), dtype=np.float32)
        for index, vertex in enumerate(graph.vs):
            value = str(vertex.attributes().get("properties", ""))
            if value not in self._prop_cache:
                vectors = [
                    self._w2v_model.wv[token]
                    for token in self._tokenize_properties(value)
                    if token in self._w2v_model.wv
                ]
                vector = np.mean(vectors, axis=0).astype(np.float32) if vectors else output[index]
                norm = float(np.linalg.norm(vector))
                self._prop_cache[value] = vector / norm if norm > 0 else vector
            output[index] = self._prop_cache[value]
        return output

    def _igraph_edges_to_edge_index(self, graph):
        edges = graph.get_edgelist()
        if not edges:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
            )
        return (
            torch.tensor(edges, dtype=torch.long, device=self.device).T.contiguous(),
            torch.zeros(len(edges), dtype=torch.long, device=self.device),
        )

    def save_model(self, path: Optional[str] = None):
        destination = path or self.model_path
        torch.save({
            "params": {
                "use_temporal": self.use_temporal,
                "prop_feat_dim": self.prop_feat_dim,
                "enc_hidden_dim": self.enc_hidden_dim,
                "enc_out_dim": self.enc_out_dim,
                "gin_layers": self.gin_layers,
                "dropout": self.dropout,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "temperature": self.temperature,
                "r_hop": self.r_hop,
                "train_indices": self.train_snapshot_indices,
                "test_indices": self.test_snapshot_indices,
                "model_path": self.model_path,
                "anomaly_alpha": self.anomaly_alpha,
                "w2v_window": self.w2v_window,
                "w2v_min_count": self.w2v_min_count,
                "w2v_sg": self.w2v_sg,
                "w2v_epochs": self.w2v_epochs,
                "w2v_pretrained_path": self.w2v_pretrained_path,
                "grad_clip_norm": self.grad_clip_norm,
            },
            "encoder": self.encoder.state_dict(),
            "temporal": self.temporal.state_dict(),
            "w2v_model": self._w2v_model,
            "snapshot_node_embeddings": self.snapshot_node_embeddings,
        }, destination)

    @classmethod
    def load(cls, snapshot_sequence, path: Optional[str] = None):
        source = path or cls._default_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(source, map_location=device, weights_only=False)
        instance = cls(snapshot_sequence, **state.get("params", {}))
        instance.encoder.load_state_dict(state["encoder"])
        instance.temporal.load_state_dict(state["temporal"])
        instance._w2v_model = state.get("w2v_model")
        instance.snapshot_node_embeddings = state.get("snapshot_node_embeddings", [])
        return instance

    def generate_external_embeddings(self) -> None:
        """Embed an evaluation-only stream with frozen GIN/GRU/Word2Vec."""
        if self._w2v_model is None:
            raise RuntimeError("evaluation checkpoint lacks the training Word2Vec model")
        self.encoder.eval()
        self.temporal.eval()
        self.snapshot_node_embeddings = [{} for _ in self.snapshots]
        with torch.no_grad():
            ids = self._chronological_subset(range(len(self.snapshots)))
            self._generate_partition_embeddings(
                ids, store_ids=set(ids), use_temporal=self.use_temporal,
            )

    def embed_nodes(self):
        return self.snapshot_node_embeddings[-1] if self.snapshot_node_embeddings else {}

    def embed_edges(self):
        return {}

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        if not self.snapshot_node_embeddings:
            raise RuntimeError("node embeddings have not been generated")
        indices = snapshot_sequence if snapshot_sequence is not None else range(len(self.snapshots))
        rows = []
        for index in indices:
            values = list(self.snapshot_node_embeddings[int(index)].values())
            rows.append(
                np.mean(np.stack(values), axis=0).astype(np.float32)
                if values else np.zeros(self.enc_out_dim, dtype=np.float32)
            )
        return np.stack(rows) if rows else np.zeros((0, self.enc_out_dim), dtype=np.float32)


__all__ = ["ATHENAEncoder", "benign_anchor_indices", "malicious_anchor_indices"]
