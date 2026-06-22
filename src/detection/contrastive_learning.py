"""Paper §IV.D - Adaptive contrastive learning training loop and snapshot
embedder (GCCEmbedderDev).

Composes the typed-GIN node encoder (:mod:`src.detection.gin_encoder`) with
the per-layer GRU temporal encoder (:mod:`src.detection.temporal_encoder`)
and the strategy-MoE fusion used to weight semantic mutation variants in the
hard-sample-weighted contrastive loss.
"""
from __future__ import annotations
from collections import deque
from typing import Optional, Iterable, Tuple, List, Dict, Union
import os
import re
import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(x, **kwargs):
        return x
from ._encoder_base import GraphEmbedderBase
from .gin_encoder import (
    EDGE_CATEGORY, NUM_EDGE_CATEGORIES, MLP, TypedGINConv, GINEncoder, classify_edge,
)
from .temporal_encoder import TemporalPerLayer


class StrategyMoE(nn.Module):
    """3 kind of mutationlearnable strategyableweightedfused (follows GAugLLM SimilarityAttentionMLP) . 

    input: 3  kind of mutation's  word2vec vector (content) + original properties 's  word2vec vector (context)
    output: weightedfusedafter's featurevector

    weightbytwopartdecide: 
    - content_weights: MLP learn, measuresmutationcontenttocontrastive learning's value
    - similarity: content · context dot product, measuresmutationtooriginalattacksemantic's preservedegreedegree
    """
    def __init__(self, emb_dim: int, n_strategies: int = 3, temperature: float = 0.2):
        super().__init__()
        hidden = emb_dim // 2
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(n_strategies * hidden, n_strategies)
        self.temperature = temperature
        self.n_strategies = n_strategies

    def forward(self, content_embs: torch.Tensor, context_emb: torch.Tensor):
        """
        Args:
            content_embs: (N, 3, D) - 3  kind of mutation's  word2vec vector
            context_emb:  (N, D)    - original properties 's  word2vec vector
        Returns:
            weights: (N, 3) - each kind of strategy's weight
            fused:   (N, D) - fusedafter's feature
        """
        # Step 1: MLP contentweight
        parts = []
        for i in range(self.n_strategies):
            h = self.relu(self.fc1(content_embs[:, i]))
            parts.append(h)
        content_weights = self.fc2(torch.cat(parts, dim=1))  # (N, 3)

        # Step 2: content-contextsimilarity
        sims = []
        for i in range(self.n_strategies):
            sim = (content_embs[:, i] * context_emb).sum(dim=1, keepdim=True)
            sims.append(sim)
        sims = torch.cat(sims, dim=1)  # (N, 3)

        # Step 3: add + softmax
        weights = F.softmax((content_weights + sims) / self.temperature, dim=1)

        # Step 4: weightedfused
        fused = (weights.unsqueeze(-1) * content_embs).sum(dim=1)  # (N, D)
        return weights, fused


class GCCEmbedderDev(GraphEmbedderBase):
    _default_path = 'gcc_encoder_dev.pth'

    def __init__(
            self,
            snapshots,
            features=None,
            mapp=None,
            # isusetimememory (TemporalPerLayer) 
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
            ego_max_nodes: int = 32,
            drop_edge_p: float = 0.2,
            feat_mask_p: float = 0.2,
            train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
            model_path: Optional[str] = None,
            anomaly_alpha: float = 1,  # weightedstrength, >0 denotes anomalythe morelarge weightthe morelarge
            use_sample_weights: bool = True,
            w2v_window: int = 5,
            w2v_min_count: int = 1,
            w2v_sg: int = 1,
            w2v_epochs: int = 20,
            w2v_pretrained_path: Optional[str] = None,
            use_malicious_snapshots: bool = True,
            use_malicious_negatives: bool = False,
            combine: bool = False,
            combine_ratio: float = 0.8,
            mal_neg_ratio: float = 0.3,
            mal_neg_node_token_len: int = 1,
            mal_stopwords=None,
            # [
            # 'event', 'read', 'write'
            # , 'execute'
            # ],
            # malicioustokenstopword list, input [] denotes nofilter
            mal_print_tokens: bool = True,  # isprint malicious tokenstatisticinfo
            # Top-K similar (optional, initially disabled) 
            topk_pos: Optional[int] = 0,  # initially disabled Top-K expand, back toclassic NT-Xent
            topk_pos_min_sim: float = 0.5,  # only whensimilarity > this thresholdtimewillinclude Top-K positive sample
            use_degree_coop_augment: bool = True,
                neg_weight_scale: float = 100.0,
                # snapshotaggregate“weight”number: attr_weight_alpha ∈ [0,1]
                #   - w_base: node base weight (frequency-first , itstime degree) 
                #   - w_attr: attributerarity weight (from g insideattribute frequency reciprocal vs ) 
                # most endnodeweight: w_eff = (1 - alpha) * norm(w_base) + alpha * norm(w_attr)
                attr_weight_alpha: float = 0.3,
                use_strategy_moe: bool = False,
    ):
        super().__init__(snapshots, features, mapp)
        if mal_stopwords is None:
            mal_stopwords = []
            #     [
            #     'event', 'read', 'write', 'execute'
            # ]
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
        self.ego_max_nodes = int(ego_max_nodes)
        self.drop_edge_p = float(drop_edge_p)
        self.feat_mask_p = float(feat_mask_p)
        self.model_path = model_path or self._default_path
        self.anomaly_alpha = float(anomaly_alpha)
        self.use_sample_weights = bool(use_sample_weights)

        # Word2Vec config (only1featurefromsource) 
        self.w2v_window = int(w2v_window)
        self.w2v_min_count = int(w2v_min_count)
        self.w2v_sg = int(w2v_sg)
        self.w2v_epochs = int(w2v_epochs)
        self.w2v_pretrained_path = w2v_pretrained_path
        self._w2v_model = None

        self.use_malicious_snapshots = bool(use_malicious_snapshots)
        # isuse“maliciouscorpus”fromgenerate additionalnegative sample; andcorruptstrengthandeach nodereplaced token number
        self.use_malicious_negatives = bool(use_malicious_negatives)
        self.combine = bool(combine)
        self.combine_ratio = float(combine_ratio)
        self.mal_neg_ratio = float(mal_neg_ratio)
        self.mal_neg_node_token_len = int(mal_neg_node_token_len)
        self.mal_use_type_group = False

        # malicioustokenstopuseword: connectuseinput's listconvertisset ([] denotes nofilter) 
        # normalizestopuseword: nested(list/tuple/set)convertisstringset, appear list nested list  set() wrong
        def _flatten_to_str_set(obj):
            out = []
            if obj is None:
                return set()
            if isinstance(obj, (list, tuple, set)):
                for it in obj:
                    if isinstance(it, (list, tuple, set)):
                        out.extend(str(x) for x in it)
                    else:
                        out.append(str(it))
                return set(out)
            return {str(obj)}

        self.mal_stopwords = _flatten_to_str_set(mal_stopwords)
        self.mal_print_tokens = bool(mal_print_tokens)  # isprint malicious tokenstatistic

        # Top-K sampleconfig
        self.topk_pos = int(topk_pos) if topk_pos is not None else None
        self.topk_pos_min_sim = float(topk_pos_min_sim)
        self.use_degree_coop_augment = bool(use_degree_coop_augment)
        self.neg_weight_scale = float(neg_weight_scale)
        # attributefrequencyrightparameter (1 alpha) 
        self.attr_weight_alpha = float(attr_weight_alpha)
        # isusepositivesubgraphfusedmalicioussubgraphnegative sample (use _build_neg_block_from_snapshots_with_pos) 
        self.use_pos_fusion_neg = True  # rowtimecanconnect True enable
        self.pos_fusion_ratio = 0.5
        self.pos_fusion_cross_ratio = 0.2
        self.pos_fusion_cross_max = 8

        self.debug_sim_dump = True
        self.debug_dump_dir = './gcc_debug'
        self.debug_rows_per_batch = 100
        self.debug_max_batches = 1
        self._debug_dumped_batches = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # featurecache (properties -> vector) 
        self._prop_cache = {}
        # malicious token : appearinmalicious nodeandits's  tokens
        self.malicious_node_tokens = []

        self.train_snapshot_indices = self._resolve_train_indices(train_indices)

        in_dim = self.prop_feat_dim if self.prop_feat_dim > 0 else 1
        self.encoder = GINEncoder(in_dim, self.enc_hidden_dim, self.enc_out_dim, num_layers=self.gin_layers,
                                  dropout=self.dropout).to(self.device)
        self.proj_head = nn.Sequential(
            nn.Linear(self.enc_out_dim, self.enc_out_dim),
            nn.ReLU(),
            nn.Linear(self.enc_out_dim, self.enc_out_dim),
        ).to(self.device)
        self.temporal = TemporalPerLayer(self.encoder.layer_dims).to(self.device)
        # StrategyMoE: 3 kind of mutationlearnable strategyablefused
        self.use_strategy_moe = bool(use_strategy_moe)
        self.strategy_moe = StrategyMoE(in_dim, n_strategies=3).to(self.device) if self.use_strategy_moe else None
        # optimizecontains encoder, projection head, inusetimecontains temporal / strategy_moe
        opt_params = list(self.encoder.parameters()) + list(self.proj_head.parameters())
        if self.use_temporal:
            opt_params += list(self.temporal.parameters())
        if self.strategy_moe is not None:
            opt_params += list(self.strategy_moe.parameters())
        self.optimizer = torch.optim.Adam(opt_params, lr=self.lr, weight_decay=1e-4)

        # trainaftercache: eachsnapshot1 {node_id: vec}
        self.snapshot_node_embeddings = []

        self.temporal.reset()

    def train(self):
        """dynamicgraphcontrastive learningprimarytrainloop (optimizeversion) """
        if not self.train_snapshot_indices:
            raise RuntimeError("nocanfortrain's snapshot. check train_indices set. ")

        self._ensure_w2v_model()
        if self.combine:
            self._precollect_malicious_tokens()
            self._precollect_malicious_snapshots()
        else:
            if self.use_malicious_negatives:
                self._precollect_malicious_tokens()
            if self.use_malicious_snapshots:
                self._precollect_malicious_snapshots()

        print(
            f"[GCC-Dev] Pretrain on {len(self.train_snapshot_indices)} snapshots | batch={self.batch_size} | tau={self.temperature}")

        # contrastive learningsample's  ego = train. last1 epoch save. 
        self.train_ego_cache = []
        self._save_egos = False

        for epoch in range(self.num_epochs):
            # last1 epoch enablesave
            if epoch == self.num_epochs - 1:
                self._save_egos = True
            if self.use_temporal:
                self.temporal.reset()  # each epoch settimestate (onlyenabletime) 
            epoch_loss = 0.0
            steps_done = 0

            # timeordertraverse snapshot, smallsnapshot1timetrain
            SMALL_THRESHOLD = 64
            sorted_indices = sorted(self.train_snapshot_indices)
            small_batch = []

            for sidx in sorted_indices:
                g = self.snapshots[sidx]
                if g is None or g.vcount() == 0:
                    continue

                if g.vcount() <= SMALL_THRESHOLD:
                    small_batch.append((sidx, g))
                    if len(small_batch) < 16 and sidx != sorted_indices[-1]:
                        continue
                    batch_loss = self._train_small_snapshots_packed(small_batch)
                    n_packed = len(small_batch)
                    epoch_loss += batch_loss * n_packed
                    steps_done += n_packed
                    if n_packed > 1:
                        print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Packed {n_packed} small snapshots (idx {small_batch[0][0]}~{small_batch[-1][0]}) | Loss={batch_loss:.6f}")
                    else:
                        print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Snapshot {small_batch[0][0]} | Loss={batch_loss:.6f}")
                    small_batch = []
                else:
                    if small_batch:
                        bl = self._train_small_snapshots_packed(small_batch)
                        n_packed = len(small_batch)
                        epoch_loss += bl * n_packed
                        steps_done += n_packed
                        print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Packed {n_packed} small snapshots (idx {small_batch[0][0]}~{small_batch[-1][0]}) | Loss={bl:.6f}")
                        small_batch = []

                    batch_loss = self._train_one_snapshot(g, sidx=sidx)
                    epoch_loss += batch_loss
                    steps_done += 1
                    torch.cuda.empty_cache()
                    print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Snapshot {sidx} | Loss={batch_loss:.6f}")

            if small_batch:
                bl = self._train_small_snapshots_packed(small_batch)
                n_packed = len(small_batch)
                epoch_loss += bl * n_packed
                steps_done += n_packed
                print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Packed {n_packed} small snapshots | Loss={bl:.6f}")

            avg = epoch_loss / max(1, steps_done)
            print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} DONE | AvgLoss={avg:.6f}")

        self.save_malicious_snapshot_stats()
        self.generate_node_embeddings(use_temporal=self.use_temporal)

        # train: contrastive learningsave's positive sample ego + attack ego
        # positive sampleinlast epoch savedto self.train_ego_cache
        # attack ego from _mal_ego_cache take
        mal_cache = getattr(self, '_mal_ego_cache', [])
        for entry in mal_cache:
            x_t, ei_t, nc = entry[0], entry[1], entry[2]
            ef_t = entry[3] if len(entry) > 3 else torch.zeros(0, dtype=torch.long)
            self.train_ego_cache.append((
                x_t.numpy() if isinstance(x_t, torch.Tensor) else x_t,
                ei_t.cpu() if isinstance(ei_t, torch.Tensor) else ei_t,
                ef_t.cpu() if isinstance(ef_t, torch.Tensor) else ef_t,
                1,  # attack label
                -1,
            ))

        # mutation ego addintotrain (ego levelmutation: each g_mut alreadyviais ego size) 
        mutation_map = getattr(self, 'mutation_map', None)
        n_mut = 0
        if mutation_map:
            for b_idx, ego_list in mutation_map.items():
                if not isinstance(ego_list, list):
                    ego_list = [ego_list]
                for g_mut in ego_list:
                    if g_mut is None or g_mut.vcount() == 0:
                        continue
                    self._preheat_snapshot_properties(g_mut)
                    # ego levelmutation: integer g_mut ismutationafter's  ego, connectencode
                    x_np = self._build_node_features(g_mut)

                    # MoE modulo: mutation word2vec vector, existsgraphattributeup
                    # notinhere MoE (gradientbreak) , traintimein _encode_single_ego_graph intime
                    if self.strategy_moe is not None:
                        variants_dict = None
                        try:
                            variants_dict = g_mut["strategy_variants"]
                        except (KeyError, TypeError):
                            pass
                        if variants_dict:
                            variant_vecs = {}
                            for idx, v in variants_dict.items():
                                if idx >= g_mut.vcount():
                                    continue
                                def _get_vec(s):
                                    return self._w2v_vector_from_tokens(self._tokenize_properties(s))
                                content_np = np.stack([
                                    _get_vec(v.get("replacement", "")),
                                    _get_vec(v.get("rewriting", "")),
                                    _get_vec(v.get("extension", "")),
                                ])  # (3, D)
                                context_np = _get_vec(v.get("original", ""))  # (D,)
                                variant_vecs[idx] = (content_np, context_np)
                            g_mut["variant_vecs"] = variant_vecs

                    ei, ef = self._igraph_edges_to_edge_index(g_mut)
                    self.train_ego_cache.append((x_np, ei.cpu(), ef.cpu(), 1, -1))
                    n_mut += 1

        train_benign = sum(1 for _, _, _, lab, _ in self.train_ego_cache if lab == 0)
        train_attack = sum(1 for _, _, _, lab, _ in self.train_ego_cache if lab == 1)
        print(f"[GCC-Dev] train: {len(self.train_ego_cache)} ego "
              f"(benign={train_benign}, attack={train_attack}, contains{n_mut}mutation)")

        self._generate_test_ego_cache()
        self.save_model()

    def _generate_test_ego_cache(self):
        """buildtest: malicioussnapshot's allattack node + samplebenign node ego"""
        import random as _rng
        self.test_ego_cache = []
        print("[GCC-Dev] buildtest ego...")
        t0 = __import__('time').time()

        SAMPLE_PER_SNAPSHOT = 50
        benign_end = self.train_snapshot_indices[-1] if self.train_snapshot_indices else 0
        mal_start_idx = benign_end + 1
        mal_end_idx = len(self.snapshots) - 1

        for sidx in range(mal_start_idx, mal_end_idx + 1):
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                continue
            self._preheat_snapshot_properties(g)
            attack_nodes = [v for v in range(g.vcount()) if g.vs[v].attributes().get('label', 0) == 1]
            benign_nodes = [v for v in range(g.vcount()) if g.vs[v].attributes().get('label', 0) == 0]
            if len(benign_nodes) > SAMPLE_PER_SNAPSHOT:
                benign_nodes = _rng.sample(benign_nodes, SAMPLE_PER_SNAPSHOT)

            from src.augmentation.semantic_mutation import _get_properties
            test_ego_max = min(5, self.ego_max_nodes)
            for v in attack_nodes + benign_nodes:
                sub = self._ego_subgraph(g, v, r=self.r_hop, max_nodes=test_ego_max)
                if sub.vcount() == 0:
                    continue
                x_np = self._build_node_features(sub)
                ei, ef = self._igraph_edges_to_edge_index(sub)
                # ego levelnote: subgraphinsidecontains label=1 node → integer ego isattack
                ego_lab = 0
                for vi in range(sub.vcount()):
                    if sub.vs[vi].attributes().get('label', 0) == 1:
                        ego_lab = 1
                        break
                vtype = str(g.vs[v].attributes().get('type', ''))
                prop = _get_properties(g, v)
                self.test_ego_cache.append((x_np, ei.cpu(), ef.cpu(), ego_lab, sidx, vtype, prop, v))

        test_benign = sum(1 for e in self.test_ego_cache if e[3] == 0)
        test_attack = sum(1 for e in self.test_ego_cache if e[3] == 1)
        print(f"[GCC-Dev] test: {len(self.test_ego_cache)} ego "
              f"(benign={test_benign}, attack={test_attack}), "
              f"elapsed {__import__('time').time()-t0:.1f}s")

    def _train_small_snapshots_packed(self, snapshot_batch: list) -> float:
        """smallsnapshotbecome1 batch train, subtract per-snapshot . 

        Args:
            snapshot_batch: [(sidx, graph), ...]

        Returns:
            mean loss
        """
        device = self.device
        all_x, all_e, all_node_counts = [], [], []
        total_nodes_offset = 0

        all_ef = []
        for sidx, g in snapshot_batch:
            if g is None or g.vcount() == 0:
                continue
            self._preheat_snapshot_properties(g)
            x_np = self._build_node_features(g)
            x_t = torch.from_numpy(x_np).to(device)
            e_t, ef_t = self._igraph_edges_to_edge_index(g)
            if e_t.numel() > 0:
                e_t = e_t + total_nodes_offset
            all_x.append(x_t)
            all_e.append(e_t)
            all_ef.append(ef_t)
            all_node_counts.append(g.vcount())
            total_nodes_offset += g.vcount()

        if not all_x:
            return 0.0

        Bc = len(all_x)
        X_pos = torch.cat(all_x, dim=0)
        E_pos = torch.cat(all_e, dim=1) if any(e.numel() > 0 for e in all_e) else torch.zeros((2, 0), dtype=torch.long, device=device)
        EF_pos = torch.cat(all_ef, dim=0) if all_ef else None
        graph_ids = torch.tensor(
            [gi for gi, n in enumerate(all_node_counts) for _ in range(n)], device=device
        )

        # Forward
        Z_layers = self.encoder(X_pos, E_pos, edge_feat=EF_pos, return_all=True)
        H_last = Z_layers[-1]
        sums = torch.zeros((Bc, H_last.size(1)), device=device)
        cnts = torch.zeros(Bc, device=device)
        sums.index_add_(0, graph_ids, H_last)
        cnts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
        means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
        Z_pos = F.normalize(self.proj_head(means), dim=-1)

        # negative sample N(b) = attacksample + mutationgraph ego subgraph
        Z_neg_parts = []

        has_ego = bool(self.use_malicious_snapshots and hasattr(self, '_mal_ego_pool') and len(
            self._mal_ego_pool) > 0)
        if has_ego:
            if getattr(self, 'mimicry_mode', False):
                Z_attack = self._build_neg_augmented(
                    Bc, device, [x for x in all_x], [e for e in all_e], all_node_counts, mode='mimicry')
            else:
                Z_attack = self._build_neg_augmented(Bc, device, mode='standard')
            if Z_attack is not None:
                Z_neg_parts.append(Z_attack)

        # 's smallsnapshot: collectmutation ego (eachsnapshotat mostsample K , negative sampleencodeslow) 
        MAX_MUT_PER_SNAPSHOT = 5
        mutation_map = getattr(self, 'mutation_map', None)
        if mutation_map:
            for sidx, _ in snapshot_batch:
                if sidx in mutation_map:
                    ego_list = mutation_map[sidx]
                    if not isinstance(ego_list, list):
                        ego_list = [ego_list]
                    if len(ego_list) > MAX_MUT_PER_SNAPSHOT:
                        ego_list = random.sample(ego_list, MAX_MUT_PER_SNAPSHOT)
                    for g_ego in ego_list:
                        try:
                            Z_mut = self._encode_single_ego_graph(g_ego, device)
                            if Z_mut is not None:
                                Z_neg_parts.append(Z_mut)
                        except Exception:
                            pass

        Z_neg = torch.cat(Z_neg_parts, dim=0) if Z_neg_parts else None

        # Loss + backward
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._weighted_contrastive_loss(Z_pos, Z_neg, temperature=self.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.proj_head.parameters()), max_norm=5.0
        )
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def _encode_ego_subgraphs_from_graph(self, g, device) -> Optional[torch.Tensor]:
        """fromgraphintake label=1 node's  ego subgraph, encodeisembeddingvector. 

        return [N_attack, D], eachattack node1 ego subgraphembedding. 
        degree _mal_ego_pool consistent. 
        """
        #  label=1 node
        attack_centers = []
        labels = g.vs['label'] if 'label' in g.vs.attributes() else None
        if labels is None:
            return None
        for i, lab in enumerate(labels):
            if int(lab) == 1:
                attack_centers.append(i)
        if not attack_centers:
            return None

        self._preheat_snapshot_properties(g)
        ego_embeddings = []

        for c in attack_centers:
            sub = self._ego_subgraph(g, c, r=self.r_hop, max_nodes=self.ego_max_nodes)
            if sub.vcount() == 0:
                continue
            x_np = self._build_node_features(sub)
            x_t = torch.from_numpy(x_np).to(device)
            e_t, ef_t = self._igraph_edges_to_edge_index(sub)

            Z_layers = self.encoder(x_t, e_t, edge_feat=ef_t, return_all=True)
            H = Z_layers[-1]
            graph_emb = H.mean(dim=0, keepdim=True)
            ego_embeddings.append(graph_emb)

        if not ego_embeddings:
            return None

        Z = torch.cat(ego_embeddings, dim=0)  # [N_attack, hidden]
        return F.normalize(self.proj_head(Z), dim=-1)  # [N_attack, D]

    def _encode_single_ego_graph(self, g_ego, device) -> Optional[torch.Tensor]:
        """encode ego levelmutationgraphis [1, D] embedding. 

        g_ego alreadyviais ego size (~32node) , connectencode, notagain r-hop open. 
        MoE modulo: tohas variant_vecs 's attack node, time StrategyMoE fusedfeature (hasgradient) . 
        """
        if g_ego is None or g_ego.vcount() == 0:
            return None
        self._preheat_snapshot_properties(g_ego)
        x_np = self._build_node_features(g_ego)
        x_t = torch.from_numpy(x_np).to(device)

        # MoE: timefusedmutationvector (gradientcanback StrategyMoE) 
        if self.strategy_moe is not None:
            variant_vecs = None
            try:
                variant_vecs = g_ego["variant_vecs"]
            except (KeyError, TypeError):
                pass
            if variant_vecs:
                indices = list(variant_vecs.keys())
                contents = torch.tensor(
                    np.stack([variant_vecs[i][0] for i in indices]),
                    dtype=torch.float32, device=device)  # (K, 3, D)
                contexts = torch.tensor(
                    np.stack([variant_vecs[i][1] for i in indices]),
                    dtype=torch.float32, device=device)  # (K, D)
                _, fused = self.strategy_moe(contents, contexts)  # (K, D) hasgradient
                for j, idx in enumerate(indices):
                    x_t[idx] = fused[j]

        e_t, ef_t = self._igraph_edges_to_edge_index(g_ego)
        Z_layers = self.encoder(x_t, e_t.to(device), edge_feat=ef_t.to(device), return_all=True)
        H = Z_layers[-1]
        graph_emb = H.mean(dim=0, keepdim=True)  # [1, hidden]
        return F.normalize(self.proj_head(graph_emb), dim=-1)  # [1, D]

    def _train_one_snapshot(self, g, sidx: Optional[int] = None) -> float:
        """ snapshot train (optimize + version NT-Xent) """
        device = self.device
        num_nodes = g.vcount()
        if num_nodes == 0:
            return 0.0

        # snapshot-levelattribute: willsnapshothasnode properties vectorwritecache, traininsidetimehit
        self._preheat_snapshot_properties(g)

        # ------- Step 1: globalrightsample (notright) -------
        MAX_CENTERS_PER_SNAPSHOT = 512  # 512 contrastive learning, subtract ego subgraphtake
        sample_size = min(num_nodes, MAX_CENTERS_PER_SNAPSHOT)

        centers = list(range(num_nodes))
        if 'frequency' in g.vs.attributes():
            freqs = np.array([float(f) for f in g.vs['frequency']])
            freqs = freqs + 1e-6  # stopis0
            probs = freqs / freqs.sum()
        else:
            probs = np.ones(num_nodes) / num_nodes

        sampled_centers = np.random.choice(centers, size=sample_size, replace=(sample_size > num_nodes), p=probs)
        centers = sampled_centers.tolist()

        # ------- Step 2: init -------
        total_loss, total_steps = 0.0, 0
        bsz = max(1, int(self.batch_size))
        total_batches = math.ceil(len(centers) / bsz)
        print(f"  [Snapshot {sidx}] nodes={num_nodes}, sampled={sample_size}, batches={total_batches}")

        # initcache (savebefore batch 's  subs, x_list, e_list, freq_weights) 
        if not hasattr(self, "_ego_cache"):
            self._ego_cache = deque(maxlen=8)

        # ------- Step 3:  batch train -------

        n_centers = len(centers)
        for start in _tqdm(range(0, n_centers, bsz), total=total_batches, leave=False, desc=f"Snapshot {sidx} Batches"):
            end = min(n_centers, start + bsz)
            batch_centers = centers[start:end]
            if not batch_centers:
                continue

            subs, node_counts, freq_weights = [], [], []
            x_list, e_list, ef_list, ids_list = [], [], [], []

            #  batch ego graph (ego sizeandnegative sample/testconsistent) 
            train_ego_max = min(5, self.ego_max_nodes)
            for c in batch_centers:
                sub = self._ego_subgraph(g, c, r=self.r_hop, max_nodes=train_ego_max)
                if sub.vcount() == 0:
                    continue
                xi_t = torch.from_numpy(self._build_node_features(sub)).to(device)
                ei_t, efi_t = self._igraph_edges_to_edge_index(sub)
                subs.append(sub)
                node_counts.append(sub.vcount())
                ids_list.append([sub.vs[i]['name'] for i in range(sub.vcount())])
                x_list.append(xi_t)
                e_list.append(ei_t)
                ef_list.append(efi_t)
                freq = float(g.vs[c]['frequency']) if 'frequency' in g.vs.attributes() else 1.0
                freq_weights.append(1.0 + max(0.0, self.anomaly_alpha) * freq)

            if not subs:
                continue

            Bc = len(subs)
            offsets = np.cumsum([0] + node_counts[:-1]).tolist()
            graph_ids = torch.tensor(
                [gi for gi, n in enumerate(node_counts) for _ in range(n)], device=device
            )

            # ======== positive sample: benign ego subgraph connectencode ========
            X_pos = torch.cat(x_list, dim=0)
            E_pos_cols = [ei + off for ei, off in zip(e_list, offsets)]
            E_pos = torch.cat(E_pos_cols, dim=1)
            EF_pos = torch.cat(ef_list, dim=0) if ef_list else None

            def encode_batch(X, E, n_graphs, gids, ef=None):
                Z_layers = self.encoder(X, E, edge_feat=ef, return_all=True)
                H_last = Z_layers[-1]
                sums = torch.zeros((n_graphs, H_last.size(1)), device=device)
                cnts = torch.zeros(n_graphs, device=device)
                sums.index_add_(0, gids, H_last)
                cnts.index_add_(0, gids, torch.ones_like(gids, dtype=torch.float32))
                means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
                return F.normalize(self.proj_head(means), dim=-1)

            Z_pos = encode_batch(X_pos, E_pos, Bc, graph_ids, ef=EF_pos)  # [Bc, D]

            # ======== negative sample N(b) = attacksample(share) + G̃_b egosubgraph() ========
            Z_neg_parts = []

            # (1) shareattack: sample Bc  ego subgraph
            has_ego = bool(self.use_malicious_snapshots and hasattr(self, '_mal_ego_pool') and len(
                self._mal_ego_pool) > 0)
            if has_ego:
                if getattr(self, 'mimicry_mode', False):
                    Z_attack = self._build_neg_augmented(
                        Bc, device, x_list, e_list, node_counts, mode='mimicry')
                else:
                    Z_attack = self._build_neg_augmented(Bc, device, mode='standard')
                if Z_attack is not None:
                    Z_neg_parts.append(Z_attack)

            # (2) mutation ego (eachsnapshotat mostsample K ) 
            MAX_MUT_PER_SNAPSHOT = 5
            mutation_map = getattr(self, 'mutation_map', None)
            if mutation_map and sidx is not None and sidx in mutation_map:
                ego_list = mutation_map[sidx]
                if not isinstance(ego_list, list):
                    ego_list = [ego_list]
                if len(ego_list) > MAX_MUT_PER_SNAPSHOT:
                    ego_list = random.sample(ego_list, MAX_MUT_PER_SNAPSHOT)
                for g_ego in ego_list:
                    try:
                        Z_mut = self._encode_single_ego_graph(g_ego, device)
                        if Z_mut is not None:
                            Z_neg_parts.append(Z_mut)
                    except Exception:
                        pass

            Z_neg = torch.cat(Z_neg_parts, dim=0) if Z_neg_parts else None

            # ======== loss: hasweightedcontrastive loss (paperformula 5, 6)  ========
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._weighted_contrastive_loss(
                Z_pos, Z_neg,
                temperature=self.temperature,
            )

            loss.backward()
            clip_params = list(self.encoder.parameters()) + list(self.proj_head.parameters())
            if self.strategy_moe is not None:
                clip_params += list(self.strategy_moe.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=5.0)
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1

            # last1 epoch savepositive sample ego originaldata (buildtrain) 
            if getattr(self, '_save_egos', False):
                for xi, ei_raw, efi, sub_g in zip(x_list, e_list, ef_list, subs):
                    lab = 0
                    self.train_ego_cache.append((
                        xi.detach().cpu().numpy(),
                        ei_raw.cpu(),
                        efi.cpu(),
                        lab,
                        sidx,
                    ))

            # cacheto CPU (onlyinneeds token negative sampletime, without's /memoryuse) 
            if self.use_malicious_negatives:
                cpu_x = [xi.detach().cpu() for xi in x_list]
                cpu_e = [ei.cpu() for ei in e_list]
                self._ego_cache.append((subs, cpu_x, cpu_e, freq_weights))

            # put GPU 
            if total_steps % 50 == 0:
                torch.cuda.empty_cache()

        return total_loss / max(1, total_steps)

    def _preheat_snapshot_properties(self, g) -> None:
        """willsnapshotinhasnode's  properties first vectorwrite `_prop_cache`. 
        notmodifyexternalsemantic, onlysubtracttrain's time. """
        n = g.vcount()
        if n == 0:
            return
        if self.prop_feat_dim <= 0:
            return
        if self._w2v_model is None:
            self._ensure_w2v_model()

        # batchread properties
        if 'properties' in g.vs.attributes():
            try:
                props: List[str] = [str(p) for p in g.vs['properties']]
            except Exception:
                props = [str(g.vs[i].attributes().get('properties', '')) for i in range(n)]
        else:
            props = [str(g.vs[i].attributes().get('properties', '')) for i in range(n)]

        uncached = {k for k in props if k not in self._prop_cache}
        if not uncached:
            return
        for key in uncached:
            tokens = self._tokenize_properties(key)
            vec = self._w2v_vector_from_tokens(tokens)
            self._prop_cache[key] = vec

    # ---------- malicious tokens support (fornegative sample)  ----------
    def _precollect_malicious_tokens(self, save_path: str = "malicious_tokens_log.txt"):
        """collectmalicious node tokens (nodelevel + cansavetofile) , simultaneouslyrecordeach node's fromsourcesnapshotindex. """
        if getattr(self, "malicious_node_tokens", None) is None:
            self.malicious_node_tokens = []
        if getattr(self, "malicious_node_origin", None) is None:
            self.malicious_node_origin = []

        total_nodes, malicious_nodes, total_snapshots = 0, 0, 0
        stop = getattr(self, "mal_stopwords", set())

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("[maliciousTokencollectlog]\n")
            f.write("=" * 60 + "\n")

            for snap_idx, g in enumerate(self.snapshots):
                if g is None or g.vcount() == 0:
                    continue
                total_snapshots += 1

                snapshot_node_map = {}

                for i in range(g.vcount()):
                    total_nodes += 1
                    try:
                        lab = int(g.vs[i].attributes().get("label", 0))
                    except Exception:
                        lab = 0
                    if lab != 1:
                        continue

                    malicious_nodes += 1
                    toks = self._get_node_tokens(g, i)
                    if stop:
                        toks = [t for t in toks if t not in stop]
                    if not toks:
                        continue

                    #  tokens, recordnodefrom's  snapshot index
                    self.malicious_node_tokens.append(toks)
                    self.malicious_node_origin.append(snap_idx)

                    snapshot_node_map[i] = toks

                if snapshot_node_map:
                    header = f"\n[Snapshot {snap_idx:02d}] malicious nodetokenmapping:"
                    print(header)
                    f.write(header + "\n")
                    for nid, toks in snapshot_node_map.items():
                        line = f"  {nid}: {toks}"
                        print(line)
                        f.write(line + "\n")

            from collections import Counter
            counter = Counter(t for toks in self.malicious_node_tokens for t in toks)
            total_mal_tokens = sum(counter.values())

            summary = "\n" + "=" * 60 + "\n"
            summary += "[maliciousTokenstatistic-nodelevel]\n"
            summary += f"  snapshotnumber: {total_snapshots}\n"
            summary += f"  totalnodenumber: {total_nodes}\n"
            summary += f"  malicious nodenumber: {malicious_nodes}\n"
            summary += f"  malicious nodetokensetnumber: {len(self.malicious_node_tokens)}\n"
            summary += f"  collectto's tokentotal: {total_mal_tokens}\n"
            summary += f"  Top-10: {counter.most_common(10)}\n"
            if stop:
                summary += f"  stopusewordcount: {len(stop)}\n"
            summary += "=" * 60 + "\n"

            print(summary)
            f.write(summary)

        print(f"[✅ logsavedto]: {save_path}")

    def _sample_malicious_tokens(self, num_nodes: int) -> List[str]:
        """
        fromnode's maliciouscorpusinextract token list (eachin's node's all token willcollect) . 
        - num_nodes: extractmalicious node (without) . if num_nodes > canusenodenumber, will () . 
        return: flatten after's  token list (length = sum(len(node_tokens) for chosen nodes); ifcorpusnot, returnalreadyhas's ) . 
        --- note: notagain token number, eachnode's has token include (integernodereplace) . 
        """
        num_nodes = int(max(0, num_nodes))
        if num_nodes == 0:
            return []

        if not hasattr(self, "malicious_node_tokens") or not self.malicious_node_tokens:
            return []

        if not hasattr(self, "malicious_snapshot_stats"):
            self.malicious_snapshot_stats = {}  # {snapshot_id: count}

        node_lists = self.malicious_node_tokens  # List[List[str]]
        node_origins = getattr(self, "malicious_node_origin", None)  # List[int]
        total_nodes = len(node_lists)

        if total_nodes >= num_nodes:
            chosen_idx = random.sample(range(total_nodes), k=num_nodes)
        else:
            chosen_idx = list(range(total_nodes))
            need = num_nodes - total_nodes
            if total_nodes > 0 and need > 0:
                chosen_idx.extend(random.choices(range(total_nodes), k=need))

        out_tokens: List[str] = []
        for idx in chosen_idx:
            toks = node_lists[idx]
            if not toks:
                continue
            # node's has token alladdinto (notbreak) 
            out_tokens.extend(toks)
            if node_origins is not None:
                sid = node_origins[idx]
                self.malicious_snapshot_stats[sid] = self.malicious_snapshot_stats.get(sid, 0) + 1

        # : ifhasnodeno token () , fromhasnodeif (non-emptytimereturn) 
        if not out_tokens:
            flat = [t for toks in node_lists for t in toks]
            if not flat:
                return []
            out_tokens.append(random.choice(flat))

        return out_tokens

    def _precollect_malicious_snapshots(self):
        """
        collectmalicious node's  ego graphsample: 
        - eachtimerowwill ()  malicious_tokens_log.txt; 
        - by (snapshot_idx, local_node_idx) save; 
        - likerecordnodeattribute, notbreak; 
        - outputandlogall integer. 
        """
        self._mal_ego_pool: List[Tuple[int, int]] = []
        train_ids = list(range(len(self.snapshots)))
        per_snapshot_mal: Dict[int, int] = {}
        log_path = "malicious_tokens_log.txt"

        try:
            f_log = open(log_path, "w", encoding="utf-8")  # ⚠️ use "w" modulo
            f_log.write("[maliciousEGOsubgraphcollect - integerrecordmodulo]\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"[warning] withoutlogfile: {e}")
            f_log = None

        # cachemaliciousegosubgraph's featureandedge, afterfused
        if not hasattr(self, '_mal_ego_cache'):
            self._mal_ego_cache: List[Tuple[torch.Tensor, torch.Tensor, int]] = []  # (X, edge_index, node_count)

        for sidx in train_ids:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                continue

            try:
                labels = g.vs['label'] if 'label' in g.vs.attributes() else None
            except Exception:
                labels = None

            if labels is not None:
                iterator = enumerate(labels)
            else:
                iterator = ((i, g.vs[i].attributes().get('label', 0)) for i in range(g.vcount()))

            for i, lab in iterator:
                try:
                    if int(lab) == 1:
                        self._mal_ego_pool.append((sidx, i))
                        per_snapshot_mal[sidx] = per_snapshot_mal.get(sidx, 0) + 1

                        sub = self._ego_subgraph(g, center=i, r=self.r_hop, max_nodes=self.ego_max_nodes)
                        nv, ne = sub.vcount(), sub.ecount()
                        line = f"[Snapshot {sidx:02d}] center={i} -> ego(nodes={nv}, edges={ne})"
                        print(line)
                        if f_log:
                            f_log.write(line + "\n")

                        try:
                            x_m_np = self._build_node_features(sub)
                            e_m, ef_m = self._igraph_edges_to_edge_index(sub)
                            self._mal_ego_cache.append((torch.from_numpy(x_m_np), e_m, sub.vcount(), ef_m))
                        except Exception as ce:
                            if f_log:
                                f_log.write(f"[cachefailed] Snapshot {sidx:02d} center={i}: {ce}\n")

                        for vi in range(sub.vcount()):
                            v = sub.vs[vi]
                            attrs = v.attributes()
                            name = str(attrs.get('name', ''))
                            lab = str(attrs.get('label', ''))
                            freq = str(attrs.get('frequency', ''))
                            prop = str(attrs.get('properties', ''))
                            node_line = f"    node[{vi}]: name={name} label={lab} freq={freq} props={prop}"
                            print(node_line)
                            if f_log:
                                f_log.write(node_line + "\n")

                except Exception as ex:
                    warn = f"[EGOsavefailed] Snapshot {sidx:02d}, center={i}: {ex}"
                    print(warn)
                    if f_log:
                        f_log.write(warn + "\n")

        print(f"[maliciousEGO] alreadycollect: {len(self._mal_ego_pool)} maliciouscenter, logalreadywrite {log_path}")
        if f_log:
            try:
                f_log.write("\n[totaleachsnapshotmaliciouscenternumber]\n")
                for sid in sorted(per_snapshot_mal.keys()):
                    m = per_snapshot_mal.get(sid, 0)
                    f_log.write(f"  Snapshot {sid:02d}: maliciouscenter={m}\n")
                f_log.write("=" * 80 + "\n")
            finally:
                f_log.disable()

    def _build_neg_augmented(self, Bc: int, device: torch.device,
                             benign_x_list=None, benign_e_list=None,
                             benign_node_counts=None, mode='standard'):
        """
        hascontrastive learning:  Bc increasestrongafter's negative sampleembedding. 
        increasestrongonlyuseinnegative sample (attack ego subgraph) . 

        mode:
          'standard' - usewhenbeforestrategy's increasestrong (drop_edge_p / feat_mask_p / degree_aware) 
          'mimicry'  - towardmalicioussubgraphnoteintobenignedgeandfeature

        return: Z_neg [Bc, D] or None
        """
        pool = self._mal_ego_pool if hasattr(self, '_mal_ego_pool') else None
        if not pool:
            return None

        chosen = [random.choice(pool) for _ in range(Bc)]

        _ef_zero = torch.zeros(0, dtype=torch.long, device=device)
        neg_x_list, neg_e_list, neg_ef_list, neg_node_counts = [], [], [], []
        for (sidx, center) in chosen:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                neg_x_list.append(torch.zeros((1, self.prop_feat_dim), dtype=torch.float32, device=device))
                neg_e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                neg_ef_list.append(_ef_zero)
                neg_node_counts.append(1)
                continue
            # negative sample ego usemoresmall's  max_nodes, subtract MemoryObject voiceattacksignal
            neg_ego_max = min(5, self.ego_max_nodes)
            try:
                sub = self._ego_subgraph(g, center=center, r=self.r_hop, max_nodes=neg_ego_max)
            except Exception:
                sub = None
            if sub is None or sub.vcount() == 0:
                neg_x_list.append(torch.zeros((1, self.prop_feat_dim), dtype=torch.float32, device=device))
                neg_e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                neg_ef_list.append(_ef_zero)
                neg_node_counts.append(1)
                continue
            x_np = self._build_node_features(sub)
            eidx, ef = self._igraph_edges_to_edge_index(sub)
            neg_x_list.append(torch.from_numpy(x_np).to(device))
            neg_e_list.append(eidx.to(device))
            neg_ef_list.append(ef.to(device))
            neg_node_counts.append(sub.vcount())

        if sum(neg_node_counts) == 0:
            return None

        Bn = len(neg_node_counts)
        offsets_neg = np.cumsum([0] + neg_node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(neg_node_counts) for _ in range(n)], device=device
        )

        X_neg_raw = torch.cat(neg_x_list, dim=0)

        aug_ef_parts = None  # increasestrongafter's  edge_feat list
        if mode == 'mimicry' and benign_x_list is not None and len(benign_x_list) > 0:
            # Mimicry [32]: towardattack nodeconnectbenignedge, modulomalicioussignal
            X_neg_aug = X_neg_raw
            aug_e_cols = []
            aug_ef_parts = []
            benign_all_x = torch.cat(benign_x_list, dim=0)
            n_benign_nodes = benign_all_x.size(0)

            for ei, efi, off, nc in zip(neg_e_list, neg_ef_list, offsets_neg, neg_node_counts):
                aug_e_cols.append(ei + off)
                aug_ef_parts.append(efi)
                if nc > 0 and n_benign_nodes > 0:
                    n_inject = max(1, nc // 3)
                    src = torch.randint(off, off + nc, (n_inject,), device=device)
                    dst = torch.randint(off, off + nc, (n_inject,), device=device)
                    injected = torch.stack([src, dst], dim=0)
                    aug_e_cols.append(injected)
                    # noteintoedge's  edge_feat is 0 (processclass) 
                    aug_ef_parts.append(torch.zeros(n_inject, dtype=efi.dtype, device=device))

            E_neg = torch.cat(aug_e_cols, dim=1)
        else:
            # standard modulo: toeachsubgraphincreasestrong, againconcatenate
            # no_aug: drop_edge_p=0, feat_mask_p=0 → notincreasestrong
            # graphcl: all edge+
            # gca: degreeedge+
            aug_e_cols = []
            aug_ef_parts = []
            aug_x_parts = []
            for xi, ei, efi, off in zip(neg_x_list, neg_e_list, neg_ef_list, offsets_neg):
                # edgeincreasestrong (localindex) + stepfilter edge_feat
                if self.drop_edge_p > 0 and ei.numel() > 0:
                    n_orig = ei.size(1)
                    if self.use_degree_coop_augment:
                        ei_aug = self._augment_edges_degree_aware(ei, self.drop_edge_p)
                    else:
                        ei_aug = self._augment_edges(ei, self.drop_edge_p)
                    # edge_feat andedgeafter's edgenumberalign
                    if efi.numel() > 0 and ei_aug.size(1) < n_orig:
                        efi_aug = efi[:ei_aug.size(1)]
                    else:
                        efi_aug = efi
                else:
                    ei_aug = ei
                    efi_aug = efi
                aug_e_cols.append(ei_aug + off)
                aug_ef_parts.append(efi_aug)
                # featureincreasestrong (eachsubgraph,  shape notmatch) 
                if self.feat_mask_p > 0 and xi.numel() > 0:
                    if self.use_degree_coop_augment:
                        xi_aug = self._augment_features_degree_aware(xi, self.feat_mask_p, ei_aug)
                    else:
                        xi_aug = self._augment_features(xi, self.feat_mask_p)
                else:
                    xi_aug = xi
                aug_x_parts.append(xi_aug)
            E_neg = torch.cat(aug_e_cols, dim=1)
            X_neg_aug = torch.cat(aug_x_parts, dim=0)

        EF_neg = torch.cat(aug_ef_parts, dim=0) if aug_ef_parts else None

        Z_layers = self.encoder(X_neg_aug, E_neg, edge_feat=EF_neg, return_all=True)
        H_last = Z_layers[-1]
        sums = torch.zeros((Bn, H_last.size(1)), device=device)
        cnts = torch.zeros(Bn, device=device)
        sums.index_add_(0, graph_ids_neg, H_last)
        cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
        means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
        return F.normalize(self.proj_head(means), dim=-1)  # [Bn, D]

    def _build_neg_block_from_snapshots(self, Bc: int, device: torch.device):
        """frommalicious node ego insample Bc centernode, its r-hop ego subgraphencodebecometwo's negative sample (each Bc×D) . 
        return: ([Z_neg_block_view1[Bc,D], Z_neg_block_view2[Bc,D]], freq_weights_neg[Bc]); ifisemptyreturn ([], zeros). 
        """
        pool = self._mal_ego_pool if hasattr(self, '_mal_ego_pool') else None
        if not pool:
            return [], torch.zeros(Bc, device=device)

        if len(pool) >= Bc:
            chosen = random.sample(pool, k=Bc)
        else:
            chosen = [random.choice(pool) for _ in range(Bc)]

        neg_ego_max = min(5, self.ego_max_nodes)
        x_list, e_list, node_counts = [], [], []
        for (sidx, center) in chosen:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            try:
                sub = self._ego_subgraph(g, center=center, r=self.r_hop, max_nodes=neg_ego_max)
            except Exception:
                sub = None
            if sub is None or sub.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            x_np = self._build_node_features(sub)
            eidx, _ef = self._igraph_edges_to_edge_index(sub)
            x_list.append(torch.from_numpy(x_np).to(device))
            e_list.append(eidx.to(device))
            node_counts.append(sub.vcount())

        if sum(node_counts) == 0:
            return [], torch.zeros(Bc, device=device)

        offsets_neg = np.cumsum([0] + node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts) for _ in range(n)],
            device=device
        )
        X_neg = torch.cat([xi for xi in x_list if xi.numel() > 0], dim=0) if any(
            n > 0 for n in node_counts) else torch.zeros((0, self.prop_feat_dim), device=device)

        Z_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if any(n > 0 for n in node_counts):
                if self.use_degree_coop_augment:
                    e_cols = [self._augment_edges_degree_aware(ei, self.drop_edge_p) + off for ei, off in
                              zip(e_list, offsets_neg)]
                else:
                    e_cols = [self._augment_edges(ei, self.drop_edge_p) + off for ei, off in zip(e_list, offsets_neg)]
            else:
                e_cols = []
            EN = torch.cat(e_cols, dim=1) if e_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
            if self.use_degree_coop_augment:
                XN = self._augment_features_degree_aware(X_neg, self.feat_mask_p, EN)
            else:
                XN = self._augment_features(X_neg, self.feat_mask_p)
            ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
            NL = ZN_layers[-1]
            sums = torch.zeros((Bc, NL.size(1)), device=device)
            cnts = torch.zeros(Bc, device=device)
            sums.index_add_(0, graph_ids_neg, NL)
            cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
            means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
            Z_block = F.normalize(self.proj_head(means), dim=-1)
            Z_blocks.append(Z_block)

        # simplestartsee, negative sampleweight1is 1
        w_neg = torch.ones(Bc, dtype=torch.float32, device=device)
        return Z_blocks, w_neg

    def _build_neg_block_mimicry(self, Bc: int, device: torch.device,
                                 benign_x_list=None, benign_e_list=None,
                                 benign_node_counts=None):
        """Mimicry [ProvNinja] 's negative sample: 
        frommalicious ego samplesubgraph, towardwherenoteintobenignedgeandbenign nodefeature, 
        malicioussubgraphbecomebenign, ishard negative. 

        : 
        1. samplemalicioussubgraph
        2. fromwhenbefore batch 's benignsubgraphintakenode
        3. inmalicioussubgraphandbenign nodeaddedge
        4. replacepartmalicious node's featureisbenignfeature (attributemodulo) 
        """
        pool = self._mal_ego_pool if hasattr(self, '_mal_ego_pool') else None
        if not pool:
            return [], torch.zeros(Bc, device=device)

        if len(pool) >= Bc:
            chosen = random.sample(pool, k=Bc)
        else:
            chosen = [random.choice(pool) for _ in range(Bc)]

        neg_ego_max = min(5, self.ego_max_nodes)
        x_list, e_list, node_counts = [], [], []
        for (sidx, center) in chosen:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            try:
                sub = self._ego_subgraph(g, center=center, r=self.r_hop, max_nodes=neg_ego_max)
            except Exception:
                sub = None
            if sub is None or sub.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            x_np = self._build_node_features(sub)
            eidx, _ef = self._igraph_edges_to_edge_index(sub)
            x_list.append(torch.from_numpy(x_np).to(device))
            e_list.append(eidx.to(device))
            node_counts.append(sub.vcount())

        if sum(node_counts) == 0:
            return [], torch.zeros(Bc, device=device)

        # Mimicry increasestrong: towardmalicioussubgraphnoteintobenignsignal
        has_benign = (benign_x_list is not None and len(benign_x_list) > 0
                      and benign_node_counts is not None and sum(benign_node_counts) > 0)

        if has_benign:
            benign_feats_all = torch.cat([bx for bx in benign_x_list if bx.numel() > 0], dim=0)

            for gi in range(len(x_list)):
                xi = x_list[gi]
                ei = e_list[gi]
                nc = node_counts[gi]
                if nc == 0 or xi.numel() == 0:
                    continue

                # (1) featurereplace: will 30% 's malicious nodefeaturereplaceisbenign nodefeature
                replace_ratio = 0.3
                n_replace = max(1, int(nc * replace_ratio))
                replace_idx = torch.randperm(nc, device=device)[:n_replace]
                benign_sample_idx = torch.randint(0, benign_feats_all.size(0), (n_replace,), device=device)
                xi_new = xi.clone()
                xi_new[replace_idx] = benign_feats_all[benign_sample_idx]
                x_list[gi] = xi_new

                # (2) edgenoteinto: inmalicious nodeand"benign node"addedge
                n_inject = max(1, int(ei.size(1) * 0.2))  # noteinto 20% 's edge
                src_new = torch.randint(0, nc, (n_inject,), device=device)
                dst_new = torch.randint(0, nc, (n_inject,), device=device)
                inject_edges = torch.stack([
                    torch.cat([src_new, dst_new]),
                    torch.cat([dst_new, src_new])
                ])
                e_list[gi] = torch.cat([ei, inject_edges], dim=1)

        offsets_neg = np.cumsum([0] + node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts) for _ in range(n)],
            device=device
        )
        X_neg = torch.cat([xi for xi in x_list if xi.numel() > 0], dim=0) if any(
            n > 0 for n in node_counts) else torch.zeros((0, self.prop_feat_dim), device=device)

        Z_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if any(n > 0 for n in node_counts):
                e_cols = [self._augment_edges(ei, self.drop_edge_p) + off
                          for ei, off in zip(e_list, offsets_neg)]
            else:
                e_cols = []
            EN = torch.cat(e_cols, dim=1) if e_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
            XN = self._augment_features(X_neg, self.feat_mask_p)
            ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
            NL = ZN_layers[-1]
            sums = torch.zeros((Bc, NL.size(1)), device=device)
            cnts = torch.zeros(Bc, device=device)
            sums.index_add_(0, graph_ids_neg, NL)
            cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
            means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
            Z_block = F.normalize(self.proj_head(means), dim=-1)
            Z_blocks.append(Z_block)

        w_neg = torch.ones(Bc, dtype=torch.float32, device=device)
        return Z_blocks, w_neg

    def _build_neg_block_from_snapshots_with_pos(
            self,
            Bc: int,
            device: torch.device,
            pos_x_list: List[torch.Tensor],
            pos_e_list: List[torch.Tensor],
            pos_node_counts: List[int],
            pos_ratio: float = 0.5,
            cross_edge_ratio: float = 0.2,
            cross_edge_max: int = 8,
    ):
        """
        based onwhenbefore batch 's positivesubgraph + malicioussubgraphcache, if“fusednegativesubgraph”, againtwoencode. 
        return: ([Z_neg_view1[N_neg,D], Z_neg_view2[N_neg,D]], w_neg[N_neg])
        note: notagainforce Bc , canfusedcountreturn. 
        """
        # ---- 0. maliciouscachecheck ----
        mal_cache = getattr(self, "_mal_ego_cache", None)
        if not mal_cache:  # None orempty
            return [], torch.zeros(0, device=device)

        #  vs parameterto [0,1]
        pos_ratio = float(min(max(pos_ratio, 0.0), 1.0))
        cross_edge_ratio = float(min(max(cross_edge_ratio, 0.0), 1.0))
        cross_edge_max = int(max(0, cross_edge_max))

        x_list: List[torch.Tensor] = []
        e_list: List[torch.Tensor] = []
        node_counts: List[int] = []

        total_pos = len(pos_x_list)
        # not Bc approximately: fusedcountequal tomaliciouscachenumber
        num_build = len(mal_cache)
        if total_pos == 0:
            return [], torch.zeros(0, device=device)

        # ---- 1. malicioussubgraphtraverse: eachmalicioussubgraph1positivesubgraphfused ----
        for m_idx, cache_entry in enumerate(mal_cache):
            x_m_cached, e_m_cached, mal_cnt = cache_entry[0], cache_entry[1], cache_entry[2]
            ef_m_cached = cache_entry[3] if len(cache_entry) > 3 else None
            pi = random.randrange(total_pos)
            xi = pos_x_list[pi]
            ei = pos_e_list[pi]
            nc = int(pos_node_counts[pi]) if pi < len(pos_node_counts) else (
                int(xi.size(0)) if isinstance(xi, torch.Tensor) else 0)

            use_pos = (
                    xi is not None and ei is not None and
                    isinstance(xi, torch.Tensor) and isinstance(ei, torch.Tensor) and
                    nc > 0 and xi.numel() > 0 and ei.numel() > 0
            )

            # ---- 1.1 positivesubgraphnotcanuse: connectuse1malicioussubgraphbit ----
            if not use_pos:
                ridx = random.randrange(len(mal_cache))
                x_rand, e_rand, rand_cnt = mal_cache[ridx][0], mal_cache[ridx][1], mal_cache[ridx][2]
                x_list.append(x_rand.to(device))
                e_list.append(e_rand.to(device))
                node_counts.append(int(rand_cnt))
                continue

            # ---- 1.2 positivesubgraphcanuse: samplepartpositivenode + 1malicioussubgraph, concatenatefused ----
            xi = xi.to(device)
            ei = ei.to(device)
            nc = int(nc)

            k_pos = max(1, int(round(pos_ratio * nc)))
            k_pos = min(k_pos, nc)

            src_pos, dst_pos = ei[0], ei[1]
            mask_pos = (src_pos < k_pos) & (dst_pos < k_pos)
            ei_pos_sub = ei[:, mask_pos]
            xi_pos_sub = xi[:k_pos, :]

            x_m = x_m_cached.to(device)
            e_m = e_m_cached.to(device)
            mal_cnt = int(mal_cnt)

            x_fused = torch.cat([xi_pos_sub, x_m], dim=0)  # [k_pos + mal_cnt, F]
            e_m_shift = e_m + k_pos
            e_fused = torch.cat([ei_pos_sub, e_m_shift], dim=1)
            n_fused = k_pos + mal_cnt

            if k_pos > 0 and mal_cnt > 0 and cross_edge_ratio > 0.0 and cross_edge_max > 0:
                base = min(k_pos, mal_cnt)
                target = int(round(cross_edge_ratio * base))
                num_cross = max(1, min(base, cross_edge_max, target))

                pos_idx = torch.randint(0, k_pos, (num_cross,), device=device)
                mal_idx = torch.randint(0, mal_cnt, (num_cross,), device=device) + k_pos

                cross_edges = torch.stack([pos_idx, mal_idx], dim=0)
                cross_edges_rev = torch.stack([mal_idx, pos_idx], dim=0)
                e_cross = torch.cat([cross_edges, cross_edges_rev], dim=1)

                e_fused = torch.cat([e_fused, e_cross], dim=1)

            x_list.append(x_fused)
            e_list.append(e_fused)
            node_counts.append(n_fused)

        if sum(node_counts) == 0:
            return [], torch.zeros(0, device=device)

        # ---- 2. become1largegraph's node/edge, twoincreasestrong + encode ----
        offsets_neg = np.cumsum([0] + node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts) for _ in range(n)],
            device=device
        )

        if any(n > 0 for n in node_counts):
            X_neg = torch.cat([xi for xi in x_list if xi.numel() > 0], dim=0)
        else:
            X_neg = torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device)

        has_nodes = X_neg.numel() > 0

        # ---- 3. twoincreasestrong + encode, totwo [N_neg, D] 's negative sample ----
        Z_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if has_nodes:
                if self.use_degree_coop_augment:
                    e_cols = [
                        self._augment_edges_degree_aware(ei, self.drop_edge_p) + off
                        for ei, off in zip(e_list, offsets_neg)
                    ]
                else:
                    e_cols = [
                        self._augment_edges(ei, self.drop_edge_p) + off
                        for ei, off in zip(e_list, offsets_neg)
                    ]
                EN = torch.cat(e_cols, dim=1) if e_cols else torch.zeros((2, 0), dtype=torch.long, device=device)

                if self.use_degree_coop_augment:
                    XN = self._augment_features_degree_aware(X_neg, self.feat_mask_p, EN)
                else:
                    XN = self._augment_features(X_neg, self.feat_mask_p)

                ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
                NL = ZN_layers[-1]

                N_neg = len(node_counts)
                sums = torch.zeros((N_neg, NL.size(1)), device=device)
                cnts = torch.zeros(N_neg, device=device)
                sums.index_add_(0, graph_ids_neg, NL)
                cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
                means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))

                Z_blocks.append(F.normalize(self.proj_head(means), dim=-1))
            else:
                # : nonodetime 0 vector
                Z_blocks.append(torch.zeros((0, self.enc_out_dim), dtype=torch.float32, device=device))

            # beforetohard negative1weight 1 (lengthis N_neg) 
            N_neg = len(node_counts)
            w_neg = torch.ones(N_neg, dtype=torch.float32, device=device)
        return Z_blocks, w_neg

    def _build_neg_block_from_tokens(self, Bc: int, device: torch.device):
        """based oncorpuscorrupt's negative samplebuild, returntwo's  Bc×D block andweight. 
        dependency self._ego_cache (subgraph) and self.malicious_node_tokens. 
        ifconditionnotthenreturnemptyandzeroweight. 
        """
        if not (self.use_malicious_negatives
                and hasattr(self, 'malicious_node_tokens') and len(self.malicious_node_tokens) > 0
                and hasattr(self, '_ego_cache') and len(self._ego_cache) > 0):
            return [], torch.zeros(Bc, device=device)

        all_subs, all_x, all_e, all_w = [], [], [], []
        for subs_prev, x_prev, e_prev, w_prev in self._ego_cache:
            all_subs.extend(subs_prev)
            all_x.extend(x_prev)
            all_e.extend(e_prev)
            all_w.extend(w_prev)

        total_prev = len(all_subs)
        if total_prev == 0:
            return [], torch.zeros(Bc, device=device)

        # ifnot Bc, hasputbacksample
        replace = total_prev < Bc
        idxs = np.random.choice(total_prev, size=Bc, replace=replace)

        X_neg_list, E_neg_list, node_counts_neg, freq_neg = [], [], [], []
        for i in idxs:
            sub, xi, ei, w = all_subs[i], all_x[i], all_e[i], all_w[i]
            xneg_np = self._corrupt_features_with_malicious(
                sub, xi.cpu().numpy(),
                ratio=float(self.mal_neg_ratio),
                node_token_len=int(self.mal_neg_node_token_len)
            )
            X_neg_list.append(torch.from_numpy(xneg_np).to(device))
            E_neg_list.append(ei.to(device) if ei.device != device else ei)
            node_counts_neg.append(sub.vcount())
            freq_neg.append(w)

        offsets_neg = np.cumsum([0] + node_counts_neg[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts_neg) for _ in range(n)],
            device=device
        )
        X_neg = torch.cat(X_neg_list, dim=0)

        Z_neg_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if self.use_degree_coop_augment:
                e_cols = [self._augment_edges_degree_aware(ei, self.drop_edge_p) + off for ei, off in
                          zip(E_neg_list, offsets_neg)]
            else:
                e_cols = [self._augment_edges(ei, self.drop_edge_p) + off for ei, off in zip(E_neg_list, offsets_neg)]
            EN = torch.cat(e_cols, dim=1)
            if self.use_degree_coop_augment:
                XN = self._augment_features_degree_aware(X_neg, self.feat_mask_p, EN)
            else:
                XN = self._augment_features(X_neg, self.feat_mask_p)
            ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
            NL = ZN_layers[-1]
            sums = torch.zeros((Bc, NL.size(1)), device=device)
            cnts = torch.zeros(Bc, device=device)
            sums.index_add_(0, graph_ids_neg, NL)
            cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
            means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
            Z_neg_blocks.append(F.normalize(self.proj_head(means), dim=-1))

        w_neg = torch.tensor(freq_neg, dtype=torch.float32, device=device)
        return Z_neg_blocks, w_neg

    def _corrupt_features_with_malicious(self, g, X_base: np.ndarray, ratio: float, node_token_len: int) -> np.ndarray:
        n = g.vcount()
        out = X_base.copy()
        if ratio <= 0 or not hasattr(self, "malicious_node_tokens") or len(self.malicious_node_tokens) == 0:
            return out
        for i in range(n):
            if random.random() < ratio:
                # fromnodecorpusinextractifmalicioustoken
                tokens = self._sample_malicious_tokens(max(1, int(node_token_len)))
                if not tokens:
                    continue
                # convertis embedding vector (viaalreadyhas's  W2V modulo) 
                vec = self._w2v_vector_from_tokens(tokens)
                out[i] = vec.astype(np.float32)

        return out

    def save_malicious_snapshot_stats(self, save_path: str = "malicious_tokens_log.txt"):
        """willglobalmalicious nodesamplefromsourcestatisticaddwritelogfile"""
        stats = getattr(self, "malicious_snapshot_stats", None)
        if not stats:
            print("[⚠️ nocansave's malicious nodesamplestatistic]")
            return

        total = sum(stats.values())

        with open(save_path, "a", encoding="utf-8") as f:
            f.write("\n[📊 globalmalicious nodesamplestatistic]\n")
            for sid, cnt in sorted(stats.items()):
                pct = cnt / total * 100
                f.write(f"  Snapshot {sid:02d}: {cnt} time ({pct:.2f}%)\n")
            f.write(f"  total: {total} timesample\n")
            f.write("=" * 60 + "\n")

        print(f"[✅ alreadywillsamplestatisticaddsaveto]: {save_path}")

    def _collect_subgraph_tokens(self, sub, max_tokens: int = 512) -> List[str]:
        """collectsubgraph's  tokens (based onnode properties) , total. 
        whenbeforeimplementation: mergehasnode's  properties word. 
        """
        toks: List[str] = []
        for i in range(sub.vcount()):
            toks.extend(self._get_node_tokens(sub, i))
            if len(toks) >= int(max_tokens):
                break
        if len(toks) > int(max_tokens):
            toks = toks[: int(max_tokens)]
        return toks

    def _fingerprint_from_tokens(self, tokens: List[str], m_bits: int = 1024) -> np.ndarray:
        """will tokens mappingislengthis m_bits 's  0/1 vector (MD5 takemodulo) . """
        m = max(1, int(m_bits))
        fp = np.zeros(m, dtype=np.float32)
        if not tokens:
            return fp
        for t in tokens:
            # use MD5 mappingto [0, m)
            h = hashlib.md5(t.encode('utf-8')).hexdigest()
            idx = int(h, 16) % m
            fp[idx] = 1.0
        return fp

    def _subgraph_fingerprint(self, sub, m_bits: int = 1024) -> np.ndarray:
        toks = self._collect_subgraph_tokens(sub)
        return self._fingerprint_from_tokens(toks, m_bits=m_bits)

    def _subgraph_semantic_vector(self, sub) -> np.ndarray:
        """willsubgraph tokens aggregateissemanticvector (Word2Vec mean, alreadynormalize) . """
        toks = self._collect_subgraph_tokens(sub)
        return self._w2v_vector_from_tokens(toks)

    # alreadyremove WL treeandguidesimilaritylossmethod

    def embed_nodes(self):
        return self.snapshot_node_embeddings[-1] if self.snapshot_node_embeddings else {}

    def embed_edges(self):
        return {}

    def prepare_text_encoder(self):
        """optional's process: beforetrain/load Word2Vec modulo. 
        somelineflow (for exampleonlysnapshot embeddingandnotuse train) canfirst usemethod. 
        """
        self._ensure_w2v_model()

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        if not self.snapshot_node_embeddings:
            raise RuntimeError("nonodeembedding, first use train()")
        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshots)))

        result = []
        α = float(np.clip(self.attr_weight_alpha, 0.0, 1.0))

        for i in snapshot_sequence:
            g = self.snapshots[i]
            if g is None or g.vcount() == 0:
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue

            emb = self.snapshot_node_embeddings[i]
            if not emb:
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue

            N = g.vcount()

            # ===== read node → vector =====
            vecs = np.zeros((N, self.enc_out_dim), dtype=np.float32)
            valid = np.zeros(N, dtype=bool)
            for j in range(N):
                nid = g.vs[j]['name']
                v = emb.get(nid)
                if v is not None:
                    vecs[j] = v
                    valid[j] = True

            if not valid.any():
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue

            # ===== base weight (frequencypriority first , otherwisedegree) =====
            if 'frequency' in g.vs.attributes():
                base_w = np.array(g.vs['frequency'], dtype=np.float32)
                base_w = np.maximum(base_w, 0)
            else:
                base_w = np.maximum(np.array(g.degree(), dtype=np.float32), 0)

            # divide 0
            b_norm = base_w / (base_w.mean() + 1e-12)

            # ===== attributeseeweight 1 - p(attr) =====
            # igraph 's  Vertex without .get method, use attributes()/column
            if 'properties' in g.vs.attributes():
                props = [str(p) for p in g.vs['properties']]
            else:
                props = [''] * N

            prop_w = {}
            for p, w in zip(props, base_w):
                if w > 0:
                    prop_w[p] = prop_w.get(p, 0.0) + w

            if prop_w:
                maxv = max(prop_w.values())
                prop_norm = {k: v / maxv for k, v in prop_w.items()}
            else:
                prop_norm = {}

            a = np.array([1.0 - prop_norm.get(props[j], 0.0) for j in range(N)], dtype=np.float32)
            a_norm = a / (a.mean() + 1e-12)

            # ===== most endweight w_eff =====
            w_eff = (1 - α) * b_norm + α * a_norm
            w_eff = np.maximum(w_eff, 0)

            if w_eff.sum() == 0:
                snapshot_vec = vecs[valid].mean(axis=0)
            else:
                snapshot_vec = (vecs * w_eff[:, None]).sum(axis=0) / (w_eff.sum() + 1e-12)

            result.append(snapshot_vec.astype(np.float32))

        arr = np.vstack(result) if result else np.zeros((0, self.enc_out_dim), dtype=np.float32)
        print(f"[GCC-Dev] Snapshot embeddings: {arr.shape}")
        return arr

    def compute_malicious_deviation_per_snapshot(
            self,
            snapshot_sequence: Optional[List[int]] = None,
            metric: str = 'cosine',
            center_weighting: str = 'none',
            save_path: str = "malicious_tokens_log.txt",
    ) -> List[Dict[str, object]]:
        """
        save: eachsnapshotinmalicious nodeinnode“descending”'s 100 vs  (rank_pct) , 
        statisticmean, benign vs , max/minnodeetc.. 
        """

        if not self.snapshot_node_embeddings:
            raise RuntimeError("nonodeembedding, first use train()")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshots)))

        def _weights_for(g):
            if center_weighting == 'none':
                return None
            if center_weighting == 'degree':
                deg = np.asarray(g.degree(), dtype=np.float32)
                return np.maximum(deg, 0.0)
            # auto: frequencypriority first , backdegree
            try:
                freqs = g.vs['frequency'] if 'frequency' in g.vs.attributes() else None
            except Exception:
                freqs = None
            if freqs is not None:
                w = np.zeros(g.vcount(), dtype=np.float32)
                for idx in range(g.vcount()):
                    try:
                        v = float(freqs[idx])
                    except Exception:
                        v = 0.0
                    if np.isfinite(v) and v > 0:
                        w[idx] = v
                if w.sum() > 0:
                    return w
            deg = np.asarray(g.degree(), dtype=np.float32)
            return np.maximum(deg, 0.0)

        rows: List[Dict[str, object]] = []

        with open(save_path, "a", encoding="utf-8") as f:
            f.write("\n[malicious node100 vs statistic]\n")
            f.write("=" * 70 + "\n")

            for i in snapshot_sequence:
                g = self.snapshots[i]
                if g is None or g.vcount() == 0:
                    continue

                emb_dict = self.snapshot_node_embeddings[i] if i < len(self.snapshot_node_embeddings) else {}
                if not emb_dict:
                    continue

                names, vecs, labels = [], [], []
                for local_idx in range(g.vcount()):
                    nid = g.vs[local_idx]['name']
                    vec = emb_dict.get(nid)
                    if vec is None:
                        continue
                    names.append(nid)
                    vecs.append(np.asarray(vec, dtype=np.float32))
                    try:
                        lab = int(g.vs[local_idx].attributes().get('label', 0))
                    except Exception:
                        lab = 0
                    labels.append(lab)

                if not vecs:
                    continue

                V = np.vstack(vecs).astype(np.float32)
                W = _weights_for(g)
                if W is not None and len(names) == g.vcount() and W.sum() > 0:
                    center = (V * W[:, None]).sum(axis=0) / (W.sum() + 1e-12)
                else:
                    center = V.mean(axis=0)

                if metric == 'l2':
                    devs = np.linalg.norm(V - center, axis=1)
                else:
                    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
                    cn = center / (np.linalg.norm(center) + 1e-12)
                    devs = 1.0 - np.matmul(Vn, cn)

                devs1d = np.asarray(devs).reshape(-1)
                N = int(devs1d.shape[0])

                order = np.argsort(-devs1d)  # indices
                rank_map = {int(idx): int(r + 1) for r, idx in enumerate(order)}  # 1..N

                max_idx = int(np.argmax(devs1d))
                min_idx = int(np.argmin(devs1d))
                max_name, max_val = names[max_idx], float(devs1d[max_idx])
                min_name, min_val = names[min_idx], float(devs1d[min_idx])

                mal_idx = [k for k, lab in enumerate(labels) if lab == 1]
                ben_idx = [k for k, lab in enumerate(labels) if lab == 0]
                if not mal_idx:
                    continue

                num_mal = len(mal_idx)
                num_benign = len(ben_idx)
                benign_ratio = num_benign / N if N > 0 else 0.0

                mean_dev_all = float(np.mean(devs1d))

                mal_rank_entries = []
                rank_pcts = []
                for idx in mal_idx:
                    rk = rank_map[idx]  # 1..N
                    rk_pct = rk / N * 100.0
                    rank_pcts.append(rk_pct)
                    mal_rank_entries.append((names[idx], rk, rk_pct, float(devs1d[idx])))

                mean_mal_rank_pct = float(np.mean(rank_pcts)) if rank_pcts else 0.0

                print(f"\n[Snapshot {i:02d}]")
                print(f"  mean: {mean_dev_all:.6f}")
                print(f"  benign nodenumber: {num_benign} ({benign_ratio:.2%})")
                print(f"  maxnode: {max_name} ({max_val:.6f})")
                print(f"  minnode: {min_name} ({min_val:.6f})")
                print(f"  malicious node mean100 vs : {mean_mal_rank_pct:.2f}%")
                print("  malicious node: ")
                #  rank ascendingopen (more) 
                mal_rank_entries.sort(key=lambda x: x[1])
                for name, rk, rk_pct, dev in mal_rank_entries:
                    print(f"    - {name}: rank={rk}, rank_pct={rk_pct:.2f}%, dev={dev:.6f}")

                f.write(f"Snapshot {i:02d}: mean={mean_dev_all:.6f}\n")
                f.write(f"  benign nodenumber={num_benign} ({benign_ratio:.2%})\n")
                f.write(f"  maxnode: {max_name} ({max_val:.6f})\n")
                f.write(f"  minnode: {min_name} ({min_val:.6f})\n")
                f.write(f"  malicious node mean100 vs : {mean_mal_rank_pct:.2f}%\n")
                for name, rk, rk_pct, dev in mal_rank_entries:
                    f.write(f"    - {name}: rank={rk}, rank_pct={rk_pct:.2f}%, dev={dev:.6f}\n")
                f.write("-" * 70 + "\n")

                rows.append({
                    'snapshot': i,
                    'num_nodes': N,
                    'num_mal': num_mal,
                    'num_benign': num_benign,
                    'benign_ratio': benign_ratio,
                    'mean_dev_all': mean_dev_all,
                    'max_dev_node': max_name,
                    'max_dev_val': max_val,
                    'min_dev_node': min_name,
                    'min_dev_val': min_val,
                    'mean_mal_rank_pct': mean_mal_rank_pct,
                    # eachmalicious node: (name, rank, rank_pct, deviation)
                    'mal_rank_table': mal_rank_entries,
                })

            f.write("=" * 70 + "\n")

        print(f"[GCC-Dev] malicious nodestatisticsavedto: {save_path}")
        return rows

    def save_model(self, path: Optional[str] = None):
        path = path or self.model_path
        state = {
            'params': {
                'use_temporal': self.use_temporal,
                'prop_feat_dim': self.prop_feat_dim,
                'enc_hidden_dim': self.enc_hidden_dim,
                'enc_out_dim': self.enc_out_dim,
                'gin_layers': self.gin_layers,
                'dropout': self.dropout,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'temperature': self.temperature,
                'r_hop': self.r_hop,
                'ego_max_nodes': self.ego_max_nodes,
                'drop_edge_p': self.drop_edge_p,
                'feat_mask_p': self.feat_mask_p,
                'train_indices': self.train_snapshot_indices,
                'model_path': self.model_path,
                'anomaly_alpha': self.anomaly_alpha,
                'use_sample_weights': self.use_sample_weights,
                # W2V config
                'w2v_window': self.w2v_window,
                'w2v_min_count': self.w2v_min_count,
                'w2v_sg': self.w2v_sg,
                'w2v_epochs': self.w2v_epochs,
                'w2v_pretrained_path': self.w2v_pretrained_path,
                # maliciousTokenconfig
                'mal_stopwords': list(self.mal_stopwords) if self.mal_stopwords else [],
                'mal_print_tokens': self.mal_print_tokens,
                'use_degree_coop_augment': self.use_degree_coop_augment,
                'attr_weight_alpha': self.attr_weight_alpha,
            },
            'encoder': self.encoder.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'temporal': self.temporal.state_dict(),
            'snapshot_node_embeddings': self.snapshot_node_embeddings,
        }
        torch.save(state, path)
        print(f"[GCC-Dev] Model saved to {path}")

    @classmethod
    def load(cls, snapshot_sequence, path: Optional[str] = None):
        path = path or cls._default_path
        print(f"[GCC-Dev] Loading model from {path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)
        raw_params = dict(state.get('params', {}))
        allowed = {
            'use_temporal',
            'prop_feat_dim', 'enc_hidden_dim', 'enc_out_dim', 'gin_layers', 'dropout',
            'num_epochs', 'batch_size', 'lr', 'temperature',
            'r_hop', 'ego_max_nodes', 'drop_edge_p', 'feat_mask_p', 'train_indices', 'model_path',
            'anomaly_alpha', 'use_sample_weights',
            # W2V config
            'w2v_window', 'w2v_min_count', 'w2v_sg', 'w2v_epochs', 'w2v_pretrained_path',
            # maliciousTokenconfig
            'mal_stopwords', 'mal_print_tokens',
            'use_degree_coop_augment',
        }
        params = {k: v for k, v in raw_params.items() if k in allowed}
        inst = cls(snapshot_sequence, **params)
        inst.encoder.load_state_dict(state['encoder'])
        inst.proj_head.load_state_dict(state['proj_head'])
        if 'temporal' in state:
            try:
                inst.temporal.load_state_dict(state['temporal'])
            except Exception as e:
                print(f"[GCC-Dev] Warning: load temporal failed: {e}")
        inst.snapshot_node_embeddings = state.get('snapshot_node_embeddings', [])
        print("[GCC-Dev] Model loaded successfully")
        return inst

    def _resolve_train_indices(self, indices: Optional[Union[Iterable[int], Tuple[int, int], int]]) -> List[int]:
        total = len(self.snapshots)
        if total == 0:
            return []
        if indices is None:
            raw = list(range(total))
        elif isinstance(indices, int):
            raw = [indices]
        elif isinstance(indices, tuple) and len(indices) == 2:
            a, b = int(indices[0]), int(indices[1])
            if a > b:
                a, b = b, a
            raw = list(range(a, b + 1))
        else:
            raw = list(indices)  # type: ignore[arg-type]
        valid = sorted({int(i) for i in raw if 0 <= int(i) < total})
        if not valid:
            raise ValueError("train_indices notcontainshasindex")
        return valid

    def _ego_subgraph(self, g, center: int, r: int, max_nodes: int):
        """BFS ordertake ego subgraph, . """
        from collections import deque
        visited = [center]
        visited_set = {center}
        queue = deque([center])
        while queue and len(visited) < max_nodes:
            v = queue.popleft()
            for nb in g.neighbors(v, mode="all"):
                if nb not in visited_set and len(visited) < max_nodes:
                    visited.append(nb)
                    visited_set.add(nb)
                    queue.append(nb)
        return g.subgraph(sorted(visited))

    # ---------- Word2Vec support ----------
    def _tokenize_properties(self, text: str) -> List[str]:
        """modifyversion token take (preservepath, , UUID, numberetc., not) """
        if not text:
            return []

        s = str(text).strip()
        # usepositivethentake's  [A-Za-z0-9_-.:/\] segment, preservepath, UUID, number
        tokens = re.findall(r"[A-Za-z0-9_\-./:\\]+", s)

        return tokens

    def _get_node_tokens(self, g, i: int) -> List[str]:
        try:
            prop = g.vs[i]['properties']
        except Exception:
            prop = g.vs[i].attributes().get('properties', '')
        return self._tokenize_properties(str(prop))

    def _gather_neighbor_tokens(self, g, i: int) -> List[str]:
        """strategy's  token collect: 1-hop, containsself,  256  token. """
        try:
            nodes = set(g.neighborhood(vertices=i, order=1))
        except Exception:
            nodes = {i}
        out: List[str] = []
        for nid in nodes:
            out.extend(self._get_node_tokens(g, nid))
            if len(out) >= 256:
                break
        if len(out) > 256:
            out = out[:256]
        return out

    def _augment_tokens(self, tokens: List[str]) -> List[str]:
        """to token increasestrong: 
        - 10% probabilitydropword
        - appendorder bigram (twowordconcatenate) 
        - at mostpreserve 256  token
        """
        if not tokens:
            return []
        kept = [t for t in tokens if random.random() > 0.1]
        if not kept:
            kept = list(tokens)
        # append bigram
        bigrams = [kept[i] + '_' + kept[i + 1] for i in range(len(kept) - 1)] if len(kept) > 1 else []
        out = kept + bigrams
        if len(out) > 256:
            out = out[:256]
        return out

    def _collect_w2v_corpus(self) -> List[List[str]]:
        seen_props: Dict[str, List[str]] = {}
        ids = self.train_snapshot_indices or list(range(len(self.snapshots)))
        for sidx in ids:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                continue
            for i in range(g.vcount()):
                try:
                    prop = g.vs[i]['properties']
                except Exception:
                    prop = g.vs[i].attributes().get('properties', '')
                key = str(prop)
                if key not in seen_props:
                    seen_props[key] = self._tokenize_properties(key)
        corpus = [tokens for tokens in seen_props.values() if tokens]
        return corpus

    def _ensure_w2v_model(self):
        if self._w2v_model is not None:
            return
        try:
            import importlib
            _w2v_mod = importlib.import_module('gensim.models')
            Word2Vec = getattr(_w2v_mod, 'Word2Vec')
        except Exception:
            raise RuntimeError("[GCC-Dev] needs gensim canuse Word2Vec feature, first  gensim. ")
        if isinstance(self.w2v_pretrained_path, str) and os.path.exists(self.w2v_pretrained_path):
            try:
                self._w2v_model = Word2Vec.load(self.w2v_pretrained_path)
                vec_dim = int(getattr(self._w2v_model.wv, 'vector_size', self.prop_feat_dim))
                if int(vec_dim) == int(self.prop_feat_dim):
                    print("[GCC-Dev] alreadyloadtrain Word2Vec modulo. ")
                    return
                else:
                    print(
                        f"[GCC-Dev] trainvectordimension({vec_dim}) != prop_feat_dim({self.prop_feat_dim}), willmodifyistrainbymatchdimension. ")
                    self._w2v_model = None
            except Exception as e:
                print(f"[GCC-Dev] loadtrain Word2Vec failed: {e}, willtrytrain. ")
        corpus = self._collect_w2v_corpus()
        if not corpus:
            raise RuntimeError("[GCC-Dev] W2V corpusisempty, withoutbuild Word2Vec feature. ")
        print(
            f"[GCC-Dev] currentlytrainword2vec | corpus={len(corpus)} | dim={int(self.prop_feat_dim)} | window={int(self.w2v_window)} | min_count={int(self.w2v_min_count)} | sg={int(self.w2v_sg)} | epochs={int(self.w2v_epochs)}")
        self._w2v_model = Word2Vec(
            sentences=corpus,
            vector_size=int(self.prop_feat_dim),
            window=int(self.w2v_window),
            min_count=int(self.w2v_min_count),
            sg=int(self.w2v_sg),
            workers=4,
            epochs=int(self.w2v_epochs),
        )
        print(f"[GCC-Dev] train Word2Vec done: corpus={len(corpus)} entry, dim={int(self.prop_feat_dim)}")

    def _w2v_vector_from_tokens(self, tokens: List[str]) -> np.ndarray:
        if not tokens or self._w2v_model is None:
            return np.zeros(int(self.prop_feat_dim), dtype=np.float32)
        vecs = []
        wv = self._w2v_model.wv
        for t in tokens:
            if t in wv:
                vecs.append(wv[t])
        if not vecs:
            return np.zeros(int(self.prop_feat_dim), dtype=np.float32)
        v = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def _build_node_features(self, g) -> np.ndarray:
        n = g.vcount()
        if self.prop_feat_dim <= 0:
            return np.ones((n, 1), dtype=np.float32)
        if self._w2v_model is None:
            self._ensure_w2v_model()
        X = np.zeros((n, int(self.prop_feat_dim)), dtype=np.float32)
        for i in range(n):
            try:
                prop = g.vs[i]['properties']
            except Exception:
                prop = g.vs[i].attributes().get('properties', '')
            key = str(prop)
            if key in self._prop_cache:
                X[i] = self._prop_cache[key]
                continue
            tokens = self._tokenize_properties(key)
            vec = self._w2v_vector_from_tokens(tokens)
            self._prop_cache[key] = vec
            X[i] = vec
        return X

    def _igraph_edges_to_edge_index(self, g):
        """return (edge_index [2, E*2], edge_cat [E*2] integerclassindex)"""
        edges = g.get_edgelist()
        if len(edges) == 0:
            return (torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.zeros(0, dtype=torch.long, device=self.device))
        src, dst, cats = [], [], []
        has_actions = 'actions' in g.es.attributes() if g.ecount() > 0 else False
        for i, (u, v) in enumerate(edges):
            action_str = str(g.es[i].attributes().get('actions', '')) if has_actions else ''
            cat = classify_edge(action_str)
            src.append(u); dst.append(v); cats.append(cat)
            src.append(v); dst.append(u); cats.append(cat)
        return (torch.tensor([src, dst], dtype=torch.long, device=self.device),
                torch.tensor(cats, dtype=torch.long, device=self.device))

    def augment_ego(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_feat: torch.Tensor = None,
                    drop_edge_p: float = None, feat_mask_p: float = None):
        """1's  ego subgraphincreasestrongfunction,  Stage 1 / Stage 2 use. 

        Args:
            x: nodefeature [N, D]
            edge_index: edgeindex [2, E]
            edge_feat: edgeclass [E] (optional)
            drop_edge_p: edgeprobability, None timeuse self.drop_edge_p
            feat_mask_p: featureprobability, None timeuse self.feat_mask_p

        Returns:
            (x_aug, edge_index_aug, edge_feat_aug)
        """
        dp = drop_edge_p if drop_edge_p is not None else self.drop_edge_p
        fp = feat_mask_p if feat_mask_p is not None else self.feat_mask_p

        # --- edge ( keep mask, stepfilter edge_feat) ---
        ei_aug = edge_index
        ef_aug = edge_feat
        if edge_index.numel() > 0 and dp > 0:
            E = edge_index.size(1)
            if self.use_degree_coop_augment:
                src, dst = edge_index[0], edge_index[1]
                num_nodes = int(torch.max(torch.stack([src, dst])).item() + 1)
                deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
                deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
                deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
                dmin, dmax = deg.min(), deg.max()
                deg_norm = (deg - dmin) / (dmax - dmin + 1e-12) if (dmax - dmin) > 1e-12 else torch.zeros_like(deg)
                s_e = 0.5 * (deg_norm[src] + deg_norm[dst])
                p_e = torch.clamp(dp * s_e, 0.0, 1.0)
                keep = torch.rand_like(p_e) > p_e
            else:
                keep = torch.rand(E, device=edge_index.device) > dp
            if keep.sum() < 1:
                keep[random.randrange(0, E)] = True
            ei_aug = edge_index[:, keep]
            if edge_feat is not None and edge_feat.numel() > 0:
                ef_aug = edge_feat[keep]

        x_aug = x
        if x.numel() > 0 and fp > 0:
            if self.use_degree_coop_augment and ei_aug.numel() > 0:
                N = x.size(0)
                src_a, dst_a = ei_aug[0], ei_aug[1]
                nn = max(N, int(torch.max(torch.stack([src_a, dst_a])).item() + 1)) if src_a.numel() > 0 else N
                deg = torch.zeros(nn, dtype=torch.float32, device=x.device)
                deg.index_add_(0, src_a, torch.ones_like(src_a, dtype=torch.float32))
                deg.index_add_(0, dst_a, torch.ones_like(dst_a, dtype=torch.float32))
                deg = deg[:N]
                dmin, dmax = deg.min(), deg.max()
                deg_norm = (deg - dmin) / (dmax - dmin + 1e-12) if (dmax - dmin) > 1e-12 else torch.zeros_like(deg)
                p_node = torch.clamp(fp * (1.0 - deg_norm), 0.0, 1.0)
                mask = (torch.rand_like(x) < p_node.view(-1, 1)).float()
                x_aug = x * (1.0 - mask)
            else:
                mask = (torch.rand_like(x) < fp).float()
                x_aug = x * (1.0 - mask)

        return x_aug, ei_aug, ef_aug

    def _augment_edges(self, edge_index: torch.Tensor, drop_p: float) -> torch.Tensor:
        """original (all ) edgeincreasestrong: not"""
        if edge_index.numel() == 0 or drop_p <= 0:
            return edge_index
        E = edge_index.size(1)
        keep = torch.rand(E, device=edge_index.device) > drop_p
        if keep.sum() < 1:
            keep[random.randrange(0, E)] = True
        return edge_index[:, keep]

    def _augment_features(self, x: torch.Tensor, mask_p: float) -> torch.Tensor:
        """original (all ) feature: not"""
        if x.numel() == 0 or mask_p <= 0:
            return x
        mask = (torch.rand_like(x) < mask_p).float()
        return x * (1.0 - mask)

    def _augment_edges_degree_aware(self, edge_index: torch.Tensor, drop_p: float) -> torch.Tensor:
        if edge_index.numel() == 0 or drop_p <= 0:
            return edge_index
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]
        if src.numel() == 0:
            return edge_index
        num_nodes = int(torch.max(torch.stack([src, dst])).item() + 1)
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        dmin = torch.min(deg)
        dmax = torch.max(deg)
        if float(dmax.item() - dmin.item()) < 1e-12:
            deg_norm = torch.zeros_like(deg)
        else:
            deg_norm = (deg - dmin) / (dmax - dmin + 1e-12)
        s_e = 0.5 * (deg_norm[src] + deg_norm[dst])
        p_e = torch.clamp(drop_p * s_e, 0.0, 1.0)
        keep = (torch.rand_like(p_e) > p_e)
        if keep.sum() < 1:
            keep[random.randrange(0, keep.numel())] = True
        return edge_index[:, keep]

    def _augment_features_degree_aware(self, x: torch.Tensor, mask_p: float, edge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or mask_p <= 0 or edge_index.numel() == 0:
            return x
        device = x.device
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        num_nodes = max(N, int(torch.max(torch.stack([src, dst])).item() + 1))
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg[:N]
        dmin = torch.min(deg)
        dmax = torch.max(deg)
        if float(dmax.item() - dmin.item()) < 1e-12:
            deg_norm = torch.zeros_like(deg)
        else:
            deg_norm = (deg - dmin) / (dmax - dmin + 1e-12)
        p_node = torch.clamp(mask_p * (1.0 - deg_norm), 0.0, 1.0)
        rand = torch.rand_like(x)
        mask = (rand < p_node.view(-1, 1)).float()
        return x * (1.0 - mask)

    def _weighted_contrastive_loss(
        self,
        Z_pos: torch.Tensor,
        Z_neg: Optional[torch.Tensor],
        temperature: float,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        paperformula (5)(6): hasweightedcontrastive loss. 

        Z_pos: [2*Bp, D] benign sampleembedding (twowrong: v1_0, v2_0, v1_1, v2_1, ...) 
        Z_neg: [2*Bn, D] malicious sampleembedding (twowrong) , canis None
        temperature: degreeparameter τ
        beta: strength β, hard negativeweight's indegreedegree

        positive sample P(b): 1anchor's additionally1 + otherbenign sample's has
        negative sample N(b): hasmalicious sample's has
        weight w_n = softmax(β * sim(z_b, z_n) / τ) hard negative
        """
        Z_pos = F.normalize(Z_pos, dim=-1)
        Np = Z_pos.size(0)  # 2*Bp

        if Z_neg is not None and Z_neg.size(0) > 0:
            Z_neg = F.normalize(Z_neg, dim=-1)
            Nn = Z_neg.size(0)  # 2*Bn
        else:
            Nn = 0

        # positive samplesimilarity [Np, Np]
        sim_pp = torch.mm(Z_pos, Z_pos.t()) / temperature
        mask_self = torch.eye(Np, device=Z_pos.device).bool()
        sim_pp = sim_pp.masked_fill(mask_self, -1e9)

        if Nn > 0:
            # positive-negativesimilarity [Np, Nn]
            sim_pn = torch.mm(Z_pos, Z_neg.t()) / temperature

            # formula (5): based onsimilarity's negative sampleweight w_n = softmax(β * s_bn)
            w_neg = F.softmax(beta * sim_pn, dim=-1)  # [Np, Nn]

            # weightednegative sample log-sum-exp: |N(b)| * Σ w_n * exp(s_bn)
            weighted_neg = (Nn * w_neg * torch.exp(sim_pn)).sum(dim=-1)  # [Np]
        else:
            weighted_neg = torch.zeros(Np, device=Z_pos.device)

        # formula (6): toeachanchor b, traverseitspositive sample
        # P(b) = hasotherbenign sample (divideself) 
        # L = -1/|P(b)| * Σ_{j∈P(b)} log [ exp(s_bj) / (Σ_{j'∈P(b)} exp(s_bj') + |N(b)|·Σ w_n·exp(s_bn)) ]
        exp_pp = torch.exp(sim_pp)  # [Np, Np], toalready mask is ~0
        denom = exp_pp.sum(dim=-1) + weighted_neg  # [Np]

        log_prob = sim_pp - torch.log(denom.unsqueeze(1).clamp_min(1e-9))  # [Np, Np]
        pos_mask = (~mask_self).float()
        n_pos = pos_mask.sum(dim=-1).clamp_min(1.0)  # [Np]
        loss = -(pos_mask * log_prob).sum(dim=-1) / n_pos  # [Np]

        return loss.mean()

    def generate_node_embeddings(self, use_temporal: bool = False):
        """generatenodeembedding (1implementation, use_temporal 1) . 
        - use_temporal=False: only encoder (static) 
        - use_temporal=True: settimememoryafter, timeorder fetch→encoder(return_all=True)→temporal→commit () 
        resultwrite self.snapshot_node_embeddings
        """
        self.encoder.eval()
        if use_temporal:
            self.temporal.reset()
        self.snapshot_node_embeddings.clear()
        with torch.no_grad():
            for g in self.snapshots:
                if g is None or g.vcount() == 0:
                    self.snapshot_node_embeddings.append({})
                    continue
                x_np = self._build_node_features(g)
                eidx, ef = self._igraph_edges_to_edge_index(g)
                x = torch.from_numpy(x_np).to(self.device)
                curr_ids = [g.vs[i]['name'] for i in range(g.vcount())]
                if use_temporal:
                    H_prev = self.temporal.fetch(curr_ids, device=self.device)
                    Z_list = self.encoder(x, eidx, edge_feat=ef, return_all=True)
                    H_list = self.temporal(Z_list, H_prev)
                    self.temporal.commit(curr_ids, [h.detach() for h in H_list])
                    h_last = H_list[-1]
                else:
                    h_last = self.encoder(x, eidx, edge_feat=ef)
                emb_dict: Dict[str, np.ndarray] = {}
                for i in range(g.vcount()):
                    nid = g.vs[i]['name']
                    emb_dict[nid] = h_last[i].detach().cpu().numpy().astype(np.float32)
                self.snapshot_node_embeddings.append(emb_dict)
        mode = 'temporal' if use_temporal else 'static'
        print(f"[GCC-Dev] Generated {mode} node embeddings: {len(self.snapshot_node_embeddings)} snapshots")
