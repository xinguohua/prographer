# embedders/prographer.py
import math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .base import GraphEmbedderBase


# =========================================================================
# =================== 修改后的 ProGrapherEmbedder 类 =======================
# =========================================================================

class ProGrapherEmbedder(GraphEmbedderBase):
    # 新增：定义一个类级别的常量来存储模型保存路径
    # 这样修改路径时只需要改这一个地方。
    MODEL_SAVE_PATH = 'prographer_encoder.pth'

    def __init__(self, snapshot_sequence,
                 # --- Encoder (Graph2Vec) Parameters from Paper ---
                 embedding_dim=256,
                 wl_depth=2,
                 neg_samples=15,
                 # --- Training Hyperparameters ---
                 learning_rate=1e-3,
                 epochs=3,
                 weight_decay=1e-5,
                 # --- 新增：序列长度参数 ---
                 sequence_length=12
                 ):

        super().__init__(snapshot_sequence,features=None,mapp=None)
        self.snapshot_sequence = self.G
        self.embedding_dim = embedding_dim
        self.wl_depth = wl_depth
        self.neg_samples = neg_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.sequence_length = sequence_length  # 新增：保存序列长度
        self.rsg_vocab = {}
        self.snapshot_embeddings_layer = None
        self.rsg_embeddings_layer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- ProGrapherEmbedder will use device: {self.device} ---")
        print(f"--- ProGrapherEmbedder sequence length: {self.sequence_length} ---")

    @staticmethod
    def _get_neighbor_info(graph, edge, node_idx):
        if edge.source == node_idx:
            return edge.target
        return edge.source

    @staticmethod
    def generate_rsg(graph, node_idx, depth):
        if depth == 0:
            # print(f"name: {graph.vs[node_idx]['name']} properties: {graph.vs[node_idx]['properties']}")
            return str(graph.vs[node_idx]['name'] )
        prev_rsg = ProGrapherEmbedder.generate_rsg(graph, node_idx, depth - 1)
        incident_edges = graph.es[graph.incident(node_idx, mode="all")]
        if not incident_edges:
            return prev_rsg
        neighbor_info_parts = []
        for edge in incident_edges:
            try:
                edge_type = str(edge['actions'])
            except (KeyError, TypeError):
                edge_type = "UNKNOWN"
            neighbor_idx = ProGrapherEmbedder._get_neighbor_info(graph, edge, node_idx)
            neighbor_rsg = ProGrapherEmbedder.generate_rsg(graph, neighbor_idx, depth - 1)
            neighbor_info_parts.append(f"{edge_type}:{neighbor_rsg}")
        sorted_neighbor_info = sorted(neighbor_info_parts)
        return f"{prev_rsg}-({'_'.join(sorted_neighbor_info)})"

    def _build_vocabulary(self):
        print("Building RSG vocabulary from all snapshots...")
        all_rsgs = set()
        for snapshot in tqdm(self.snapshot_sequence, desc="Processing Snapshots for Vocab"):
            for v_idx in range(len(snapshot.vs)):
                for d in range(self.wl_depth + 1):
                    all_rsgs.add(ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d))
        self.rsg_vocab = {rsg: i for i, rsg in enumerate(sorted(list(all_rsgs)))}
        print(f"Vocabulary built with {len(self.rsg_vocab)} unique RSGs.")
        save_path = "rsg_vocab.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            for rsg, idx in self.rsg_vocab.items():
                f.write(f"{idx}\t{rsg}\n")
        print(f"[INFO] RSG vocabulary saved to {save_path}")

    def train(self):
        # 训练 Graph2Vec 模型，并在训练结束后自动保存模型。

        print("--- Training ProGrapher Encoder (Graph2Vec with frequency weighting) ---")
        if not self.snapshot_sequence:
            print("Warning: No snapshots to train on.")
            return

        self._build_vocabulary()
        num_snapshots = len(self.snapshot_sequence)
        num_rsgs = len(self.rsg_vocab)

        if num_rsgs == 0:
            print("Warning: RSG vocabulary is empty. Cannot train.")
            return

        # Embedding 表
        self.snapshot_embeddings_layer = nn.Embedding(num_snapshots, self.embedding_dim).to(self.device)
        self.rsg_embeddings_layer = nn.Embedding(num_rsgs, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.snapshot_embeddings_layer.weight)
        nn.init.xavier_uniform_(self.rsg_embeddings_layer.weight)

        # 注意：reduction="none"，方便单独加权
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        optimizer = optim.Adam(
            list(self.snapshot_embeddings_layer.parameters()) + list(self.rsg_embeddings_layer.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_updates = 0
            shuffled_indices = np.random.permutation(num_snapshots)

            for snapshot_idx in tqdm(shuffled_indices, desc=f"Encoder Epoch {epoch + 1}/{self.epochs}", leave=False):
                snapshot = self.snapshot_sequence[snapshot_idx]

                # ----------------- 正样本统计 (Counter) -----------------
                rsg_counts = Counter()
                for v_idx in range(len(snapshot.vs)):
                    node_freq = snapshot.vs[v_idx]["frequency"]  # 节点频次
                    for d in range(self.wl_depth + 1):
                        rsg = ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d)
                        if rsg in self.rsg_vocab:
                            rsg_counts[self.rsg_vocab[rsg]] += node_freq
                if not rsg_counts:
                    continue
                # ---------------------------------------------------------

                # ------------------ 遍历每个正样本 -------------------
                for rsg_id, freq in rsg_counts.items():
                    # 负采样
                    neg_sample_ids = []
                    while len(neg_sample_ids) < self.neg_samples:
                        sample = np.random.randint(0, num_rsgs)
                        if sample != rsg_id and sample not in rsg_counts:  # 排除正样本
                            neg_sample_ids.append(sample)

                    target_ids = torch.LongTensor([rsg_id] + neg_sample_ids).to(self.device)
                    labels = torch.FloatTensor([1.0] + [0.0] * self.neg_samples).to(self.device)
                    snapshot_id_tensor = torch.LongTensor([snapshot_idx]).to(self.device)

                    snapshot_vec = self.snapshot_embeddings_layer(snapshot_id_tensor)  # [1, dim]
                    rsg_vecs = self.rsg_embeddings_layer(target_ids)  # [1+neg, dim]
                    logits = torch.sum(snapshot_vec * rsg_vecs, dim=1)  # [1+neg]

                    # -------- loss：只对正样本乘以权重 --------
                    raw_loss = criterion(logits, labels)  # [1+neg]
                    weights = torch.ones_like(labels)
                    weights[0] = math.log(1.0 + freq)  # 正样本加权，负样本权重=1
                    loss = (raw_loss * weights).mean()
                    # ---------------------------------------

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_updates += 1
                # ---------------------------------------------------------

            avg_loss = total_loss / num_updates if num_updates > 0 else 0
            print(f"Encoder Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.6f}")

        print("\nEncoder training complete.")

        self.save_model(self.MODEL_SAVE_PATH)


    def save_model(self, path):
        """
        保存编码器的状态。
        """
        if self.snapshot_embeddings_layer is None or self.rsg_embeddings_layer is None:
            raise RuntimeError("模型尚未训练，无法保存。")
        print(f"Saving encoder model to {path}...")
        state = {
            'params': {
                'embedding_dim': self.embedding_dim, 'wl_depth': self.wl_depth,
                'neg_samples': self.neg_samples, 'learning_rate': self.learning_rate,
                'epochs': self.epochs, 'weight_decay': self.weight_decay
            },
            'snapshot_embeddings_state_dict': self.snapshot_embeddings_layer.state_dict(),
            'rsg_embeddings_state_dict': self.rsg_embeddings_layer.state_dict(),
            'rsg_vocab': self.rsg_vocab,
            'num_snapshots': len(self.snapshot_sequence)
        }
        torch.save(state, path)
        print("Encoder model saved successfully.")

    @classmethod
    def load(cls, path, snapshot_sequence):
        """
        从文件加载预训练的编码器模型。
        【修改】适应测试时不同的快照数量
        """
        print(f"Loading encoder model from {path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)

        instance = cls(snapshot_sequence, **state['params'])

        # 【修改】使用当前快照序列的长度，而不是训练时保存的长度
        current_num_snapshots = len(snapshot_sequence)
        num_rsgs = len(state['rsg_vocab'])
        original_num_snapshots = state['num_snapshots']
        
        print(f"Original training snapshots: {original_num_snapshots}")
        print(f"Current test snapshots: {current_num_snapshots}")

        # 【关键修改】创建新的快照嵌入层以适应当前快照数量
        instance.snapshot_embeddings_layer = nn.Embedding(current_num_snapshots, instance.embedding_dim)
        instance.rsg_embeddings_layer = nn.Embedding(num_rsgs, instance.embedding_dim)

        # 【修改】只加载RSG嵌入的权重，快照嵌入重新初始化
        # RSG嵌入是通用的，可以直接加载
        instance.rsg_embeddings_layer.load_state_dict(state['rsg_embeddings_state_dict'])
        
        # 【新增】快照嵌入需要重新初始化，因为数量可能不同
        if current_num_snapshots == original_num_snapshots:
            # 如果快照数量相同，直接加载
            instance.snapshot_embeddings_layer.load_state_dict(state['snapshot_embeddings_state_dict'])
            print("Loaded original snapshot embeddings (same snapshot count)")
        else:
            # 如果快照数量不同，重新初始化快照嵌入
            nn.init.xavier_uniform_(instance.snapshot_embeddings_layer.weight)
            print(f"Reinitialized snapshot embeddings for {current_num_snapshots} snapshots")
        
        instance.rsg_vocab = state['rsg_vocab']

        instance.snapshot_embeddings_layer.to(instance.device)
        instance.rsg_embeddings_layer.to(instance.device)
        instance.snapshot_embeddings_layer.eval()
        instance.rsg_embeddings_layer.eval()

        print("Encoder model loaded successfully.")
        return instance

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        """
        获取给定快照序列的嵌入向量。
        如果不传 snapshot_sequence，就默认用初始化时的 self.snapshot_sequence。
        """
        if self.snapshot_embeddings_layer is None:
            raise RuntimeError("模型尚未训练或加载，无法获取快照嵌入。")

        # 如果传入新的快照序列，用它来决定要生成哪些 index
        if snapshot_sequence is None:
            snapshot_sequence = self.snapshot_sequence

        num_snapshots = len(snapshot_sequence)

        # 构造 snapshot indices [0, 1, 2, ..., num_snapshots-1]
        indices = torch.arange(num_snapshots, device=self.device)

        # 直接通过 embedding layer 获取向量
        with torch.no_grad():
            embeddings = self.snapshot_embeddings_layer(indices)

        # 返回到 CPU 并转成 numpy 方便后续使用
        return embeddings.cpu().numpy()

    def get_rsg_embeddings(self):
        print("Retrieving all RSG embeddings and vocabulary...")
        if self.rsg_embeddings_layer is None:
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        rsg_embeddings = self.rsg_embeddings_layer.weight.detach().cpu().numpy()
        return rsg_embeddings, self.rsg_vocab

    # ... embed_nodes 和 embed_edges 方法保持不变 ...
    def embed_nodes(self):
        pass

    def embed_edges(self):
        pass