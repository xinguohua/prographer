import time
import math
import random
import mmh3
from collections import defaultdict
import numpy as np

from process.embedders.base import GraphEmbedderBase


# --- WL histogram + decay (流式 / 全局直方图) ---
class WLHistogram:
    def __init__(self, R=3, decay_lambda=0.0005):
        self.R = R
        self.decay_lambda = decay_lambda
        self.hist = defaultdict(float)     # global histogram: label -> weight
        self.labels = {}                   # node -> current label (string)
        # 去掉 lambda：使用 defaultdict(dict) + 显式 setdefault
        self.adj = defaultdict(dict)       # node -> edge_type -> set(neighbors)
        self.last_decay_ts = time.time()

    def ingest_edges(self, edges, types, node_gids, node_labels=None):
        """
        批量 ingest 多条边
        edges: list/iter of (u_local, v_local)
        types: list/iter of edge types (len 对齐 edges)
        node_gids: dict {local_vid -> global_id}
        node_labels: dict {local_vid -> init_label} (可选)
        """
        # 批量设置初始标签
        if node_labels is not None:
            for vid_local, label in node_labels.items():
                gid = node_gids[vid_local]
                self.labels.setdefault(gid, label)

        # 批量建邻接
        for i, (u_local, v_local) in enumerate(edges):
            et = types[i]
            u = node_gids[u_local]
            v = node_gids[v_local]

            et_map_u = self.adj[u]
            if et not in et_map_u:
                et_map_u[et] = set()
            et_map_u[et].add(v)

            rev_type = f"rev_{et}"
            et_map_v = self.adj[v]
            if rev_type not in et_map_v:
                et_map_v[rev_type] = set()
            et_map_v[rev_type].add(u)

    def _decay(self):
        now = time.time()
        dt = now - self.last_decay_ts
        if dt <= 0:
            return
        factor = math.exp(-self.decay_lambda * dt)
        keys = list(self.hist.keys())
        for k in keys:
            self.hist[k] *= factor
            if self.hist[k] < 1e-12:
                del self.hist[k]
        self.last_decay_ts = now

    def run_wl_rounds(self):
        """优化版 WL：用整数哈希替代排序+字符串拼接"""
        self._decay()
        cur = self.labels.copy()

        for _ in range(self.R):
            nxt = {}
            for n, lab in cur.items():
                et_map = self.adj.get(n, {})
                for et, nbrs in et_map.items():
                    neigh_hash = 0
                    for x in nbrs:
                        lbl = cur.get(x, "")
                        # 避免字符串拼接: 直接对 (etype, label) 哈希
                        hv = mmh3.hash64(f"{et}:{lbl}")[0]
                        neigh_hash ^= hv  # XOR 聚合，顺序无关

                    # combine 自己的标签哈希 + 邻居聚合哈希
                    self_hash = mmh3.hash64(lab)[0]
                    sig_val = self_hash ^ neigh_hash
                    new_lab = hex(sig_val & ((1 << 64) - 1))

                    nxt[n] = new_lab
                    self.hist[new_lab] += 1.0
            cur = nxt

        self.labels.update(cur)


# --- 极简 HistoSketch 占位实现（用来把直方图压成固定长度 sketch） ---
class HistoSketch:
    def __init__(self, sketch_size=64, seed=42):
        self.K = sketch_size
        random.seed(seed)
        # 用随机参数作为占位（真实 CWS 实现应替换这里）
        self.a = [random.random() + 1e-6 for _ in range(self.K)]
        self.b = [random.random() + 1e-6 for _ in range(self.K)]

    def _cws_hash(self, key, weight, k):
        # 占位：返回 (hash_key, score) 其中 score 越小越优
        # 真正的 Consistent Weighted Sampling 需要精确实现
        h = mmh3.hash64(f"{key}:{k}")[0] & ((1 << 64) - 1)
        # 将 weight -> score 做个单调转换（越大越优 => 越小score）
        score = -math.log(max(weight, 1e-9)) / (self.a[k])
        return (h, score)

    def sketch(self, histogram):
        sig = [(None, float('inf'))] * self.K
        for key, w in histogram.items():
            if w <= 0:
                continue
            for k in range(self.K):
                cand = self._cws_hash(key, w, k)
                if cand[1] < sig[k][1]:
                    sig[k] = cand
        # 返回固定长度整数向量（hash keys）
        return [int(x[0]) if x[0] is not None else 0 for x in sig]


# --- 具体的子类：把 UNICORN 风格嵌入实现进来 ---
class UnicornGraphEmbedder(GraphEmbedderBase):
    def __init__(self, snapshots, features=None, mapp=None,
                 R=3, decay_lambda=0.0005, sketch_size=64,
                 snapshot_edges=2000, wl_batch=500, embedding_dim=32):
        super().__init__(snapshots, features, mapp)
        # 内部模块
        self.snapshots = self.G
        self.wl = WLHistogram(R=R, decay_lambda=decay_lambda)
        self.hs = HistoSketch(sketch_size=sketch_size)
        self.snapshot_edges = snapshot_edges
        self.wl_batch = wl_batch
        # 目标输出维度（对 sketch 序列做降维后得到的 snapshot embeddings）
        self.embedding_dim = embedding_dim

        # bookkeeping
        self.sketch_snapshots = []     # list of (ts, sketch)
        self.snapshot_embeddings = None  # numpy array (n_snapshots, embedding_dim)
        # node-level cache for last computed node embeddings
        self._node_embeddings = {}
        self._edge_embeddings = {}


    def train(self):
        """
        基于 self.snapshots（每个元素是 igraph.Graph）生成快照嵌入。
        每个快照处理完：运行 WL -> 将全局直方图压缩成 sketch -> 存入 self.sketch_snapshots
        """
        if not hasattr(self, "snapshots") or not self.snapshots:
            raise RuntimeError("self.snapshots 为空，请先设置快照数据")

        for sidx, g in enumerate(self.snapshots):
            if g is None:
                continue

            # 批量取数据（一次性从 C 层拿出，避免逐元素 Python 属性访问）
            edges = g.get_edgelist()          # [(u,v), ...]
            types = g.es["type"]              # 边类型数组（len == |E|）
            props = g.vs["properties"]        # 节点属性数组（len == |V|）

            # 建立本快照的 {local_vid -> global_id}
            vcount = g.vcount()
            # 建立 {local_vid -> name}
            node_gids = {vid: g.vs[vid]['name'] for vid in range(g.vcount())}
            # 建立 {local_vid -> 属性} (假设属性存在；可按需判空)
            node_labels = {vid: props[vid] for vid in range(vcount)}
            print("ingest_edges")
            # 一次性送进 WL
            self.wl.ingest_edges(edges, types, node_gids, node_labels=node_labels)

            # WL + Sketch
            try:
                print("run_wl_rounds")
                self.wl.run_wl_rounds()
            except Exception:
                # 为稳健起见，忽略 WL 内部可能的个别异常，不影响后续流程
                pass

            sketch = self.hs.sketch(self.wl.hist)
            self.sketch_snapshots.append((time.time(), sketch))

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        """
        从 self.sketch_snapshots 里取出快照 embedding。
        每个快照是 (timestamp, sketch)，这里只取 sketch。
        """
        if not self.sketch_snapshots:
            raise RuntimeError("还没有任何快照，请先调用 train() 生成。")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.sketch_snapshots)))

        embeddings = []
        for idx in snapshot_sequence:
            ts, sketch = self.sketch_snapshots[idx]
            embeddings.append(sketch)

        return np.array(embeddings)

    def embed_edges(self):
        pass

    def embed_nodes(self):
        pass