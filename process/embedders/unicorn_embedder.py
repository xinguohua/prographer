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
        self.adj = defaultdict(lambda: defaultdict(set))  # node -> edge_type -> set(neighbors)
        self.last_decay_ts = time.time()

    def ingest_edge(self, u, v, edge_type, u_init_label=None, v_init_label=None):
        # 维护邻接结构与初始标签
        if u_init_label is not None:
            self.labels.setdefault(u, u_init_label)
        if v_init_label is not None:
            self.labels.setdefault(v, v_init_label)
        self.adj[u][edge_type].add(v)
        self.adj[v][f"rev_{edge_type}"].add(u)

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
        """对当前标签做 R 次 WL 迭代，将产生的标签计入全局直方图"""
        self._decay()
        cur = self.labels.copy()
        for _ in range(self.R):
            nxt = {}
            for n, lab in cur.items():
                for et, nbrs in self.adj[n].items():
                    neigh_sig = sorted((et, cur.get(x, "")) for x in nbrs)
                    # stable string repr
                    neigh_repr = "|".join(f"{et}:{lbl}" for et, lbl in ((et, lbl) for et, lbl in neigh_sig))
                    # 哈希将邻域压缩
                    sig = f"{lab}|{neigh_repr}"
                    new_lab = hex(mmh3.hash64(sig)[0] & ((1<<64)-1))
                    nxt[n] = new_lab
                    self.hist[new_lab] += 1.0
            cur = nxt
        # 更新全局 node label
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
        h = mmh3.hash64(f"{key}:{k}")[0] & ((1<<64)-1)
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
        self.edge_count = 0
        self.sketch_snapshots = []     # list of (ts, sketch)
        self.snapshot_embeddings = None  # numpy array (n_snapshots, embedding_dim)
        # node-level cache for last computed node embeddings
        self._node_embeddings = {}
        self._edge_embeddings = {}

        # # 如果 G 已包含边，先 ingest 一遍（可选）
        # self._ingest_graph(G)

    def _ingest_graph(self, G):
        """尝试从 G（networkx-like）把边/节点 ingest 进 wl.adj"""
        if G is None:
            return
        # accept either networkx graph or adjacency dict: prefer (u,v,edata) iteration if possible
        try:
            for u, v, ed in G.edges(data=True):
                et = ed.get('etype', ed.get('type', 'unk'))
                u_lab = G.nodes[u].get('init_label') if hasattr(G, 'nodes') else None
                v_lab = G.nodes[v].get('init_label') if hasattr(G, 'nodes') else None
                self.wl.ingest_edge(u, v, et, u_init_label=u_lab, v_init_label=v_lab)
                self.edge_count += 1
        except Exception:
            # fallback: if G is adjacency dict
            if isinstance(G, dict):
                for u, nbrinfo in G.items():
                    for et, nbrs in nbrinfo.items():
                        for v in nbrs:
                            self.wl.ingest_edge(u, v, et)
                            self.edge_count += 1

    def _node_gid(self, g, vid, sidx):
        """
        把 igraph 顶点索引 -> 全局唯一节点ID
        优先顺序: vertex['id'] -> vertex['name'] -> f"s{sidx}:v{vid}"
        """
        attrs = g.vs.attributes()
        if 'id' in attrs and g.vs[vid]['id'] is not None:
            return g.vs[vid]['id']
        if 'name' in attrs and g.vs[vid]['name'] is not None:
            return g.vs[vid]['name']
        return f"s{sidx}:v{vid}"

    def train(self, max_steps=None):
        """
        基于 self.snapshots（每个元素是 igraph.Graph）生成快照嵌入。
        每个快照处理完：运行 WL -> 将全局直方图压缩成 sketch -> 存入 self.sketch_snapshots
        """
        if not hasattr(self, "snapshots") or not self.snapshots:
            raise RuntimeError("self.snapshots 为空，请先设置快照数据")

        for sidx, g in enumerate(self.snapshots):
            if g is None:
                continue

            # 遍历当前图的所有边
            for e in g.es:
                u_local, v_local = e.tuple
                u = self._node_gid(g, u_local, sidx)
                v = self._node_gid(g, v_local, sidx)

                et = e['type']
                u_init = g.vs[u_local]['properties']
                v_init = g.vs[v_local]['properties']

                # ingest edge 到 WL 模块
                self.wl.ingest_edge(u, v, et, u_init_label=u_init, v_init_label=v_init)

                self.edge_count += 1

            # 当前 snapshot 处理完后 -> 做一次 WL + 生成 sketch
            try:
                self.wl.run_wl_rounds()
            except Exception:
                pass
            sketch = self.hs.sketch(self.wl.hist)
            self.sketch_snapshots.append((time.time(), sketch))

        # 训练结束后，尝试降维为 snapshot embeddings
        if self.sketch_snapshots:
            try:
                self.snapshot_embeddings = self._compute_snapshot_embeddings(
                    n_components=self.embedding_dim
                )
            except Exception:
                self.snapshot_embeddings = None


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

    def _compute_snapshot_embeddings(self, n_components=32):
        """
        使用简单的线性降维（SVD/PCA）将 sketch 序列转换为低维连续 embedding。
        这里的 sketch 是整数哈希列表（长度 K）。为了把它们转换为数值矩阵，
        我们对每个 sketch 做简单的 one-hot-ish hashing -> 稀疏计数向量转换，
        然后对矩阵进行 SVD 并取左奇异向量乘奇异值得到低维表示。
        这是一个轻量级、可替换的降维策略，便于后续替换为论文中准确的训练流程。
        """
        if not self.sketch_snapshots:
            return np.zeros((0, n_components))

        sketches = [s for _, s in self.sketch_snapshots]
        K = len(sketches[0])

        # 建立 hash -> column id 映射（根据已出现的 hash 值）
        uniq = {}
        cols = 0
        rows = len(sketches)
        # 估计稀疏矩阵大小，先收集所有唯一 hash
        for sk in sketches:
            for h in sk:
                if h not in uniq:
                    uniq[h] = cols
                    cols += 1

        # 若唯一哈希过多，限制列数以免内存暴涨：使用模运算压缩
        max_cols = max(cols, 1)
        if cols > 10000:
            # 压缩列数到 K * 64（经验值），通过对 hash 值取模
            max_cols = K * 64

        mat = np.zeros((rows, max_cols), dtype=float)
        for i, sk in enumerate(sketches):
            for h in sk:
                if cols <= 10000:
                    j = uniq[h]
                else:
                    j = int(h) % max_cols
                mat[i, j] += 1.0

        # 中心化
        mat -= mat.mean(axis=0, keepdims=True)

        # SVD 降维（如果维度过大用 economy SVD）
        try:
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            # 使用 U * S[:, :n_components]
            comps = min(n_components, U.shape[1])
            embeddings = U[:, :comps] * S[:comps]
            # 若需要更多列，用零填充
            if comps < n_components:
                padded = np.zeros((rows, n_components), dtype=float)
                padded[:, :comps] = embeddings
                embeddings = padded
            return embeddings
        except Exception:
            # 退化方案：返回行和/或 sketch 的简单统计特征
            stats = np.vstack([np.mean(mat, axis=1), np.std(mat, axis=1)]).T
            if stats.shape[1] >= n_components:
                return stats[:, :n_components]
            padded = np.zeros((rows, n_components), dtype=float)
            padded[:, :stats.shape[1]] = stats
            return padded

    def embed_edges(self):
        pass

    def embed_nodes(self):
        pass
