import os
import re
import time
import igraph as ig
import orjson
import pandas as pd
from .base import BaseProcessor
from .common import collect_json_paths, collect_label_paths
from .common import merge_properties, add_node_properties
from .type_enum import ObjectType


class DARPAHandler(BaseProcessor):
    def __init__(self, base_path=None, train=True):
        """保持原有初始化，只添加快照相关变量"""
        super().__init__(base_path, train)
        # 新增：用于快照生成的变量
        self.snapshots = []
        self.cache_graph = ig.Graph(directed=True)
        self.node_timestamps = {}
        self.first_flag = True
        # 新增：场景映射
        self.scene_names_in_order = []
        self.all_dfs_map = {}

    def load(self):
        """保持您原有的 load 逻辑，只修改编码问题"""
        json_map = collect_json_paths(self.base_path)
        label_map = collect_label_paths(self.base_path)
        for scene, category_data in json_map.items():
            # TODO: for test - 改为 theia33
            if scene != "theia33":
                continue

            # 记录场景名称
            if scene not in self.scene_names_in_order:
                self.scene_names_in_order.append(scene)

            if self.train == False:
                if scene in label_map:
                    label_file = open(label_map[scene])
                    print(f"正在处理: 场景={scene}, label={label_map[scene]}")
                    self.all_labels.extend([
                        line.strip() for line in label_file.read().splitlines() if line.strip()
                    ])

            scene_dfs = []  # 收集当前场景的所有数据

            for category, json_files in category_data.items():
                #  训练只处理良性类别
                if self.train and category != "benign":
                    continue
                #  测试只处理恶意类别
                if self.train != True and category == "benign":
                    continue

                print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
                scene_category = f"/{scene}_{category}.txt"
                f = open(self.base_path + scene_category)
                self.total_loaded_bytes += os.path.getsize(self.base_path + scene_category)
                # 训练分隔
                data = f.read().split('\n')
                # TODO:
                data = [line.split('\t') for line in data]
                # for test
                # data = [line.split('\t') for line in data[:10000]]
                df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
                df = df.dropna()
                df.sort_values(by='timestamp', ascending=True, inplace=True)

                # 形成一个更完整的视图
                netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)
                print("==========collect_edges_from_log=======start")
                t0 = time.time()
                df = collect_edges_from_log(df, json_files)
                t1 = time.time()
                print("==========collect_edges_from_log=======end")
                print(f"耗时: {t1 - t0:.2f} 秒")

                if self.train:
                    # 只取良性前80%训练
                    num_rows = int(len(df) * 0.9)
                    df = df.iloc[:num_rows]
                    scene_dfs.append(df)
                else:
                    # 数据选择逻辑
                    if category == "benign":
                        # 取后10%
                        num_rows = int(len(df) * 0.9)
                        df = df.iloc[num_rows:]
                        scene_dfs.append(df)
                    elif category == "malicious":
                        # 使用全部
                        scene_dfs.append(df)
                        pass
                    else:
                        continue
                merge_properties(netobj2pro, self.all_netobj2pro)
                merge_properties(subject2pro, self.all_subject2pro)
                merge_properties(file2pro, self.all_file2pro)

            # 合并当前场景的所有数据
            if scene_dfs:
                scene_combined_df = pd.concat(scene_dfs, ignore_index=True)
                self.all_dfs_map[scene] = scene_combined_df
                self.all_dfs.append(scene_combined_df)  # 保持原有逻辑

        # 训练用的数据集
        use_df = pd.concat(self.all_dfs, ignore_index=True)
        self.use_df = use_df.drop_duplicates()

    def build_graph(self):
        """保持原有的图构建逻辑，同时生成快照"""
        use_df = self.use_df
        all_labels = set(self.all_labels)

        _otype_cache = {}

        def _otype(v):
            if v not in _otype_cache:
                _otype_cache[v] = ObjectType[v].value
            return _otype_cache[v]

        nodes_props, nodes_type, edges_map = {}, {}, {}

        # === 新增：快照生成相关变量 ===
        snapshot_size = 300
        forgetting_rate = 0.3
        self.cache_graph = ig.Graph(directed=True)
        self.node_timestamps = {}
        self.first_flag = True
        self.snapshots = []

        # === 扫描 DataFrame 收集节点与边 ===
        for r in use_df.itertuples(index=False):
            action = getattr(r, "action")
            actor_id = getattr(r, "actorID")
            object_id = getattr(r, "objectID")

            # 新增：获取时间戳用于快照生成
            timestamp = getattr(r, "timestamp", 0)
            try:
                timestamp = int(timestamp) if timestamp != '' else 0
            except (ValueError, TypeError):
                timestamp = 0

            # actor 节点
            props_actor = extract_properties(actor_id, r, action,
                                             self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, actor_id, props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = getattr(r, "actor_type")

            # object 节点
            props_obj = extract_properties(object_id, r, action,
                                           self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, object_id, props_obj)
            if object_id not in nodes_type:
                nodes_type[object_id] = getattr(r, "object")

            # 累加动作到 set
            edges_map.setdefault((actor_id, object_id), set()).add(action)

            # === 新增：快照生成逻辑 ===
            # 添加节点到缓存图
            try:
                self.cache_graph.vs.find(name=actor_id)
            except ValueError:
                actor_type_enum = ObjectType[getattr(r, "actor_type")]
                self.cache_graph.add_vertex(name=actor_id, type=actor_type_enum.value,
                                            type_name=actor_type_enum.name, properties=props_actor)
            self.node_timestamps[actor_id] = timestamp

            try:
                self.cache_graph.vs.find(name=object_id)
            except ValueError:
                object_type_enum = ObjectType[getattr(r, "object")]
                self.cache_graph.add_vertex(name=object_id, type=object_type_enum.value,
                                            type_name=object_type_enum.name, properties=props_obj)
            self.node_timestamps[object_id] = timestamp

            # 添加边
            actor_idx = self.cache_graph.vs.find(name=actor_id).index
            object_idx = self.cache_graph.vs.find(name=object_id).index
            if not self.cache_graph.are_connected(actor_idx, object_idx):
                self.cache_graph.add_edge(actor_idx, object_idx, actions=action, timestamp=timestamp)

            # 快照生成
            n_nodes = len(self.cache_graph.vs)
            if self.first_flag and n_nodes >= snapshot_size:
                self._generate_snapshot("theia33")
                self.first_flag = False
            elif not self.first_flag and n_nodes >= snapshot_size * (1 + forgetting_rate):
                self._retire_old_nodes(snapshot_size, forgetting_rate)
                self._generate_snapshot("theia33")

        # 处理剩余节点
        if len(self.cache_graph.vs) > 0:
            self._generate_snapshot("theia33")

        # === 创建图节点（保持原有逻辑）===
        node_ids = list(nodes_props.keys())
        index_map = {nid: i for i, nid in enumerate(node_ids)}

        G = ig.Graph(directed=True)
        G.add_vertices(len(node_ids))
        G.vs["name"] = node_ids
        G.vs["type"] = [nodes_type.get(nid) for nid in node_ids]
        G.vs["properties"] = [nodes_props[nid] for nid in node_ids]
        G.vs["label"] = [1 if nid in all_labels else 0 for nid in node_ids]

        # === 创建图边（保持原有逻辑）===
        unique_edges = list(edges_map.keys())
        if unique_edges:
            edge_idx = [(index_map[a], index_map[b]) for (a, b) in unique_edges]
            G.add_edges(edge_idx)
            G.es["action"] = [list(edges_map[(a, b)]) for (a, b) in unique_edges]

        # === 下游需要的结构（保持原有逻辑）===
        features = [nodes_props[nid] for nid in node_ids]
        edge_index = [[], []]
        relations_index = {}
        for a, b in unique_edges:
            s, d = index_map[a], index_map[b]
            edge_index[0].append(s)
            edge_index[1].append(d)
            relations_index[(s, d)] = list(edges_map[(a, b)])

        # 为快照打标签
        for snapshot in self.snapshots:
            for v in snapshot.vs:
                v["label"] = int(v["name"] in all_labels)

        # 返回原有的5个值，第5个值改为快照列表，后面添加兼容性参数
        return features, edge_index, node_ids, relations_index, self.snapshots, [], []

    def _retire_old_nodes(self, snapshot_size: int, forgetting_rate: float) -> None:
        """移除老旧节点的函数"""
        n_nodes_to_remove = int(snapshot_size * forgetting_rate)
        if n_nodes_to_remove <= 0:
            return
        sorted_nodes = sorted(self.node_timestamps.items(), key=lambda item: item[1])
        nodes_to_remove = [node_id for node_id, _ in sorted_nodes[:n_nodes_to_remove]]
        try:
            indices_to_remove = [self.cache_graph.vs.find(name=name).index for name in nodes_to_remove]
            self.cache_graph.delete_vertices(indices_to_remove)
        except ValueError:
            pass
        for node_id in nodes_to_remove:
            if node_id in self.node_timestamps:
                del self.node_timestamps[node_id]

    def _generate_snapshot(self, scene_name) -> None:
        """生成快照"""
        snapshot = self.cache_graph.copy()
        self.snapshots.append(snapshot)

# 其他函数保持不变...
def collect_nodes_from_log(paths):
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    for p in paths:
        # 修复编码问题
        with open(p, encoding='utf-8', errors='ignore') as f:
            for line in f:
                # --- NetFlowObject ---
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line
                        )[0]
                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]
                        nodeproperty = f"{srcaddr},{srcport},{dstaddr},{dstport}"
                        netobj2pro[nodeid] = nodeproperty
                    except:
                        pass

                # --- Subject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                    try:
                        res = re.findall(
                            'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        cmdLine = res[2]
                        tgid = res[4]
                        try:
                            path_str = re.findall('"path":"(.*?)"', line)[0]
                            path = path_str
                        except:
                            path = "null"
                        nodeProperty = f"{cmdLine},{tgid},{path}"
                        subject2pro[nodeid] = nodeProperty
                    except:
                        pass

                # --- FileObject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                    try:
                        res = re.findall(
                            'FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        filepath = res[2]
                        nodeproperty = filepath
                        file2pro[nodeid] = nodeproperty
                    except:
                        pass

    return netobj2pro, subject2pro, file2pro

def collect_edges_from_log(d, paths):
    info = []
    for p in paths:
        with open(p, "rb") as f:
            for line in f:
                if b"EVENT" not in line:
                    continue
                try:
                    x = orjson.loads(line)
                except Exception:
                    continue

                try:
                    ev = x["datum"]["com.bbn.tc.schema.avro.cdm18.Event"]
                except Exception:
                    continue

                action = ev.get("type", "")
                actor = (ev.get("subject") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID", "")
                obj = (ev.get("predicateObject") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID", "")
                timestamp = ev.get("timestampNanos", "")
                cmd = ((ev.get("properties") or {}).get("map") or {}).get("cmdLine", "")
                path = (ev.get("predicateObjectPath") or {}).get("string", "")
                path2 = (ev.get("predicateObject2Path") or {}).get("string", "")

                obj2 = (ev.get("predicateObject2") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID")
                if obj2:
                    info.append({
                        "actorID": actor, "objectID": obj2, "action": action,
                        "timestamp": timestamp, "exec": cmd, "path": path2
                    })

                info.append({
                    "actorID": actor, "objectID": obj, "action": action,
                    "timestamp": timestamp, "exec": cmd, "path": path
                })

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how="inner",
                   on=["actorID", "objectID", "action", "timestamp"]) \
        .drop_duplicates()


def extract_properties(node_id, row, action, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        exec_cmd = getattr(row, "exec", "")
        path_val = getattr(row, "path", "")
        return " ".join([exec_cmd, action] + ([path_val] if path_val else []))


