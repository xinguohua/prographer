# D:\baseline\process\datahandlers\atlas_handler.py

import os.path
import pandas as pd
import igraph as ig
import re

from .base import BaseProcessor
from .common import merge_properties, collect_dot_paths, collect_atlas_label_paths
from .type_enum import ObjectType




class ATLASHandler(BaseProcessor):
    def __init__(self, base_path=None, train=True, MALICIOUS_INTERVALS_PATH = None, use_time_split=False, test_window_minutes=20):
        """【修改】初始化用于图级别追踪的变量"""
        super().__init__(base_path, train)
        self.all_dfs_map = {}
        # 用于存储图(dot文件)被处理的顺序
        self.graph_names_in_order = []
        # 其他实例变量保持不变
        self.all_labels = []
        # 用于按图（场景）分开存储其对应的恶意标签
        self.graph_to_label = {}
        self.all_netobj2pro = {}
        self.all_subject2pro = {}
        self.all_file2pro = {}
        self.total_loaded_bytes = 0
        self.graph_to_nodes_map = {}
        self.snapshot_to_nodes_map = {}
        self.node_frequency = {}
        # 【新增】时间分割相关参数
        self.use_time_split = use_time_split
        self.test_window_minutes = test_window_minutes

        # 【新增】时间分割功能相关参数
        self.use_time_split = use_time_split
        self.test_window_seconds = test_window_minutes * 60 if use_time_split else 0

    def load(self):
        """加载 ATLAS 数据集。
        - 训练模式：去掉恶意标签部分
        - 测试模式：保留所有数据
        """
        print("处理 ATLAS 数据集...")
        graph_files = collect_dot_paths(self.base_path)  # 所有 .dot 文件路径
        label_map = collect_atlas_label_paths(self.base_path)  # 标签映射

        # 清空缓存
        self.all_dfs_map.clear()
        self.graph_names_in_order.clear()
        self.all_labels.clear()
        self.graph_to_label.clear()

        for dot_file in graph_files:
            dot_name = os.path.splitext(os.path.basename(dot_file))[0]

            # 根据模式选择加载哪个文件
            if self.train and "M1-CVE-2015-5122_windows_h2" not in dot_name:
                continue
            if not self.train and "M1-CVE-2015-5122_windows_h1" not in dot_name:
                continue

            self.total_loaded_bytes += os.path.getsize(dot_file)
            print(f"正在加载场景: {dot_name}")
            self.graph_names_in_order.append(dot_name)

            # 加载标签
            graph_specific_labels = []
            if dot_name in label_map:
                with open(label_map[dot_name], 'r', encoding='utf-8') as label_file:
                    graph_specific_labels = [line.strip() for line in label_file if line.strip()]
                    self.all_labels.extend(graph_specific_labels)
            else:
                if not self.train:
                    print(f"  - 警告: 未找到场景 '{dot_name}' 的标签文件。")

            self.graph_to_label[dot_name] = graph_specific_labels

            # 解析 .dot 文件 -> DataFrame
            netobj2pro, subject2pro, file2pro, dns, ips, conns, sess, webs = collect_nodes_from_log(dot_file)
            self.all_dfs= collect_edges_from_log(dot_file, dns, ips, conns, sess, webs, subject2pro, file2pro)

            if self.train:
                if graph_specific_labels:
                    bad = set(graph_specific_labels)
                    before = len(self.all_dfs)
                    mask_bad = self.all_dfs.isin(bad).any(axis=1)  # 这一行任意列命中恶意label → True
                    self.use_df = self.all_dfs.loc[~mask_bad]  # 过滤掉整行
                    after = len(self.use_df)
                    print(f"  - 训练模式: 从 {before} 条边中过滤恶意标签，保留 {after} 条边")
                else:
                    self.use_df = self.all_dfs
                    print(f"  - 训练模式: 无恶意标签，使用全部数据 {len(self.use_df)} 条边")
            else:
                self.use_df = self.all_dfs
                print(f"  - 测试模式: 使用全部数据 {len(self.use_df)} 条边")

                out_path = f"{dot_name}.csv"
                self.use_df.to_csv(out_path, index=False, encoding="utf-8")
                print(f"[INFO] 已保存 {dot_name} 的 DataFrame 到 {out_path}, 共 {len(self.use_df)} 行")
            # ===============================

            self.all_dfs_map[dot_name] = self.use_df
            merge_properties(netobj2pro, self.all_netobj2pro)
            merge_properties(subject2pro, self.all_subject2pro)
            merge_properties(file2pro, self.all_file2pro)



        # 去重标签
        self.all_labels = list(set(self.all_labels))
        print(f"所有 {len(self.graph_names_in_order)} 个图的数据加载完毕。")

        if not self.train:
            print(f"共找到 {len(self.all_labels)} 个唯一恶意标签用于本次评估。")



    def build_graph(self):
        """ 生成快照，并在节点创建时打标，同时记录所有出现过的节点以供评估。"""
        self.snapshots = []
        self.graph_to_nodes_map = {}
        self.snapshot_to_nodes_map = {}
        self.node_frequency = {}
        current_graph_all_nodes = set()
        snapshot_size = 300
        forgetting_rate = 0.3
        self.cache_graph = ig.Graph(directed=True)
        self.node_timestamps = {}
        self.first_flag = True
        sorted_df = self.use_df.sort_values(by='timestamp') if 'timestamp' in self.use_df.columns else self.use_df

        out_path = "sorted_df.csv"
        sorted_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[INFO] 已保存 sorted_df 的 DataFrame 到 {out_path}, 共 {len(sorted_df)} 行")
        for _, row in sorted_df.iterrows():
            actor_id, object_id = row["actorID"], row["objectID"]
            action, timestamp = row["action"], row.get('timestamp', 0)
            current_graph_all_nodes.add(actor_id)
            current_graph_all_nodes.add(object_id)
            self.node_frequency[actor_id] = self.node_frequency.get(actor_id, 0) + 1
            self.node_frequency[object_id] = self.node_frequency.get(object_id, 0) + 1

            target_node = "192.168.223.3"
            current_snapshot_idx = len(self.snapshots)  # 已经生成的快照数，0-based
            # print(f"[DEBUG] {timestamp}: 节点 {actor_id} -> {object_id} (当前属于 Snapshot {current_snapshot_idx})")

            if target_node in str(actor_id) or target_node in str(object_id):
                print(
                    f"[DEBUG] {timestamp}: 节点 {target_node} 出现在边 {actor_id} -> {object_id} (当前属于 Snapshot {current_snapshot_idx})")
            try:
                v_actor = self.cache_graph.vs.find(name=actor_id)
                v_actor["frequency"] = self.node_frequency[actor_id]
            except ValueError:
                actor_type_enum = ObjectType[row['actor_type']]
                self.cache_graph.add_vertex(
                    name=actor_id, type=actor_type_enum.value, type_name=actor_type_enum.name,
                    properties=extract_properties(actor_id, self.all_netobj2pro, self.all_subject2pro, self.all_file2pro),
                    label=int(actor_id in self.all_labels),
                    frequency = self.node_frequency[actor_id]
                )
            self.node_timestamps[actor_id] = timestamp
            try:
                v_objctor = self.cache_graph.vs.find(name=object_id)
                v_objctor["frequency"] = self.node_frequency[object_id]
            except ValueError:
                object_type_enum = ObjectType[row['object']]
                self.cache_graph.add_vertex(
                    name=object_id, type=object_type_enum.value, type_name=object_type_enum.name,
                    properties=extract_properties(object_id, self.all_netobj2pro, self.all_subject2pro, self.all_file2pro),
                    label=int(object_id in self.all_labels),
                    frequency=self.node_frequency[actor_id]
                )
            self.node_timestamps[object_id] = timestamp

            actor_idx, object_idx = self.cache_graph.vs.find(name=actor_id).index, self.cache_graph.vs.find(name=object_id).index # 获取 actor 和 object 节点在图中的整数索引（index）
            if not self.cache_graph.are_connected(actor_idx, object_idx): #检查这两个节点之间是否已经存在一条边
                self.cache_graph.add_edge(actor_idx, object_idx, actions=action, timestamp=timestamp)  #加边

            # --- 快照生成逻辑 ---
            n_nodes = len(self.cache_graph.vs)
            if self.first_flag and n_nodes >= snapshot_size:
                self._generate_snapshot()
                self.first_flag = False
            elif not self.first_flag and n_nodes >= snapshot_size * (1 + forgetting_rate):
                self._retire_old_nodes(snapshot_size, forgetting_rate)
                self._generate_snapshot()

        # 处理该图末尾剩余的节点
        if len(self.cache_graph.vs) > 0:
            self._generate_snapshot()

        # --- 后期处理：为所有快照的所有节点打上最终标签 ---
        print("\n正在检查所有节点的属性...")
        for snapshot in self.snapshots:
            for v in snapshot.vs:
                if 'type_name' not in v.attributes():
                    try:
                        v['type_name'] = ObjectType(v['type']).name
                    except ValueError:
                        v['type_name'] = "UNKNOWN_TYPE"

        print("图构建和打标流程全部完成。")

        # === 新增：把每个快照的节点 name+属性写到 TXT ===
        out_path = "snapshot_nodes.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for snapshot_idx, nodes in self.snapshot_to_nodes_map.items():
                for node in nodes:
                    name = node.get("name", "<NA>")
                    # 先写快照信息
                    f.write(f"[Snapshot {snapshot_idx}] ")
                    # 再写节点信息
                    f.write(f"[Node {name}] ")

                    # 写属性
                    attrs = [f"{k}={v}" for k, v in node.items() if k != "name"]
                    if attrs:
                        f.write(" | ".join(attrs))
                    f.write("\n")  # 一行一个节点
        print(f"[INFO] 已保存快照-节点映射到: {out_path}")


        return self.snapshots,  self.graph_to_nodes_map, self.graph_to_label

    def _retire_old_nodes(self, snapshot_size: int, forgetting_rate: float) -> None:
        """这个函数保持不变"""
        n_nodes_to_remove = int(snapshot_size * forgetting_rate)
        if n_nodes_to_remove <= 0: return
        sorted_nodes = sorted(self.node_timestamps.items(), key=lambda item: item[1])
        nodes_to_remove = [node_id for node_id, _ in sorted_nodes[:n_nodes_to_remove]]
        try:
            indices_to_remove = [self.cache_graph.vs.find(name=name).index for name in nodes_to_remove]
            self.cache_graph.delete_vertices(indices_to_remove)
        except ValueError:
            pass # 节点可能已被删除
        for node_id in nodes_to_remove:
            if node_id in self.node_timestamps:
                del self.node_timestamps[node_id]

    def _generate_snapshot(self ) -> None:
        """【修改】记录快照所属的图"""
        snapshot = self.cache_graph.copy()
        self.snapshots.append(snapshot)
        snapshot_idx = len(self.snapshots) - 1
        self.snapshot_to_nodes_map[snapshot_idx] = [
            {
                "name": v['name'],
                **{k: v[k] for k in v.attributes()}  # 把该节点所有属性也一起存进去
            }
            for v in snapshot.vs
        ]

    def extract_label_timestamps(self, dot_file_path: str, labels: list) -> dict:
        """从.dot文件中提取恶意标签的首次出现时间戳"""
        label_timestamps = {}

        with open(dot_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        statements = content.split(";")
        edge_pattern = re.compile(
            r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\[.*?capacity=.*?type="?([^",\]]+)"?.*?timestamp=(\d+)',
            re.IGNORECASE | re.DOTALL
        )

        print(f"正在提取标签时间戳，标签列表: {labels}")

        for stmt in statements:
            if "capacity=" not in stmt:
                continue
            match = edge_pattern.search(stmt)
            if match:
                source, target, edge_type, timestamp = match.groups()
                timestamp = int(timestamp)

                # 检查源节点和目标节点是否在恶意标签中
                for label in labels:
                    if (source.strip() == label or target.strip() == label):
                        if label not in label_timestamps or timestamp < label_timestamps[label]:
                            label_timestamps[label] = timestamp

        print(f"提取到的标签时间戳: {label_timestamps}")
        return label_timestamps

    def split_dataframe_by_time(self, df: pd.DataFrame, label_intervals: dict,
                                buffer_ratio: float = 1.0):
        """
        根据恶意时间区间划分数据：
        - train_df : 远离恶意区间的良性数据
        - test_df  : 恶意数据 + 恶意区间附近的良性数据

        buffer = 恶意区间宽度 × buffer_ratio

        参数:
            df : 原始 DataFrame，必须包含 'timestamp'
            label_intervals : {label: [(start, end), ...]}
            buffer_ratio : buffer 与恶意区间宽度的比例 (默认 3.0)
        返回:
            train_df, test_df
        """
        df = df.copy()
        df["is_malicious"] = 0

        mal_mask = pd.Series(False, index=df.index)
        test_mask = pd.Series(False, index=df.index)

        # 遍历每个 label 的多个时间区间
        for label, intervals in label_intervals.items():
            for (start, end) in intervals:
                width = end - start
                buffer = int(width * buffer_ratio)

                # 恶意区间
                mal_mask |= df["timestamp"].between(start, end, inclusive="both")

                # 测试区间 = 恶意区间 ± buffer
                test_start = start - buffer
                test_end = end + buffer
                test_mask |= df["timestamp"].between(test_start, test_end, inclusive="both")

                print(f"[INFO] Label={label}, 恶意区间 {start}-{end} "
                      f"(宽度={width}) -> buffer={buffer}, 测试区间 {test_start}-{test_end}")

        # 标记恶意
        df.loc[mal_mask, "is_malicious"] = 1

        # 划分
        test_df = df[test_mask].copy()  # 包含恶意 + 附近良性
        train_df = df[~test_mask].copy()  # 远离恶意的纯良性

        print(f"[RESULT] train={len(train_df)}, test={len(test_df)} (其中恶意={mal_mask.sum()})")

        return train_df, test_df

    def save_split_datasets(self, graph_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """将分割后的数据集保存为CSV文件"""
        try:
            # 创建输出文件路径
            train_file = os.path.join(self.base_path, f"{graph_name}_trainlogs.csv")
            test_file = os.path.join(self.base_path, f"{graph_name}_testlogs.csv")


            # 【新增】数据分割质量验证
            total_original = len(train_df) + len(test_df)
            overlap_train_test = len(pd.concat([train_df, test_df]).drop_duplicates()) < total_original

            print(f"  - 数据分割质量检查:")
            print(f"    * 训练集与测试集重叠: {'是' if overlap_train_test else '否'}")

            # 保存数据集到CSV文件
            if not train_df.empty:
                train_df.to_csv(train_file, index=False, encoding='utf-8')
                print(f"  - 训练集已保存到: {train_file} ({len(train_df)} 条边)")
            else:
                print(f"  - 训练集为空，跳过保存")

            if not test_df.empty:
                test_df.to_csv(test_file, index=False, encoding='utf-8')
                print(f"  - 测试集已保存到: {test_file} ({len(test_df)} 条边)")
            else:
                print(f"  - 测试集为空，跳过保存")


        except Exception as e:
            print(f"  - 保存数据集时出错: {str(e)}")



def collect_nodes_from_log(paths):  # dot文件的路径
    # 创建字典
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    nodes = []

    # 读取整个文件
    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分号分隔，处理每个段落
    statements = content.split(';')

    # 正则表达式匹配节点定义
    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)

    for stmt in statements:
        if 'capacity=' in stmt:
            continue  # 跳过包含 capacity 字段的段落
        match = node_pattern.search(stmt)
        if match:
            node_name = match.group(1)
            node_typen = match.group(2)
            nodes.append((node_name, node_typen))
    for node_name, node_typen in nodes:  # 遍历所有的节点
        node_id = node_name  # 节点id赋值
        node_type = node_typen  # 赋值type属性
        # -- 网络流节点 --
        if node_type == 'domain_name':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            domain_name_set[node_id] = nodeproperty
        if node_type == 'IP_Address':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            ip_set[node_id] = nodeproperty
        if node_type == 'connection':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            connection_set[node_id] = nodeproperty
        if node_type == 'session':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            session_set[node_id] = nodeproperty
        if node_type == 'web_object':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            web_object_set[node_id] = nodeproperty
        # -- 进程节点 --
        elif node_type == 'process':
            nodeproperty = node_id
            subject2pro[node_id] = nodeproperty
        # -- 文件节点 --
        elif node_type == 'file':
            nodeproperty = node_id
            file2pro[node_id] = nodeproperty

    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set


def collect_edges_from_log(paths, domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro,
                           file2pro) -> pd.DataFrame:
    """
    从 DOT-like 日志文件中提取含 capacity 的边，并识别 source/target 属于哪个节点集合。
    返回一个包含 source、target、type、timestamp、source_type、target_type 的 DataFrame。
    """
    # 预定义的节点集合

    edges = []

    with open(paths, "r", encoding="utf-8") as f:
        content = f.read()

    statements = content.split(";")

    edge_pattern = re.compile(
        r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\['
        r'.*?capacity=.*?'
        r'type="?([^",\]]+)"?.*?'
        r'timestamp=(\d+)',
        re.IGNORECASE | re.DOTALL
    )

    for stmt in statements:
        if "capacity=" not in stmt:
            continue
        m = edge_pattern.search(stmt)
        if m:
            source, target, edge_type, ts = (x.strip() for x in m.groups())

            # 判断 source/target 所属集合
            if source in domain_name_set:
                source_type = "NETFLOW_OBJECT"
            elif source in ip_set:
                source_type = "NETFLOW_OBJECT"
            elif source in connection_set:
                source_type = "NETFLOW_OBJECT"
            elif source in session_set:
                source_type = "NETFLOW_OBJECT"
            elif source in web_object_set:
                source_type = "NetFlowObject"
            elif source in subject2pro:
                source_type = "SUBJECT_PROCESS"
            elif source in file2pro:
                source_type = "FILE_OBJECT_BLOCK"
            else:
                source_type = "PRINCIPAL_LOCAL"

            if target in domain_name_set:
                target_type = "NETFLOW_OBJECT"
            elif target in ip_set:
                target_type = "NETFLOW_OBJECT"
            elif target in connection_set:
                target_type = "NETFLOW_OBJECT"
            elif target in session_set:
                target_type = "NETFLOW_OBJECT"
            elif target in web_object_set:
                target_type = "NetFlowObject"
            elif target in subject2pro:
                target_type = "SUBJECT_PROCESS"
            elif target in file2pro:
                target_type = "FILE_OBJECT_BLOCK"
            else:
                target_type = "PRINCIPAL_LOCAL"

            edges.append((source, source_type, target, target_type, edge_type, int(ts)))

    return pd.DataFrame(edges, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])



def extract_properties(node_id, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        return node_id