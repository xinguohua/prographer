import os.path
import pandas as pd
import igraph as ig
import re
from igraph import Graph
from ._base import BaseProcessor
from ._common import merge_properties, collect_dot_paths, collect_atlas_label_paths
from ._type_enum import ObjectType
from typing import Optional
class ATLASHandler(BaseProcessor):
    #def __init__(self, base_path=None, train=True):
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None):
        super().__init__(base_path, train)

        self.graph_to_label = {}
        self.all_netobj2pro = {}
        self.all_subject2pro = {}
        self.all_file2pro = {}
        self.total_loaded_bytes = 0


    def load(self):
        """load ATLAS dataset. 
        - trainmodulo: maliciouslabelpart
        - testmodulo: preservehasdata
        """
        print("process ATLAS dataset...")
        graph_files = collect_dot_paths(self.base_path)  # has .dot filepath
        label_map = collect_atlas_label_paths(self.base_path)

        self.all_labels.clear()
        self.graph_to_label.clear()

        def filter_bad_edges(df, labels):
            """filtermaliciousedge"""
            if not labels:
                return df, len(df), len(df)
            bad = set(labels)
            before = len(df)
            mask_bad = df.isin(bad).any(axis=1)
            df_clean = df.loc[~mask_bad]
            after = len(df_clean)
            return df_clean, before, after

        malicious_name = "M1-CVE-2015-5122_windows_h1"
        benign_name = "M1-CVE-2015-5122_windows_h2"

        for dot_file in graph_files:
            dot_name = os.path.splitext(os.path.basename(dot_file))[0]
            if dot_name not in [malicious_name, benign_name]:
                continue
            print(f"currentlyloadscene: {dot_name}")

            if dot_name in label_map:
                with open(label_map[dot_name], 'r', encoding='utf-8') as label_file:
                    graph_labels = [line.strip() for line in label_file if line.strip()]
                    self.graph_to_label[dot_name] = graph_labels
            else:
                if not self.train:
                    print(f"  - warning: toscene '{dot_name}' 's labelfile. ")


            # parse .dot file -> DataFrame
            netobj2pro, subject2pro, file2pro, dns, ips, conns, sess, webs = collect_nodes_from_log(dot_file)
            dot_df= collect_edges_from_log(dot_file, dns, ips, conns, sess, webs, subject2pro, file2pro)

            if dot_name == benign_name:
                df_begin, before, after = filter_bad_edges(dot_df, self.graph_to_label[dot_name])
                print(f"  - benign graphall: {len(df_begin)} entryedge")
                self.begin = df_begin
            elif dot_name == malicious_name:
                print(f"  - malicious graphall: {len(dot_df)} entryedge")
                self.malicious = dot_df
                self.all_labels.extend(self.graph_to_label[dot_name])

            merge_properties(netobj2pro, self.all_netobj2pro)
            merge_properties(subject2pro, self.all_subject2pro)
            merge_properties(file2pro, self.all_file2pro)

        self.all_labels = list(set(self.all_labels))
        if not self.train:
            print(f"to {len(self.all_labels)} only1maliciouslabel: {self.all_labels}")

    def create_snapshots_from_graph(self, df, is_malicious):
        snapshots = []
        if df is None or len(df) == 0:
            return []

        sorted_df = df.sort_values(by='timestamp') if 'timestamp' in df.columns else df
        chunks = []
        if 'timestamp' in sorted_df.columns:
            ts = pd.to_numeric(sorted_df['timestamp'], errors='coerce')
            if ts.notna().any():
                tmp = sorted_df.copy()
                tmp['timestamp_dt'] = pd.to_datetime(ts, unit='s', errors='coerce')
                if tmp['timestamp_dt'].isna().all():
                    tmp['timestamp_dt'] = pd.to_datetime(ts, unit='ms', errors='coerce')
                if not tmp['timestamp_dt'].isna().all():
                    window = pd.Timedelta(minutes=1)
                    t_min, t_max = tmp['timestamp_dt'].min(), tmp['timestamp_dt'].max()
                    bins = pd.date_range(start=t_min, end=t_max + window, freq=window)
                    chunks = [
                        tmp[(tmp['timestamp_dt'] >= bins[i]) & (tmp['timestamp_dt'] < bins[i + 1])]
                        for i in range(len(bins) - 1)
                    ]
        if not chunks:
            snapshot_size = 100
            chunks = [
                sorted_df.iloc[start:start + snapshot_size]
                for start in range(0, len(sorted_df), snapshot_size)
            ]

        for chunk in chunks:
            if chunk.empty:
                continue

            # statisticin's nodefrequencyandclass (attribute frequency takeinsideappeartimenumber) 
            node_freq = {}
            node_type_map = {}
            for _, row in chunk.iterrows():
                actor_id, object_id = row["actorID"], row["objectID"]
                node_freq[actor_id] = node_freq.get(actor_id, 0) + 1
                node_freq[object_id] = node_freq.get(object_id, 0) + 1
                node_type_map.setdefault(actor_id, row['actor_type'])
                node_type_map.setdefault(object_id, row['object'])

            g = ig.Graph(directed=True)
            for node_id, freq in node_freq.items():
                type_str = node_type_map.get(node_id, 'UNKNOWN')
                try:
                    type_enum = ObjectType[type_str]
                    type_name = type_enum.name
                except Exception:
                    type_name = str(type_str)
                g.add_vertex(
                    name=node_id,
                    type=type_name,
                    properties=extract_properties(node_id, self.all_netobj2pro, self.all_subject2pro, self.all_file2pro),
                    label=int(any(lbl in node_id for lbl in self.all_labels)),
                    frequency=int(freq)
                )

            for _, row in chunk.iterrows():
                actor_id, object_id = row["actorID"], row["objectID"]
                action = row["action"]
                timestamp = row.get("timestamp", 0)
                try:
                    a_idx = g. vs .find(name=actor_id).index
                    o_idx = g. vs .find(name=object_id).index
                    g.add_edge(a_idx, o_idx, actions=action, timestamp=timestamp)
                except ValueError:
                    continue

            snapshots.append(g)

        return snapshots



    def _retire_old_nodes(self, snapshot_size: int, forgetting_rate: float, node_timestamps: dict, cache_graph: Graph) -> None:
        """thisfunctionnot"""
        n_nodes_to_remove = int(snapshot_size * forgetting_rate)
        if n_nodes_to_remove <= 0:
            return
        sorted_nodes = sorted(node_timestamps.items(), key=lambda item: item[1])
        nodes_to_remove = [node_id for node_id, _ in sorted_nodes[:n_nodes_to_remove]]
        try:
            indices_to_remove = [cache_graph. vs .find(name=name).index for name in nodes_to_remove]
            cache_graph.delete_vertices(indices_to_remove)
        except ValueError:
            pass
        for node_id in nodes_to_remove:
            if node_id in node_timestamps:
                del node_timestamps[node_id]

    def _generate_snapshot(self, cache_graph, snapshots) -> None:
        snapshot = cache_graph.copy()
        snapshots.append(snapshot)






def collect_nodes_from_log(paths):  # dotfile's path
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    nodes = []

    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()

    statements = content.split(';')

    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)

    for stmt in statements:
        if 'capacity=' in stmt:
            continue  # skipcontains capacity field's paragraph
        match = node_pattern.search(stmt)
        if match:
            node_name = match.group(1)
            node_typen = match.group(2)
            nodes.append((node_name, node_typen))
    for node_name, node_typen in nodes:
        node_id = node_name  # nodeid
        node_type = node_typen  # typeattribute
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
        elif node_type == 'process':
            nodeproperty = node_id
            subject2pro[node_id] = nodeproperty
        elif node_type == 'file':
            nodeproperty = node_id
            file2pro[node_id] = nodeproperty

    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set


def collect_edges_from_log(paths, domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro,
                           file2pro) -> pd.DataFrame:
    """
    from DOT-like logfileintakecontains capacity 's edge, identify source/target innodeset. 
    return1contains source, target, type, timestamp, source_type, target_type 's  DataFrame. 
    """

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

            # break source/target set
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
