import os
import re
import time
import numpy as np
import igraph as ig
import orjson
import pandas as pd

from ._base import BaseProcessor
from ._common import collect_json_paths, collect_label_paths
from ._common import merge_properties, add_node_properties
from src.snapshot_construction.snapshot_builder import detect_communities_with_max


from typing import Optional

class DARPAHandler(BaseProcessor):
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None):
        """DARPA handler. 

        parameter:
        - base_path: datarootpath
        - train: istrainmodulo
        - scene_name: onlyloadscene (for example "cadets314"), is None thenloadallcanusescene
        """
        super().__init__(base_path, train)
        self.scene_name = scene_name
        self.graph_to_label = {}
        self.all_netobj2pro = {}
        self.all_subject2pro = {}
        self.all_file2pro = {}
        self.total_loaded_bytes = 0
        self.all_dfs = []
    
    def load(self):
        """
        load DARPA dataset - logic, but benign/malicious fileprocess
        
        parameter:
        - load_all_for_encoder: ifisTrue, loadalldataforencodetrain; ifisFalse, logicload
        """
        self.begin = None
        self.malicious = None
        
        json_map = collect_json_paths(self.base_path)
        label_map = collect_label_paths(self.base_path)
        
        self.all_labels.clear()
        
        for scene, category_data in json_map.items():
            # ifconfig scene_name, thenonlypreservescene
            if self.scene_name and scene != self.scene_name:
                continue
            # beforeencode cadets314 's logic: timestillcanfilterto cadets314
            # ifloadall, inuse get_handler time scene_name=None
                
            if self.train:
                if scene in label_map:
                    label_file = open(label_map[scene])
                    print(f"currentlyprocess: scene={scene}, label={label_map[scene]}")
                    self.all_labels.extend([
                        line.strip() for line in label_file.read().splitlines() if line.strip()
                    ])
                    
            for category, json_files in category_data.items():
                print(f"currentlyprocess: scene={scene}, class={category}, file={json_files}")
                scene_category = f"/{scene}_{category}.txt"
                f = open(self.base_path + scene_category)
                self.total_loaded_bytes += os.path.getsize(self.base_path + scene_category)
                
                data = f.read().split('\n')
                data = [line.split('\t') for line in data]
                df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
                df = df.dropna()
                df.sort_values(by='timestamp', ascending=True, inplace=True)
                netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)

                # benign/malicious 
                if category == "benign":

                    print("==========collect_edges_from_log=======start")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, True)
                    t1 = time.time()
                    print("==========collect_edges_from_log=======end")
                    print(f"elapsed: {t1 - t0:.2f} second")

                    self.begin = df  # to base.py 's attribute
                    print(f"  - benigndata: {len(df)} entryedge")
                elif category == "malicious":
                    print("==========collect_edges_from_log=======start")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, False )
                    t1 = time.time()
                    print("==========collect_edges_from_log=======end")
                    print(f"elapsed: {t1 - t0:.2f} second")
                    self.malicious = df  # to base.py 's attribute
                    print(f"  - maliciousdata: {len(df)} entryedge")
                
                # mergetototaldataset (for use_df) 
                self.all_dfs.append(df)
                
                merge_properties(netobj2pro, self.all_netobj2pro)
                merge_properties(subject2pro, self.all_subject2pro)
                merge_properties(file2pro, self.all_file2pro)
                
        use_df = pd.concat(self.all_dfs, ignore_index=True)
        self.use_df = use_df.drop_duplicates()
        # === according toafter's  use_df size, can mock scale ===
        try:
            base_bytes = int(self.use_df.memory_usage(deep=True).sum())
        except Exception:
            base_bytes = self.use_df.memory_usage().sum()
        print(f"[Handler] use_df aftermemory: {base_bytes/1024/1024:.2f} MB")

        # optional: connecttargetscale, supportnumber (section) orstring (like "10GB", "20GB", "500MB") 
        def _parse_size(val):
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                s = val.strip().upper()
                if s.endswith("GB"):
                    num = float(s[:-2].strip())
                    return int(num * (1024 ** 3))
                if s.endswith("MB"):
                    num = float(s[:-2].strip())
                    return int(num * (1024 ** 2))
                if s.endswith("KB"):
                    num = float(s[:-2].strip())
                    return int(num * 1024)
                try:
                    return int(float(s))
                except Exception:
                    return 0
            return 0

        # connectinthis set: 0 / "10GB" / "20GB" / "500MB" etc.
        # target_size = "40GB"  # example: target_size = "10GB" or 0 denotes no
        # target_bytes = _parse_size(target_size)
        # if target_bytes and base_bytes > 0 and len(self.use_df) > 0:
        #     factor = max(1, int((target_bytes + base_bytes - 1) // max(1, base_bytes)))
        #     if factor > 1:
        #         print(f"[Handler]  mock scale: target={target_bytes/1024/1024/1024:.2f} GB, ={factor}")
        #         # viafromnearlike, 1timelarge concat becomememory
        #         blocks = []
        #         chunk = self.use_df
        #         remaining = factor
        #         while remaining > 0:
        #             take = min(remaining, 8)  # eachtimeat most 8 timeby
        #             blocks.append(pd.concat([chunk] * take, ignore_index=True))
        #             remaining -= take
        #         self.use_df = pd.concat(blocks, ignore_index=True)
        #         try:
        #             mock_bytes = int(self.use_df.memory_usage(deep=True).sum())
        #         except Exception:
        #             mock_bytes = self.use_df.memory_usage().sum()
        #         print(f"[Handler] mock aftermemory: {mock_bytes/1024/1024/1024:.2f} GB")

    def create_snapshots_from_graph(self, df, is_malicious=False, mode="time"):
        """
        usesnapshotgeneratefunction
        - mode: "community" or "time"
        - is_malicious: ismaliciousdata
        """
        if df is None or len(df) == 0:
            return []

        snapshots = []

        if mode == "community":
            features, edges, mapp, relations, G = self._build_graph_from_df(df)

            communities = detect_communities_with_max(G)
            name_to_idx = {v["name"]: v.index for v in G. vs }

            for community_id, node_names in communities.items():
                try:
                    node_indices = [name_to_idx[name] for name in node_names if name in name_to_idx]
                    if not node_indices:
                        continue

                    subgraph = G.subgraph(node_indices)
                    self._process_subgraph(subgraph, is_malicious, community_id)
                    snapshots.append(subgraph)
                except Exception as e:
                    print(f"warning: snapshottimeoutwrong: {e}")

        elif mode == "time":
            window = pd.Timedelta(minutes=1)
            df["timestamp_dt"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["timestamp_dt"] = df["timestamp_dt"] // 1000
            df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], unit="us", errors="coerce")  # convertis datetime
            t_min, t_max = df["timestamp_dt"].min(), df["timestamp_dt"].max()
            if pd.isna(t_min) or pd.isna(t_max):
                return []
            # use groupby agentforfilter, traverselargeempty
            df["_time_bin"] = df["timestamp_dt"].dt.floor("1min")
            grouped = df.groupby("_time_bin", sort=True)
            for bin_ts, part in grouped:
                if part.empty:
                    continue

                features, edges, mapp, relations, G = self._build_graph_from_df(part)

                if G.vcount() == 0 or G.ecount() == 0:
                    continue

                self._process_subgraph(G, is_malicious, bin_ts)

                snapshots.append(G)
            df.drop(columns=["_time_bin"], inplace=True, errors="ignore")

        return snapshots

    def _build_graph_from_df(self, df):
        """ DataFrame build igraph.Graph, return (features, edges, node_ids, relations, G)"""
        all_labels = set(self.all_labels)
        nodes_props, nodes_type, edges_map, node_frequency, node_last_ts = {}, {}, {}, {}, {}

        # takeattributedict's localuse, subtract self find
        _netobj2pro = self.all_netobj2pro
        _subject2pro = self.all_subject2pro
        _file2pro = self.all_file2pro

        # takecolumnis numpy array,  itertuples 's  namedtuple 
        actors = df["actorID"].values
        objects = df["objectID"].values
        actions = df["action"].values
        timestamps = df["timestamp"].values
        actor_types = df["actor_type"].values
        obj_types = df["object"].values
        # exec/path cancandoes not exist (fromnotstage's  df) 
        execs = df["exec"].values if "exec" in df.columns else [""]*len(df)
        paths = df["path"].values if "path" in df.columns else [""]*len(df)

        for i in range(len(actors)):
            action = actions[i]
            actor_id = actors[i]
            object_id = objects[i]
            raw_ts = timestamps[i]
            timestamp = float(raw_ts) if raw_ts is not None else 0.0

            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            prev_a = node_last_ts.get(actor_id, 0)
            if timestamp > prev_a:
                node_last_ts[actor_id] = timestamp
            prev_o = node_last_ts.get(object_id, 0)
            if timestamp > prev_o:
                node_last_ts[object_id] = timestamp

            # actor nodeattribute
            props_actor = _netobj2pro.get(actor_id) or _file2pro.get(actor_id) or _subject2pro.get(actor_id)
            if props_actor is None:
                exec_cmd = execs[i] if execs[i] else ""
                path_val = paths[i] if paths[i] else ""
                props_actor = f"{exec_cmd} {action} {path_val}".strip() if path_val else f"{exec_cmd} {action}"
            if actor_id not in nodes_props:
                nodes_props[actor_id] = set()
            nodes_props[actor_id].add(props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = actor_types[i]

            # object nodeattribute
            props_obj = _netobj2pro.get(object_id) or _file2pro.get(object_id) or _subject2pro.get(object_id)
            if props_obj is None:
                exec_cmd = execs[i] if execs[i] else ""
                path_val = paths[i] if paths[i] else ""
                props_obj = f"{exec_cmd} {action} {path_val}".strip() if path_val else f"{exec_cmd} {action}"
            if object_id not in nodes_props:
                nodes_props[object_id] = set()
            nodes_props[object_id].add(props_obj)
            if object_id not in nodes_type:
                nodes_type[object_id] = obj_types[i]

            key = (actor_id, object_id)
            entry = edges_map.get(key)
            if entry is None:
                edges_map[key] = {"actions": {action}, "timestamp": [timestamp]}
            else:
                entry["actions"].add(action)
                entry["timestamp"].append(timestamp)

        node_ids = list(nodes_props.keys())
        index_map = {nid: i for i, nid in enumerate(node_ids)}

        G = ig.Graph(directed=True)
        G.add_vertices(len(node_ids))
        G. vs ["name"] = node_ids
        G. vs ["type"] = [nodes_type.get(nid) for nid in node_ids]
        G. vs ["properties"] = [str(nodes_props[nid]) for nid in node_ids]
        G. vs ["label"] = [1 if nid in all_labels else 0 for nid in node_ids]
        G. vs ["frequency"] = [node_frequency.get(nid, 0) for nid in node_ids]
        G. vs ["timestamp"] = [node_last_ts.get(nid, 0) for nid in node_ids]

        unique_edges = list(edges_map.keys())
        if unique_edges:
            edge_idx = [(index_map[a], index_map[b]) for (a, b) in unique_edges]
            G.add_edges(edge_idx)
            G.es["actions"] = [
                ",".join(sorted(edges_map[(a, b)]["actions"]))
                if not isinstance(edges_map[(a, b)]["actions"], str)
                else edges_map[(a, b)]["actions"]
                for (a, b) in unique_edges
            ]
            G.es["timestamp"] = [
                max(edges_map[(a, b)]["timestamp"])
                for (a, b) in unique_edges
            ]
        features = [nodes_props[nid] for nid in node_ids]
        edge_index = [[], []]
        relations_index = {}
        for a, b in unique_edges:
            s, d = index_map[a], index_map[b]
            edge_index[0].append(s)
            edge_index[1].append(d)
            relations_index[(s, d)] = list(edges_map[(a, b)])

        return features, edge_index, node_ids, relations_index, G

    def _process_subgraph(self, subgraph, is_malicious=False, cid=None):
        pass
        # if is_malicious:
        #     labels = subgraph. vs ["label"] if "label" in subgraph. vs .attributes() else []
        #     mal_nodes = sum(lbl == 1 for lbl in labels)
        #     if mal_nodes > 0:
        #         print(f" {cid} ismalicious (malicious nodenumber={mal_nodes})")
        #         for v in subgraph. vs :
        #             for attr, old_val in v.attributes().items():
        #                 new_val = _replace_event_in_value(old_val)
        #                 if new_val != old_val:
        #                     print(f"malicious val ===== change {old_val} -> {new_val}")
        #                     v[attr] = new_val




def collect_nodes_from_log(paths):
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    for p in paths:
        with open(p) as f:
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
                    except Exception:
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
                        except Exception:
                            path = "null"
                        nodeProperty = f"{cmdLine},{tgid},{path}"
                        subject2pro[nodeid] = nodeProperty
                    except Exception:
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
                    except Exception:
                        pass

    return netobj2pro, subject2pro, file2pro


def collect_edges_from_log(d, paths, benigin, max_lines= 600000):
    # first use set index d in's  (actorID, objectID, action, timestamp) 4tuple, 
    # afteronlycollectmatch's , large DataFrame again merge
    d = d.astype(str)
    d_keys = set(zip(d["actorID"], d["objectID"], d["action"], d["timestamp"]))

    info = []
    for p in paths:
        with open(p, "rb") as f:
            for i, line in enumerate(f):
                if benigin and i >= max_lines:
                    break
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
                timestamp = str(ev.get("timestampNanos", ""))
                cmd = ((ev.get("properties") or {}).get("map") or {}).get("cmdLine", "")
                path = (ev.get("predicateObjectPath") or {}).get("string", "")
                path2 = (ev.get("predicateObject2Path") or {}).get("string", "")

                obj2 = (ev.get("predicateObject2") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID")
                if obj2 and (actor, obj2, action, timestamp) in d_keys:
                    info.append({
                        "actorID": actor, "objectID": obj2, "action": action,
                        "timestamp": timestamp, "exec": cmd, "path": path2
                    })

                if (actor, obj, action, timestamp) in d_keys:
                    info.append({
                        "actorID": actor, "objectID": obj, "action": action,
                        "timestamp": timestamp, "exec": cmd, "path": path
                    })

    if not info:
        # withoutmatch, returnadditionalcolumn's empty DataFrame
        result = d.copy()
        for col in ["exec", "path"]:
            if col not in result.columns:
                result[col] = ""
        return result.iloc[:0]

    rdf = pd.DataFrame.from_records(info).astype(str)
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

_EVENT_TOKEN = re.compile(r'(?<!\w)EVENT[^\s]*')

def _replace_event_in_value(val):
    if isinstance(val, str):
        return _EVENT_TOKEN.sub("chentuoyu", val)
    elif isinstance(val, list):
        return [_replace_event_in_value(x) for x in val]
    elif isinstance(val, tuple):
        return tuple(_replace_event_in_value(x) for x in val)
    elif isinstance(val, dict):
        return {k: _replace_event_in_value(v) for k, v in val.items()}
    elif isinstance(val, set):
        return {_replace_event_in_value(x) for x in val}
    else:
        return val