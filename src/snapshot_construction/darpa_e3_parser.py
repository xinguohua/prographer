import os
import time
import numpy as np
import igraph as ig
import orjson
import pandas as pd

from ._base import BaseProcessor
from ._common import (
    add_node_properties,
    add_typed_event_edges,
    cdm_host_identity,
    cdm_event_id,
    collect_json_paths,
    collect_label_paths,
    load_released_malicious_uuids,
    snapshot_local_property,
)
from src.snapshot_construction.snapshot_builder import detect_communities_with_max


from typing import Optional

class DARPAHandler(BaseProcessor):
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None,
                 dataset_name: str = ""):
        """Initialize a DARPA E3 handler with an optional scene filter."""
        super().__init__(base_path, train)
        self.scene_name = scene_name
        self.dataset_name = dataset_name
        self.graph_to_label = {}
        self.total_loaded_bytes = 0
        self.all_dfs = []
    
    def load(self):
        """Join E3 benign/attack index files with their raw CDM events."""
        self.begin = None
        self.malicious = None
        self.all_dfs = []
        benign_parts = []
        malicious_parts = []
        
        json_map = collect_json_paths(self.base_path)
        label_map = collect_label_paths(self.base_path)
        
        self.all_labels.clear()
        released_labels = load_released_malicious_uuids(
            self.dataset_name, self.scene_name,
        )
        if not released_labels:
            raise RuntimeError(
                f"no released malicious labels for dataset={self.dataset_name} "
                f"scene={self.scene_name}"
            )
        self.all_labels.extend(sorted(released_labels))
        
        for scene, category_data in sorted(json_map.items()):
            # Apply the optional scene filter before loading event files.
            if self.scene_name and scene != self.scene_name:
                continue
            if self.train:
                for label_path in label_map.get(scene, []):
                    print(f"[E3] scene={scene} label_file={label_path}")
                    with open(label_path, encoding="utf-8") as label_file:
                        self.all_labels.extend(
                            line.strip() for line in label_file if line.strip()
                        )
                    
            for category, json_files in category_data.items():
                print(f"[E3] scene={scene} category={category} raw_files={json_files}")
                scene_category = f"/{scene}_{category}.txt"
                f = open(self.base_path + scene_category)
                self.total_loaded_bytes += os.path.getsize(self.base_path + scene_category)
                
                data = f.read().split('\n')
                data = [line.split('\t') for line in data]
                df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
                df = df.dropna()
                df.sort_values(by='timestamp', ascending=True, inplace=True)
                df["source_scene"] = str(scene)
                if category == "benign":
                    print("[E3] joining benign index rows to raw CDM events")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, True)
                    t1 = time.time()
                    print(f"[E3] benign_join_seconds={t1 - t0:.2f}")

                    benign_parts.append(df)
                    print(f"[E3] benign_events={len(df)}")
                elif category == "malicious":
                    print("[E3] joining attack index rows to raw CDM events")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, False )
                    t1 = time.time()
                    print(f"[E3] attack_join_seconds={t1 - t0:.2f}")
                    malicious_parts.append(df)
                    print(f"[E3] attack_events={len(df)}")
                
                # Retain each joined frame for downstream audit and reuse.
                self.all_dfs.append(df)
                
        self.begin = pd.concat(benign_parts, ignore_index=True).drop_duplicates() if benign_parts else None
        self.malicious = pd.concat(malicious_parts, ignore_index=True).drop_duplicates() if malicious_parts else None
        if not self.all_dfs:
            raise RuntimeError("no DARPA E3 scene data matched the requested dataset/scene")
        use_df = pd.concat(self.all_dfs, ignore_index=True)
        self.use_df = use_df.drop_duplicates()

    def create_snapshots_from_graph(self, df, is_malicious=False, mode="time"):
        """Build community- or time-partitioned snapshots from joined events."""
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
                    print(f"[E3] warning: failed to build community snapshot: {e}")

        elif mode == "time":
            window = pd.Timedelta(minutes=1)
            if "host_id" not in df.columns or df["host_id"].astype(str).str.strip().eq("").any():
                raise RuntimeError("DARPA E3 events require a stable host_id before snapshotting")
            if "host_id_source" not in df.columns:
                raise RuntimeError("DARPA E3 events require auditable host_id_source")
            df["timestamp_dt"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["timestamp_dt"] = df["timestamp_dt"] // 1000
            df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], unit="us", errors="coerce")
            df["_time_bin"] = df["timestamp_dt"].dt.floor("1min")
            if "source_scene" not in df.columns or df["source_scene"].astype(str).str.strip().eq("").any():
                raise RuntimeError("DARPA E3 events require source_scene before snapshotting")
            grouped = df.groupby(["source_scene", "host_id", "_time_bin"], sort=True)
            for (source_scene, host_id, bin_ts), part in grouped:
                if part.empty:
                    continue

                features, edges, mapp, relations, G = self._build_graph_from_df(part)

                if G.vcount() == 0 or G.ecount() == 0:
                    continue

                self._process_subgraph(G, is_malicious, bin_ts)
                G["host_id"] = str(host_id)
                G["source_scene"] = str(source_scene)
                G["host_id_source"] = str(part["host_id_source"].iloc[0])
                G["window_start"] = bin_ts.timestamp()
                G.vs["_athena_temporal_id"] = [
                    f"{host_id}:{name}" for name in G.vs["name"]
                ]

                snapshots.append(G)
            df.drop(columns=["_time_bin"], inplace=True, errors="ignore")

        return snapshots

    def _build_graph_from_df(self, df):
        """ DataFrame build igraph.Graph, return (features, edges, node_ids, relations, G)"""
        all_labels = set(self.all_labels)
        nodes_props, nodes_type, node_frequency, node_last_ts = {}, {}, {}, {}
        event_rows = []

        # takecolumnis numpy array,  itertuples 's  namedtuple 
        actors = df["actorID"].values
        objects = df["objectID"].values
        actions = df["action"].values
        timestamps = df["timestamp"].values
        actor_types = df["actor_type"].values
        obj_types = df["object"].values
        rows = df.itertuples(index=False)
        for i, row in enumerate(rows):
            event_rows.append(row)
            action = actions[i]
            actor_id = actors[i]
            object_id = objects[i]
            raw_ts = timestamps[i]
            timestamp = float(raw_ts) / 1_000_000_000.0 if raw_ts is not None else 0.0

            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            prev_a = node_last_ts.get(actor_id, 0)
            if timestamp > prev_a:
                node_last_ts[actor_id] = timestamp
            prev_o = node_last_ts.get(object_id, 0)
            if timestamp > prev_o:
                node_last_ts[object_id] = timestamp

            # actor nodeattribute
            props_actor = snapshot_local_property(
                row, action, "actor", actor_types[i],
            )
            if actor_id not in nodes_props:
                nodes_props[actor_id] = set()
            nodes_props[actor_id].add(props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = actor_types[i]

            # object nodeattribute
            props_obj = snapshot_local_property(
                row, action, "object", obj_types[i],
            )
            if object_id not in nodes_props:
                nodes_props[object_id] = set()
            nodes_props[object_id].add(props_obj)
            if object_id not in nodes_type:
                nodes_type[object_id] = obj_types[i]

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

        features = [nodes_props[nid] for nid in node_ids]
        edge_index, relations_index = add_typed_event_edges(
            G, index_map, event_rows, "darpa_e3", timestamp_scale=1_000_000_000.0,
        )

        return features, edge_index, node_ids, relations_index, G

    def _process_subgraph(self, subgraph, is_malicious=False, cid=None):
        pass


def collect_edges_from_log(d, paths, benigin):
    # Restrict raw parsing to indexed actor/object/action/timestamp tuples.
    d = d.astype(str)
    d_keys = set(zip(d["actorID"], d["objectID"], d["action"], d["timestamp"]))

    info = []
    for p in paths:
        with open(p, "rb") as f:
            for record_number, line in enumerate(f):
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
                host_id, host_id_source = cdm_host_identity(x, ev, p)
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
                        "timestamp": timestamp, "exec": cmd, "path": path2,
                        "host_id": host_id,
                        "host_id_source": host_id_source,
                        "event_id": cdm_event_id(ev, p, record_number, "predicateObject2"),
                    })

                if (actor, obj, action, timestamp) in d_keys:
                    info.append({
                        "actorID": actor, "objectID": obj, "action": action,
                        "timestamp": timestamp, "exec": cmd, "path": path,
                        "host_id": host_id,
                        "host_id_source": host_id_source,
                        "event_id": cdm_event_id(ev, p, record_number, "predicateObject"),
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
