import json
import os
import time
import orjson
import igraph as ig
import pandas as pd
from ._base import BaseProcessor
from ._common import (
    add_node_properties,
    add_typed_event_edges,
    cdm_host_identity,
    cdm_event_id,
    collect_json_paths,
    load_released_malicious_uuids,
    normalize_cdm_uuid,
    snapshot_local_property,
)
from src.snapshot_construction.snapshot_builder import detect_communities_with_max
from typing import Optional


class DARPAHandler5(BaseProcessor):
    """
    DARPA E5 datasethandler (CDM20 ) 
    based on DARPAHandler , supportscenefilterandsnapshotgenerate
    """
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None,
                 dataset_name: str = ""):
        """
        parameter:
        - base_path: datarootpath
        - train: istrainmodulo
        - scene_name: onlyloadscene (for example "cadets104"), is None thenloadallcanusescene
        """
        super().__init__(base_path, train)
        self.scene_name = scene_name
        self.dataset_name = dataset_name
        
        self.graph_to_label = {}
        self.total_loaded_bytes = 0
        self.all_dfs = []

    def load(self):
        """
        load DARPA E5 dataset (CDM20 )
         benign/malicious fileprocess
        """
        self.begin = None
        self.malicious = None
        self.all_dfs = []
        benign_parts = []
        malicious_parts = []
        
        json_map = collect_json_paths(self.base_path)
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
            # ifconfig scene_name, thenonlypreservescene
            if self.scene_name and scene != self.scene_name:
                continue
            # beforeencode cadets104 's logic: timestillcanfilterto cadets104
            # ifloadall, inuse get_handler time scene_name=None
                
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
                df["source_scene"] = str(scene)
                # benign/malicious 
                if category == "benign":

                    print("==========collect_edges_from_log=======start")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, True)
                    t1 = time.time()
                    print("==========collect_edges_from_log=======end")
                    print(f"elapsed: {t1 - t0:.2f} second")

                    benign_parts.append(df)
                    print(f"  - benigndata: {len(df)} entryedge")
                elif category == "malicious":
                    print("==========collect_edges_from_log=======start")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, False )
                    t1 = time.time()
                    print("==========collect_edges_from_log=======end")
                    print(f"elapsed: {t1 - t0:.2f} second")
                    malicious_parts.append(df)
                    print(f"  - maliciousdata: {len(df)} entryedge")
                
                # mergetototaldataset (for use_df) 
                self.all_dfs.append(df)
                
        self.begin = pd.concat(benign_parts, ignore_index=True).drop_duplicates() if benign_parts else None
        self.malicious = pd.concat(malicious_parts, ignore_index=True).drop_duplicates() if malicious_parts else None
        if not self.all_dfs:
            raise RuntimeError("no DARPA E5 scene data matched the requested dataset/scene")
        use_df = pd.concat(self.all_dfs, ignore_index=True)
        self.use_df = use_df.drop_duplicates()

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
            if "host_id" not in df.columns or df["host_id"].astype(str).str.strip().eq("").any():
                raise RuntimeError("DARPA E5 events require a stable host_id before snapshotting")
            if "host_id_source" not in df.columns:
                raise RuntimeError("DARPA E5 events require auditable host_id_source")
            df["timestamp_dt"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["timestamp_dt"] = df["timestamp_dt"] // 1000
            df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], unit="us", errors="coerce")  # convertis datetime
            df["_time_bin"] = df["timestamp_dt"].dt.floor("1min")
            if "source_scene" not in df.columns or df["source_scene"].astype(str).str.strip().eq("").any():
                raise RuntimeError("DARPA E5 events require source_scene before snapshotting")
            for (source_scene, host_id, bin_ts), part in df.groupby(
                ["source_scene", "host_id", "_time_bin"], sort=True,
            ):
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

        for r in df.itertuples(index=False):
            event_rows.append(r)
            action = getattr(r, "action")
            actor_id = getattr(r, "actorID")
            object_id = getattr(r, "objectID")
            raw_ts = getattr(r, "timestamp")
            timestamp = float(raw_ts) / 1_000_000_000.0 if raw_ts is not None else 0.0

            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            node_last_ts[actor_id] = max(timestamp, node_last_ts.get(actor_id, 0))
            node_last_ts[object_id] = max(timestamp, node_last_ts.get(object_id, 0))

            # actor node
            props_actor = snapshot_local_property(
                r, action, "actor", getattr(r, "actor_type", ""),
            )
            add_node_properties(nodes_props, actor_id, props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = getattr(r, "actor_type")

            # object node
            props_obj = snapshot_local_property(
                r, action, "object", getattr(r, "object", ""),
            )
            add_node_properties(nodes_props, object_id, props_obj)
            if object_id not in nodes_type:
                nodes_type[object_id] = getattr(r, "object")

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
            G, index_map, event_rows, "darpa_e5", timestamp_scale=1_000_000_000.0,
        )

        return features, edge_index, node_ids, relations_index, G

    def _process_subgraph(self, subgraph, is_malicious=False, cid=None):
        pass


def collect_edges_from_log(d, paths, benigin):
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
                    ev = x["datum"]["com.bbn.tc.schema.avro.cdm20.Event"]
                except Exception:
                    continue

                action = ev.get("type", "")
                host_id, host_id_source = cdm_host_identity(x, ev, p)
                actor = normalize_cdm_uuid(
                    (ev.get("subject") or {}).get("com.bbn.tc.schema.avro.cdm20.UUID", "")
                )
                obj = normalize_cdm_uuid(
                    (ev.get("predicateObject") or {}).get("com.bbn.tc.schema.avro.cdm20.UUID", "")
                )
                timestamp = ev.get("timestampNanos", "")
                cmd = ((ev.get("properties") or {}).get("map") or {}).get("cmdLine", "")
                path = (ev.get("predicateObjectPath") or {}).get("string", "")
                path2 = (ev.get("predicateObject2Path") or {}).get("string", "")

                obj2 = normalize_cdm_uuid(
                    (ev.get("predicateObject2") or {}).get("com.bbn.tc.schema.avro.cdm20.UUID")
                )
                if obj2:
                    info.append({
                        "actorID": actor, "objectID": obj2, "action": action,
                        "timestamp": timestamp, "exec": cmd, "path": path2,
                        "host_id": host_id,
                        "host_id_source": host_id_source,
                        "event_id": cdm_event_id(ev, p, record_number, "predicateObject2"),
                    })

                info.append({
                    "actorID": actor, "objectID": obj, "action": action,
                    "timestamp": timestamp, "exec": cmd, "path": path,
                    "host_id": host_id,
                    "host_id_source": host_id_source,
                    "event_id": cdm_event_id(ev, p, record_number, "predicateObject"),
                })

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)
    for frame in (rdf, d):
        frame["actorID"] = frame["actorID"].map(normalize_cdm_uuid)
        frame["objectID"] = frame["objectID"].map(normalize_cdm_uuid)

    return d.merge(rdf, how="inner",
                   on=["actorID", "objectID", "action", "timestamp"]) \
        .drop_duplicates()
