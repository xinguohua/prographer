import json
import hashlib
import os
import re
import time
import igraph as ig
import pandas as pd
from ._base import BaseProcessor
from ._common import (
    add_node_properties,
    add_typed_event_edges,
    collect_json_paths,
    load_optc_released_malicious_uuids_by_host,
    snapshot_local_property,
)
from src.snapshot_construction.snapshot_builder import detect_communities_with_max
from src.snapshot_construction.object_type import ObjectType as optcObjectType
from typing import Optional


PAPER_HOSTS = {"H051", "H201", "H501"}


def paper_host_from_path(path) -> Optional[str]:
    """Normalize H051/H201/H501 and host_0051/0201/0501 spellings."""
    text = str(path).upper()
    match = re.search(r"(?:HOST|H)[^0-9]*0*(51|201|501)(?![0-9])", text)
    if match is None:
        return None
    return f"H{int(match.group(1)):03d}"


class OptcHandler(BaseProcessor):
    """
    OPTC datasethandler
    based on DARPAHandler , supportscenefilterandsnapshotgenerate
    """
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None,
                 dataset_name: str = "optcday1"):
        """
        parameter:
        - base_path: datarootpath
        - train: istrainmodulo
        - scene_name: onlyloadscene (for example "0402"), is None thenloadallcanusescene
        """
        super().__init__(base_path, train)
        self.scene_name = scene_name
        self.dataset_name = dataset_name
        
        self.graph_to_label = {}
        self.total_loaded_bytes = 0
        self.all_dfs = []
        self.labels_by_host = {}

    def load(self):
        """
        load OPTC dataset
         benign/malicious fileprocess
        """
        self.begin = None
        self.malicious = None
        self.all_dfs = []
        benign_parts = []
        malicious_parts = []
        matched_hosts = set()
        
        json_map = collect_json_paths(self.base_path)
        self.all_labels.clear()
        self.labels_by_host = load_optc_released_malicious_uuids_by_host()
        missing_label_hosts = sorted(
            host for host in PAPER_HOSTS if not self.labels_by_host.get(host)
        )
        if missing_label_hosts:
            raise RuntimeError(
                "no PIDSMaker malicious-node labels for OpTC paper hosts: "
                f"{missing_label_hosts}"
            )
        self.all_labels.extend(sorted(set().union(*self.labels_by_host.values())))
        
        for scene, category_data in sorted(json_map.items()):
            # ifconfig scene_name, thenonlypreservescene
            if self.scene_name and scene != self.scene_name:
                continue
            # ifloadall, inuse get_handler time scene_name=None
                
            for category, json_files in category_data.items():
                selected_files = []
                for json_file in sorted(json_files):
                    host = paper_host_from_path(json_file)
                    if host is not None:
                        selected_files.append(json_file)
                json_files = selected_files
                if not json_files:
                    continue
                print(f"currentlyprocess: scene={scene}, class={category}, file={json_files}")
                
                # OPTC has: needstraverse JSON filefrombuild TXT path
                category_dfs = []  # collectwhenbefore category 's has df
                for jf in json_files:
                    abs_json_path = os.path.abspath(jf)
                    
                    if not os.path.isfile(abs_json_path):
                        print(f"[WARN] JSON filedoes not exist: {abs_json_path}, skip")
                        continue
                    
                    self.total_loaded_bytes += os.path.getsize(abs_json_path)
                    
                    # buildtoshould's  TXT filepath
                    dir_name = os.path.dirname(jf)
                    base_name = os.path.basename(jf)
                    name, _ext = os.path.splitext(base_name)
                    parent_dir = os.path.dirname(os.path.dirname(dir_name))
                    last1 = os.path.basename(os.path.dirname(dir_name))
                    last2 = os.path.basename(dir_name)
                    prefix = f"{last1}_{last2}"
                    txt_path = os.path.join(parent_dir, f"{prefix}_{name}.txt")
                    
                    if not os.path.isfile(txt_path):
                        print(f"[WARN] nottotoshould TXT file: {txt_path}, skip")
                        continue
                    
                    df = _read_optc_txt_as_df(txt_path)
                    df = df.dropna()
                    df.sort_values(by="timestamp", ascending=True, inplace=True)
                    
                    print("snapshot-local attributes are extracted from matched event rows")

                    # benign/malicious 
                    if category == "benign":

                        print("==========collect_edges_from_log_optc=======start")
                        t0 = time.time()
                        df = collect_edges_from_log_optc(df, [abs_json_path], True)
                        t1 = time.time()
                        print("==========collect_edges_from_log_optc=======end")
                        print(f"elapsed: {t1 - t0:.2f} second")
                    elif category == "malicious":
                        print("==========collect_edges_from_log_optc=======start")
                        t0 = time.time()
                        df = collect_edges_from_log_optc(df, [abs_json_path], False)
                        t1 = time.time()
                        print("==========collect_edges_from_log_optc=======end")
                        print(f"elapsed: {t1 - t0:.2f} second")

                    matched_host = paper_host_from_path(jf)
                    if matched_host is not None and not df.empty:
                        matched_hosts.add(matched_host)
                        df["host_id"] = matched_host
                        df["host_id_source"] = "optc_release_filename"
                        df["source_scene"] = str(scene)
                    
                    # collectto category_dfs
                    category_dfs.append(df)
                    
                    # mergetototaldataset (for use_df) 
                    self.all_dfs.append(df)
                    
                # benign/malicious 
                if category_dfs:
                    merged_df = pd.concat(category_dfs, ignore_index=True).drop_duplicates()
                    if category == "benign":
                        benign_parts.append(merged_df)
                        print(f"  - benigndata: {len(merged_df)} entryedge")
                    elif category == "malicious":
                        malicious_parts.append(merged_df)
                        print(f"  - maliciousdata: {len(merged_df)} entryedge")
                
        self.begin = pd.concat(benign_parts, ignore_index=True).drop_duplicates() if benign_parts else None
        self.malicious = pd.concat(malicious_parts, ignore_index=True).drop_duplicates() if malicious_parts else None
        if matched_hosts != PAPER_HOSTS:
            missing = sorted(PAPER_HOSTS - matched_hosts)
            raise RuntimeError(
                "OpTC paper profile requires hosts H051/H201/H501; "
                f"missing host files: {missing}"
            )
        if not self.all_dfs:
            raise RuntimeError("no OpTC scene data matched the requested dataset/scene")
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
                raise RuntimeError("OpTC events require H051/H201/H501 host_id before snapshotting")
            if "host_id_source" not in df.columns:
                raise RuntimeError("OpTC events require auditable host_id_source")
            numeric_timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
            numeric_datetime = pd.to_datetime(numeric_timestamp, unit="ms", errors="coerce", utc=True)
            text_datetime = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["timestamp_dt"] = numeric_datetime.fillna(text_datetime)
            df["_time_bin"] = df["timestamp_dt"].dt.floor("1min")
            if "source_scene" not in df.columns or df["source_scene"].astype(str).str.strip().eq("").any():
                raise RuntimeError("OpTC events require source_scene before snapshotting")
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
        host_values = {
            str(value).strip() for value in df.get("host_id", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        }
        if len(host_values) != 1:
            raise RuntimeError(
                "OpTC graph construction requires exactly one host-scoped event frame; "
                f"found {sorted(host_values)}"
            )
        host_id = next(iter(host_values))
        if host_id not in PAPER_HOSTS:
            raise RuntimeError(f"unsupported OpTC paper host {host_id!r}")
        all_labels = set(self.labels_by_host.get(host_id, set()))
        
        _otype_cache = {}
        
        def _otype(v):
            if v not in _otype_cache:
                _otype_cache[v] = optcObjectType[v].value
            return _otype_cache[v]
        
        nodes_props, nodes_type, node_frequency, node_last_ts = {}, {}, {}, {}
        event_rows = []

        for r in df.itertuples(index=False):
            event_rows.append(r)
            action = getattr(r, "action")
            actor_id = getattr(r, "actorID")
            object_id = getattr(r, "objectID")
            raw_ts = getattr(r, "timestamp")
            # OPTCuseISOstring, needsfirst convertisdatetimeagainconvertistimestamp
            if hasattr(r, "timestamp_dt") and pd.notna(r.timestamp_dt):
                timestamp = r.timestamp_dt.timestamp()
            else:
                try:
                    numeric = float(raw_ts)
                    timestamp = numeric / 1000.0 if numeric > 1e11 else numeric
                except (TypeError, ValueError):
                    try:
                        timestamp = pd.to_datetime(raw_ts, utc=True).timestamp()
                    except (TypeError, ValueError):
                        timestamp = 0.0

            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            node_last_ts[actor_id] = max(timestamp, node_last_ts.get(actor_id, 0))
            node_last_ts[object_id] = max(timestamp, node_last_ts.get(object_id, 0))

            # actor node
            props_actor = snapshot_local_property(
                r, action, "actor", _otype(getattr(r, "actor_type")),
            )
            add_node_properties(nodes_props, actor_id, props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = _otype(getattr(r, "actor_type"))

            # object node
            props_obj = snapshot_local_property(
                r, action, "object", getattr(r, "object"),
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
            G, index_map, event_rows, "optc",
        )

        return features, edge_index, node_ids, relations_index, G

    def _process_subgraph(self, subgraph, is_malicious=False, cid=None):
        pass


def _read_optc_txt_as_df(txt_path):
    """read OPTC TXT fileis DataFrame"""
    df = pd.read_csv(txt_path, sep=r"\t| {2,}|\s{1}", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "Source_ID": "actorID",
        "Source_Type": "actor_type",
        "Destination_ID": "objectID",
        "Destination_Type": "object",
        "Edge_Type": "action",
        "Timestamp": "timestamp"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for c in ["actorID", "actor_type", "objectID", "object", "action", "timestamp"]:
        df[c] = df[c].astype(str)
    return df[["actorID", "actor_type", "objectID", "object", "action", "timestamp"]]


def iter_json_records(json_path):
    """iterationread JSON record"""
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read().strip()
    if not data:
        return
    try:
        arr = json.loads(data)
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
            return
    except:
        pass
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks = re.split(r"}\s*{\s*", line)
        if len(chunks) > 1:
            chunks[0] += "}"
            chunks[-1] = "{" + chunks[-1]
            for c in chunks:
                try:
                    obj = json.loads(c)
                    if isinstance(obj, dict):
                        yield obj
                except:
                    continue
        else:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except:
                continue


def collect_edges_from_log_optc(d, paths, benigin):
    """collect OPTC edgeinfo"""
    info = []
    for p in paths:
        for record_number, x in enumerate(iter_json_records(p)):
            action = str(x.get("action", ""))
            actor = str(x.get("actorID", ""))
            obj = str(x.get("objectID", ""))
            ts = str(x.get("timestamp", ""))
            props = x.get("properties", {}) or {}
            cmd = str(props.get("command_line", "") or "")
            path = str(props.get("image_path", "") or "")
            event_id = str(x.get("event_id") or x.get("id") or x.get("uuid") or "").strip()
            if not event_id:
                stable = "\x1f".join([
                    os.path.basename(os.path.dirname(p)), os.path.basename(p),
                    str(record_number), actor, obj, action, ts,
                ])
                event_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()
            info.append({
                'actorID': actor,
                'objectID': obj,
                'action': action,
                'timestamp': ts,
                'exec': cmd,
                'path': str(props.get("file_path", "") or path),
                'actor_path': path,
                'object_path': str(props.get("file_path", "") or ""),
                'event_id': event_id,
            })
    rdf = pd.DataFrame.from_records(info).astype(str)
    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()
