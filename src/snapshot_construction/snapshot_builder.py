import igraph as ig
import leidenalg as la
from src.snapshot_construction.object_type import ObjectType
from collections import defaultdict
import re

resource_types = {ObjectType.NETFLOW_OBJECT.value, ObjectType.FILE_OBJECT_BLOCK.value, ObjectType.MemoryObject.value}

def create_process_graph():
    """1process's hastowardgraph"""
    G = ig.Graph(directed=True)

    # processes = [
    #     {"name": "Process0", "type": "process"},
    #     {"name": "Process1", "type": "process"},
    #     {"name": "Process2", "type": "process"},
    #     {"name": "Process3", "type": "process"},
    #     {"name": "Process4", "type": "process"},
    #     {"name": "FileA", "type": "resource"},  # filesourcenode
    #     {"name": "Process5", "type": "process"},
    #     {"name": "Process6", "type": "process"},
    #     {"name": "Process7", "type": "process"},
    # ]
    processes = [
        {"name": "Process0", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process1", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process2", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process3", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Process4", "type": ObjectType.SUBJECT_PROCESS.value},
        {"name": "Socket", "type": ObjectType.NETFLOW_OBJECT.value},
        {"name": "File", "type": ObjectType.FILE_OBJECT_BLOCK.value},
    ]

    G.add_vertices(len(processes))
    for i, process in enumerate(processes):
        G. vs [i]["name"] = process["name"]
        G. vs [i]["type"] = process["type"]

    # edges = [
    #     ("Process0", "Process1"),  # basic
    #     ("Process1", "Process2"),  # basic
    #     ("Process1", "Process3"),  # basic
    #     ("Process2", "Process4"),  # sourcedependency
    #     ("Process0", "FileA"),  # process2  fileA (sourcedependency) 
    #     ("Process4", "FileA"),  # process4  fileA (sourcedependency) 
    #     ("Process3", "Process5"),  # basic
    #     ("Process3", "Process6"),  # basic
    #     ("Process3", "Process7"),  # basic
    # ]
    edges = [
        ("Process0", "Socket"),  # Process0 connectto Socket
        ("Process1", "Socket"),  # Process1 via Socket communicate
        ("Process4", "Process1"),  # Process1 connect Process4
        ("Process4", "Process2"),  # Process2 connect Process4
        ("File", "Process3"),  # Process3  File
        ("File", "Process2"),  # File impact Process2
        ("Process0", "File"),  # process0 file (sourcedependency) 
        ("File", "Process1"),  # process1 file (sourcedependency) 
    ]
    G.add_edges(edges)

    return G


def set_weight(G):
    set_default_weight(G)
    # set_process_weights(G)
    # set_resource_weights(G)


def is_resource_dependent(G, source, target):
    """
    break source and target isviasourcebecomewithouttowardcycle, returncycle's size. 
    - if source to target existspath, pathupcontains resource node, thenissourcedependency. 
    - return (isexistssourcedependency, sourcedependencycycle's minsize). 
    """
    source_idx = G. vs .find(name=source).index
    target_idx = G. vs .find(name=target).index

    G_undirected = G.as_undirected()
    paths = get_all_paths(G_undirected, source_idx, target_idx)

    min_loop_size = float("inf")
    has_resource_dependency = False

    for path in paths:
        if any((G_undirected. vs [node_idx]["type"] in resource_types) for node_idx in path):
            has_resource_dependency = True
            min_loop_size = min(min_loop_size, len(path))

    if has_resource_dependency:
        return True, min_loop_size
    else:
        return False, -1  # nosourcedependency, cyclesizereturn -1


def set_resource_weights(G, W_base=1.0):
    """
    statisticeachsourcenodeprocess, according toclass's processcountintegeredgeweight. 
    - processclassisaccording towithouttowardgraph's 's . 
    - connectaccording toclass's processcountweight, withoutadditionaltraverseeachedgetwotime. 
    """
    resource_access = {}

    for v in G. vs :
        if v["type"] in resource_types:
            resource_access[v["name"]] = []

    for edge in G.es:
        source = G. vs [edge.source]["name"]
        target = G. vs [edge.target]["name"]

        if G. vs [edge.source]["type"] == ObjectType.SUBJECT_PROCESS.value and G. vs [edge.target]["type"] in resource_types:
            resource_access[target].append(source)  # source source process
        elif G. vs [edge.source]["type"] in resource_types and G. vs [edge.target]["type"] == ObjectType.SUBJECT_PROCESS.value:
            resource_access[source].append(target)  # source target process

    G_undirected = G.as_undirected()

    for resource, accessing_processes in resource_access.items():
        process_clusters = classify_processes_by_common_ancestor(G, accessing_processes)

        cluster_sizes = {frozenset(cluster): len(cluster) for cluster in process_clusters}

        process_weights = {
            proc: W_base * cluster_sizes[frozenset(cluster)]
            for cluster in process_clusters for proc in cluster
        }

        total_weight = sum(process_weights.values())

        if total_weight > 0:
            process_weights = {proc: weight / total_weight for proc, weight in process_weights.items()}

        # **connecttraverse resource 's edge, weight**
        for edge in G.es:
            source = G. vs [edge.source]["name"]
            target = G. vs [edge.target]["name"]

            if (source == resource and target in accessing_processes) or (
                    target == resource and source in accessing_processes):
                edge["weight"] = process_weights[target] if target in process_weights else process_weights[source]


def set_process_weights(G, W_base=1.0, delta_factor=5):
    """
    process's weight (based onprocessnodedimension) ,  vs toedge: 
    - first eachprocess's  totalweight
    - after  vs normalize, ensureeachentryedge's weightreverseprocess's . 
    """
    process_weights = {}
    total_weights = {}

    for source in G. vs :
        if source["type"] != ObjectType.SUBJECT_PROCESS.value:
            continue

        source_name = source["name"]
        process_weights[source_name] = {}
        total_weights[source_name] = 0

        for target_idx in G.neighbors(source, mode="out"):
            target = G. vs [target_idx]
            if target["type"] != ObjectType.SUBJECT_PROCESS.value or source_name == target["name"]:
                continue

            target_name = target["name"]

            resDepFlag, distance = is_resource_dependent(G, source_name, target_name)

            if resDepFlag:
                delta = delta_factor/ distance
                weight = W_base * (1 + delta)
            else:
                weight = W_base

            process_weights[source_name][target_name] = weight
            total_weights[source_name] += weight

    for edge in G.es:
        source_name = G. vs [edge.source]["name"]
        target_name = G. vs [edge.target]["name"]

        if source_name in process_weights and target_name in process_weights[source_name]:
            if total_weights[source_name] > 0:  # divideby 0
                edge["weight"] = process_weights[source_name][target_name] / total_weights[source_name]
            else:
                edge["weight"] = 0  # iftotalweightis 0, thenweightis 0


def get_connected_processes(G_undirected, start_proc, all_procs):
    """
    takeand start_proc can's hasprocess (become1class) . 

    parameter:
    - G_undirected: withouttowardgraph
    - start_proc: needsfind's startprocess
    - all_procs: sourceconnect's hasprocesslist

    return:
    - set(can's processset)
    """
    try:
        start_idx = G_undirected. vs .find(name=start_proc).index
        reachable_idxs = G_undirected.subcomponent(start_idx)
        reachable_procs = {G_undirected. vs [idx]["name"] for idx in reachable_idxs if
                           G_undirected. vs [idx]["name"] in all_procs}
        return reachable_procs
    except ValueError:
        return set()


def is_related(G, proc1, proc2):
    """
    breaktwoprocessishas (withouttowardpath) : 
    - if proc1 to proc2 existspath (withouttowardpath) , thenisitisprocess. 
    - otherwise, isitiswithoutprocess. 
    """
    try:
        source_idx = G. vs .find(name=proc1).index
        target_idx = G. vs .find(name=proc2).index

        G_undirected = G.as_undirected()

        paths = G_undirected.get_all_shortest_paths(source_idx, to=target_idx)

        return len(paths) > 0
    except ValueError:
        return False


def print_communities(communities):
    """printpartitionresult"""
    for cid, nodes in communities.items():
        print(f"Community {cid}: {nodes}")



def detect_communities(G):
    set_weight(G)

    """use Modularity Method row Leiden detect"""
    # partition = la.find_partition(G, la.CPMVertexPartition, weights='weight', resolution_parameter=0.05)
    partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight')

    communities = {i: [] for i in set(partition.membership)}
    Lcommunities = {i: [] for i in set(partition.membership)}
    for node, community_id in enumerate(partition.membership):
        communities[community_id].append(G. vs [node]["name"])
        Lcommunities[community_id].append((G. vs [node]["name"],G. vs [node]["properties"]))

    print_communities(communities)
    return communities


# def detect_communities_with_Max(G, threshold=100, method="RB", gamma=0.1, max_iter=10):
#     """
#     use Leiden detect, automaticintegerparameter, has <= threshold
#     - threshold: sizeup
#     - method: "RB" | "CPM" | "MOD"
#     - gamma: initialparameter (RB/CPM has)
#     - max_iter: maxiterationtimenumber
#     """
#     set_weight(G)
#
#     communities = None  # beforedeclare, useissue
#
#     for _ in range(max_iter):
#         # choose partition method
#         if method.upper() == "CPM":
#             partition = la.find_partition(
#                 G, la.CPMVertexPartition,
#                 weights="weight",
#                 resolution_parameter=gamma
#             )
#         elif method.upper() == "RB":
#             partition = la.find_partition(
#                 G, la.RBConfigurationVertexPartition,
#                 weights="weight",
#                 resolution_parameter=gamma
#             )
#         elif method.upper() == "MOD":
#             partition = la.find_partition(
#                 G, la.ModularityVertexPartition,
#                 weights="weight"
#             )
#         else:
#             raise ValueError(f"Unknown method {method}, must be one of RB/CPM/MOD")
#
#         communities = defaultdict(list)
#         for node, cid in enumerate(partition.membership):
#             communities[cid].append(G. vs [node]["name"])
#
#         max_size = max(len(c) for c in communities.values())
#         if max_size <= threshold:
#             print_communities(communities)
#             return communities
#
#         gamma *= 1.5
#
#     # if max_iter timestillnotsatisfy
#     print("warning: tomaxiterationtimenumber, stillhasexceedthreshold")
#     print_communities(communities)
#     return communities



def detect_communities_with_max(G, threshold=500, max_depth=2, min_size=2):
    set_weight(G)

    name_to_idx = {G. vs [i]["name"]: i for i in range(G.vcount())}

    def _subgraph_by_names(names):
        idxs = [name_to_idx[n] for n in names if n in name_to_idx]
        return G.subgraph(idxs)

    def _leiden_split(node_names, depth=0):
        sub = _subgraph_by_names(node_names)
        partition = la.find_partition(
            sub, la.ModularityVertexPartition,
            weights="weight",
            n_iterations=-1
        )

        cid2names = defaultdict(list)
        for v_idx, cid in enumerate(partition.membership):
            cid2names[cid].append(sub. vs [v_idx]["name"])

        refined_groups = []
        for names in cid2names.values():
            if len(names) > threshold and depth < max_depth:
                refined_groups.extend(_leiden_split(names, depth + 1))
            else:
                refined_groups.append(names)
        return refined_groups

    all_names = G. vs ["name"]
    groups = _leiden_split(all_names, depth=0)

    # filteronlyhas 1 node's  (orsmallin min_size 's ) 
    groups = [g for g in groups if len(g) >= min_size]

    communities = {i: grp for i, grp in enumerate(groups)}

    if communities:
        print_communities(communities)
    else:
        print("No communities (all groups smaller than min_size).")

    return communities

def detect_communities_with_id(G):
    set_weight(G)
    """use Modularity Method row Leiden detect"""
    # partition = la.find_partition(G, la.CPMVertexPartition, weights='weight', resolution_parameter=0.05)
    partition = la.find_partition(G, la.ModularityVertexPartition, weights='weight')
    communities = {i: [] for i in set(partition.membership)}
    for node, community_id in enumerate(partition.membership):
        communities[community_id].append(G. vs [node])
    print_communities(communities)
    return communities


def set_default_weight(G, weight=1.0):
    """
    setgraphinhasedge's weightis (default 1.0) . 
    """
    G.es["weight"] = [weight] * len(G.es)


def print_graph_info(G):
    """printgraph's andedge's weight"""
    print("Edges with weights:")
    for edge in G.es:
        source = G. vs [edge.source]["name"]
        target = G. vs [edge.target]["name"]
        weight = edge["weight"]
        print(f"{source} -> {target}, Weight: {weight:.4f}")


def get_all_paths(G, source_idx, target_idx, path=None, visited=None, max_depth=10, max_steps=1000, step_counter=[0]):
    """
    userecursivemannerfindhasfrom source_idx to target_idx 's path, maxrecursivedegreeandtotaltimenumber. 

    parameter: 
    - G: igraph graphobject
    - source_idx: startnodeindex
    - target_idx: targetnodeindex
    - path: whenbeforepath (recursiveinternaluse) 
    - visited: whenbefore's nodeset (stopcycle) 
    - max_depth: maxpathdegree
    - max_steps: maxrecursivetrytimenumber (stopstack) 
    - step_counter: fornumberrecursivetimenumber's list (notcanclass) 

    return: 
    - haspath's list, eachpathis1nodeindexlist
    """
    if path is None:
        path = []
    if visited is None:
        visited = set()

    if step_counter[0] >= max_steps:
        return []

    step_counter[0] += 1

    if source_idx in visited:
        return []

    if len(path) > max_depth:
        return []

    path.append(source_idx)
    visited.add(source_idx)

    all_paths = []
    if source_idx == target_idx:
        all_paths.append(path[:])
    else:
        for neighbor in G.neighbors(source_idx, mode="all"):
            all_paths.extend(
                get_all_paths(G, neighbor, target_idx, path, visited, max_depth, max_steps, step_counter)
            )

    path.pop()
    visited.remove(source_idx)

    return all_paths


def find_ancestors(G, proc):
    """
    findprocess's hasfirst  (hastowardup's process) . 

    parameter:
    - G: igraph hastowardgraph
    - proc: processname

    return:
    - ancestors: first processset
    """
    try:
        proc_idx = G. vs .find(name=proc).index
    except ValueError:
        return set()

    ancestors = set()
    queue = [proc_idx]

    while queue:
        current = queue.pop(0)
        for parent in G.predecessors(current):
            parent_name = G. vs [parent]["name"]
            if parent_name not in ancestors:
                ancestors.add(parent_name)
                queue.append(parent)

    return ancestors

def classify_processes_by_common_ancestor(G, accessing_processes):
    """
    according tofirst process, tosamesource's processrowclass. 

    parameter:
    - G: igraph hastowardgraph
    - accessing_processes: somesource's processset

    return:
    - process_clusters: processclass's list, eachclassis1set
    """
    process_clusters = []
    visited = set()

    ancestor_map = {proc: find_ancestors(G, proc) for proc in accessing_processes}

    # **traverse accessing_processes**
    for proc in accessing_processes:
        if proc in visited:
            continue

        proc_ancestors = ancestor_map[proc]

        cluster = {p for p in accessing_processes if not proc_ancestors.isdisjoint(ancestor_map[p])}
        cluster.add(proc)
        process_clusters.append(cluster)
        visited.update(cluster)

    return process_clusters




