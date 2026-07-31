from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import osmnx as ox


@dataclass(frozen=True)
class RouteResult:
    path: list[int] | None
    length_m: float
    time_min: float


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int | None:
    if graph.number_of_nodes() == 0:
        return None
    return int(ox.distance.nearest_nodes(graph, X=lon, Y=lat))


def _best_edge(graph: nx.MultiDiGraph, u: int, v: int) -> dict:
    data = graph.get_edge_data(u, v)
    if not data:
        return {}
    return min(data.values(), key=lambda d: d.get("length", float("inf")))


def path_distance_time(graph: nx.MultiDiGraph, path: list[int] | None) -> tuple[float, float]:
    if not path or len(path) < 2:
        return 0.0, 0.0
    length_m = 0.0
    time_s = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = _best_edge(graph, u, v)
        length_m += float(edge.get("length", 0.0))
        time_s += float(edge.get("travel_time", 0.0))
    return length_m, time_s


def shortest_route(graph: nx.MultiDiGraph, orig: int | None, dest: int | None) -> RouteResult:
    if orig is None or dest is None or orig not in graph or dest not in graph:
        return RouteResult(None, float("inf"), float("inf"))
    if orig == dest:
        return RouteResult([orig], 0.0, 0.0)
    try:
        path = nx.shortest_path(graph, orig, dest, weight="travel_time")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return RouteResult(None, float("inf"), float("inf"))
    length_m, time_s = path_distance_time(graph, path)
    return RouteResult(path, length_m, time_s / 60.0)

def shortest_route_with_disabled_nodes(
    graph,
    orig,
    dest,
    disabled_nodes,
):

    G = graph.copy()

    G.remove_nodes_from(disabled_nodes)

    try:
        path = nx.shortest_path(
            G,
            orig,
            dest,
            weight="travel_time",
        )

    except (nx.NetworkXNoPath, nx.NodeNotFound):

        return RouteResult(
            None,
            float("inf"),
            float("inf"),
        )

    length, time = path_distance_time(G, path)

    return RouteResult(
        path,
        length,
        time / 60,
    )


def block_roads_by_name(graph: nx.MultiDiGraph, blocked_names: list[str]) -> tuple[nx.MultiDiGraph, list[tuple[int, int, int]]]:
    """Return a damaged copy of the graph with every edge matching a blocked
    street name removed (case-insensitive substring match), plus the list of
    removed (u, v, key) edges so they can be drawn in red on the map."""
    if not blocked_names:
        return graph.copy(), []

    targets = [name.strip().lower() for name in blocked_names if name.strip()]
    damaged = graph.copy()
    removed: list[tuple[int, int, int]] = []

    for u, v, k, data in list(graph.edges(keys=True, data=True)):
        name = data.get("name")
        candidates = name if isinstance(name, list) else [name] if name else []
        candidates_lower = [str(c).lower() for c in candidates]
        if any(target in candidate for target in targets for candidate in candidates_lower):
            if damaged.has_edge(u, v, k):
                damaged.remove_edge(u, v, k)
                removed.append((u, v, k))

    return damaged, removed


def critical_roads(graph: nx.MultiDiGraph, sample_k: int = 60, top_n: int = 8) -> list[tuple[str, float]]:
    """Approximate the most systemically important named roads using sampled
    edge betweenness centrality (full centrality is too slow for real city graphs)."""
    simple_graph = nx.Graph(graph)
    if simple_graph.number_of_nodes() < 3:
        return []
    k = min(sample_k, simple_graph.number_of_nodes())
    try:
        centrality = nx.edge_betweenness_centrality(simple_graph, k=k, weight="length", seed=7)
    except Exception:
        return []

    scored: dict[str, float] = {}
    for (u, v), score in centrality.items():
        edge = _best_edge(graph, u, v) or _best_edge(graph, v, u)
        name = edge.get("name") if edge else None
        if isinstance(name, list):
            name = name[0] if name else None
        if not name:
            continue
        scored[str(name)] = max(scored.get(str(name), 0.0), float(score))

    return sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def network_resilience_index(
    graph: nx.MultiDiGraph,
    damaged_graph: nx.MultiDiGraph,
    sample_size: int = 25,
    seed: int = 7,
) -> float:
    """Sampled network-wide resilience score: baseline average travel cost
    divided by post-disaster average travel cost across a random set of node
    pairs. 1.0 = no degradation, lower = more degraded. All-pairs Dijkstra is
    too slow for real city-scale graphs, hence the sampling."""
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return 1.0
    rng = random.Random(seed)
    sample = rng.sample(nodes, min(sample_size, len(nodes)))
    penalty = 3600.0  # 1 hour, treated as "effectively unreachable"

    def avg_cost(g: nx.MultiDiGraph) -> float:
        total, pairs = 0.0, 0
        for i, u in enumerate(sample):
            for v in sample[i + 1 :]:
                try:
                    total += nx.shortest_path_length(g, u, v, weight="travel_time")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    total += penalty
                pairs += 1
        return total / max(pairs, 1)

    baseline = avg_cost(graph)
    damaged = avg_cost(damaged_graph)
    if damaged <= 0:
        return 0.0
    return float(baseline / damaged)
