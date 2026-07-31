from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, degrees, hypot
from typing import Iterable

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from skimage.morphology import closing, disk, skeletonize


Point = tuple[float, float]


@dataclass(frozen=True)
class DemoScene:
    image: np.ndarray
    ground_truth_mask: np.ndarray
    broken_mask: np.ndarray
    occlusion_mask: np.ndarray
    center_lat: float = 12.9716
    center_lon: float = 77.5946
    meters_per_pixel: float = 2.8


@dataclass(frozen=True)
class GraphBundle:
    raw_graph: nx.Graph
    healed_graph: nx.Graph
    healed_mask: np.ndarray
    raw_components: int
    healed_components: int
    bridged_edges: list[tuple[int, int]]


def make_demo_scene(size: int = 640, occlusion_strength: float = 0.55, seed: int = 7) -> DemoScene:
    """Create a Bengaluru-like satellite tile with roads hidden by tree/shadow occlusions."""
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:, :] = (68, 86, 70)

    noise = rng.normal(0, 8, base.shape).astype(np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Built-up parcels.
    for _ in range(90):
        x, y = rng.integers(0, size - 50, 2)
        w, h = rng.integers(18, 70, 2)
        color = tuple(int(v) for v in rng.integers([85, 80, 72], [160, 150, 135]))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)

    road_lines = [
        [(40, 95), (200, 120), (420, 115), (604, 135)],
        [(30, 295), (185, 260), (350, 282), (610, 250)],
        [(55, 500), (220, 455), (365, 470), (600, 420)],
        [(120, 30), (130, 210), (145, 420), (160, 615)],
        [(310, 25), (318, 190), (325, 385), (335, 610)],
        [(515, 35), (490, 180), (505, 370), (475, 605)],
        [(70, 610), (215, 430), (370, 235), (560, 75)],
        [(45, 380), (205, 325), (390, 330), (610, 315)],
    ]

    ground_truth = np.zeros((size, size), dtype=np.uint8)
    for line in road_lines:
        pts = np.array(line, dtype=np.int32)
        cv2.polylines(ground_truth, [pts], False, 255, thickness=15, lineType=cv2.LINE_AA)
        cv2.polylines(image, [pts], False, (185, 184, 170), thickness=17, lineType=cv2.LINE_AA)
        cv2.polylines(image, [pts], False, (118, 118, 112), thickness=2, lineType=cv2.LINE_AA)

    occlusion = np.zeros((size, size), dtype=np.uint8)
    count = int(18 + 25 * occlusion_strength)
    for _ in range(count):
        x, y = rng.integers(30, size - 90, 2)
        rx = int(rng.integers(22, 70) * (0.65 + occlusion_strength))
        ry = int(rng.integers(12, 45) * (0.65 + occlusion_strength))
        angle = float(rng.integers(0, 180))
        cv2.ellipse(occlusion, (int(x), int(y)), (rx, ry), angle, 0, 360, 255, -1)

    shadow = cv2.GaussianBlur(occlusion, (35, 35), 0)
    tree_color = np.zeros_like(image)
    tree_color[:, :] = (35, 75, 40)
    alpha = (shadow.astype(np.float32) / 255.0 * 0.72)[..., None]
    image = np.clip(image * (1 - alpha) + tree_color * alpha, 0, 255).astype(np.uint8)

    broken = ground_truth.copy()
    broken[occlusion > 35] = 0
    broken = cv2.morphologyEx(broken, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return DemoScene(image=image, ground_truth_mask=ground_truth, broken_mask=broken, occlusion_mask=occlusion)


def estimate_road_mask_from_rgb(image: np.ndarray) -> np.ndarray:
    """Simple non-ML fallback for uploaded images: detect bright/neutral linear surfaces."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
    edges = cv2.Canny(light, 50, 140)
    neutral = ((light > 118) & (saturation < 76)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(neutral, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return (mask > 0).astype(np.uint8) * 255


def _node_positions_from_skeleton(skel: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    padded = np.pad(skel.astype(np.uint8), 1)
    positions: list[tuple[int, int]] = []
    node_id = np.full(skel.shape, -1, dtype=np.int32)
    for y, x in zip(*np.nonzero(skel)):
        window = padded[y : y + 3, x : x + 3]
        degree = int(window.sum() - 1)
        if degree != 2:
            node_id[y, x] = len(positions)
            positions.append((int(x), int(y)))
    return positions, node_id


def _neighbors(y: int, x: int, shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    height, width = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx_ < width:
                yield ny, nx_


def mask_to_graph(mask: np.ndarray, min_branch_pixels: int = 8) -> nx.Graph:
    """Skeletonize a binary road mask and convert it into an endpoint/intersection graph."""
    closed = closing(mask > 0, disk(2))
    skel = skeletonize(closed)
    positions, node_id = _node_positions_from_skeleton(skel)
    graph = nx.Graph()

    if not positions:
        return graph

    for idx, (x, y) in enumerate(positions):
        graph.add_node(idx, x=float(x), y=float(y), pos=(float(x), float(y)))

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for node_idx, (x, y) in enumerate(positions):
        for ny, nx_ in _neighbors(y, x, skel.shape):
            if not skel[ny, nx_]:
                continue
            key = tuple(sorted(((y, x), (ny, nx_))))
            if key in visited_edges:
                continue

            path = [(y, x), (ny, nx_)]
            visited_edges.add(key)
            prev = (y, x)
            cur = (ny, nx_)
            end_node = node_id[cur]

            while end_node < 0:
                next_pixels = [
                    pixel
                    for pixel in _neighbors(cur[0], cur[1], skel.shape)
                    if skel[pixel] and pixel != prev
                ]
                if not next_pixels:
                    break
                nxt = next_pixels[0]
                visited_edges.add(tuple(sorted((cur, nxt))))
                path.append(nxt)
                prev, cur = cur, nxt
                end_node = node_id[cur]

            if end_node >= 0 and end_node != node_idx and len(path) >= min_branch_pixels:
                pts = np.array([(px, py) for py, px in path], dtype=np.float32)
                length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                graph.add_edge(
                    node_idx,
                    int(end_node),
                    weight=max(length, 1.0),
                    geometry=pts.tolist(),
                    healed=False,
                )

    return graph


def _edge_heading(graph: nx.Graph, node: int) -> float | None:
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return None
    x1, y1 = graph.nodes[node]["pos"]
    headings = []
    for nbr in neighbors[:3]:
        x2, y2 = graph.nodes[nbr]["pos"]
        headings.append(degrees(atan2(y2 - y1, x2 - x1)))
    return float(np.mean(headings))


def _angle_delta(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    delta = abs((a - b + 180) % 360 - 180)
    return min(delta, abs(180 - delta))


def heal_graph(graph: nx.Graph, max_gap: float = 58.0, angle_tolerance: float = 42.0) -> tuple[nx.Graph, list[tuple[int, int]]]:
    """Bridge likely road gaps between disconnected components using distance and heading."""
    healed = graph.copy()
    if healed.number_of_nodes() < 2:
        return healed, []

    parent = {node: node for node in healed.nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for component in nx.connected_components(healed):
        root = next(iter(component))
        for node in component:
            parent[node] = root

    endpoints = [node for node, degree in healed.degree if degree <= 1]
    candidates = []
    for u, v in combinations(endpoints, 2):
        if find(u) == find(v):
            continue
        x1, y1 = healed.nodes[u]["pos"]
        x2, y2 = healed.nodes[v]["pos"]
        distance = hypot(x2 - x1, y2 - y1)
        if distance > max_gap:
            continue
        angle_delta = _angle_delta(_edge_heading(healed, u), _edge_heading(healed, v))
        if angle_delta <= angle_tolerance:
            candidates.append((distance + angle_delta * 0.65, distance, u, v))

    bridged: list[tuple[int, int]] = []
    for _, distance, u, v in sorted(candidates):
        if find(u) != find(v):
            x1, y1 = healed.nodes[u]["pos"]
            x2, y2 = healed.nodes[v]["pos"]
            healed.add_edge(
                u,
                v,
                weight=max(float(distance), 1.0),
                geometry=[(float(x1), float(y1)), (float(x2), float(y2))],
                healed=True,
            )
            union(u, v)
            bridged.append((u, v))

    return healed, bridged


def graph_to_mask(graph: nx.Graph, shape: tuple[int, int], thickness: int = 7) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for u, v, data in graph.edges(data=True):
        pts = data.get("geometry")
        if not pts:
            pts = [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
        arr = np.array([(int(round(x)), int(round(y))) for x, y in pts], dtype=np.int32)
        cv2.polylines(mask, [arr], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return mask


def build_graph_bundle(mask: np.ndarray, max_gap: float, angle_tolerance: float) -> GraphBundle:
    raw_graph = mask_to_graph(mask)
    healed_graph, bridged = heal_graph(raw_graph, max_gap=max_gap, angle_tolerance=angle_tolerance)
    healed_mask = graph_to_mask(healed_graph, mask.shape)
    raw_components = nx.number_connected_components(raw_graph) if raw_graph.number_of_nodes() else 0
    healed_components = nx.number_connected_components(healed_graph) if healed_graph.number_of_nodes() else 0
    return GraphBundle(
        raw_graph=raw_graph,
        healed_graph=healed_graph,
        healed_mask=healed_mask,
        raw_components=raw_components,
        healed_components=healed_components,
        bridged_edges=bridged,
    )


def segmentation_metrics(pred_mask: np.ndarray, truth_mask: np.ndarray, occlusion_mask: np.ndarray | None = None) -> dict[str, float]:
    pred = pred_mask > 0
    truth = truth_mask > 0
    intersection = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    pred_sum = float(pred.sum())
    truth_sum = float(truth.sum())
    dice = 2 * intersection / max(pred_sum + truth_sum, 1.0)
    iou = intersection / max(union, 1.0)
    precision = intersection / max(pred_sum, 1.0)
    recall = intersection / max(truth_sum, 1.0)

    metrics = {"IoU": iou, "Dice": dice, "Precision": precision, "Recall": recall}
    if occlusion_mask is not None:
        occ_truth = truth & (occlusion_mask > 0)
        occ_hit = np.logical_and(pred, occ_truth).sum()
        metrics["Occlusion recall"] = float(occ_hit) / max(float(occ_truth.sum()), 1.0)
    return metrics


def centrality_scores(graph: nx.Graph) -> dict[int, float]:
    if graph.number_of_nodes() < 3:
        return {node: 0.0 for node in graph.nodes}
    return nx.betweenness_centrality(graph, weight="weight", normalized=True)


def average_network_cost(graph: nx.Graph, reference_nodes: list[int] | None = None, penalty: float | None = None) -> float:
    nodes = reference_nodes or list(graph.nodes)
    if len(nodes) < 2:
        return float("inf")

    if penalty is None:
        if graph.number_of_nodes():
            xs = [graph.nodes[node]["x"] for node in graph.nodes]
            ys = [graph.nodes[node]["y"] for node in graph.nodes]
            penalty = hypot(max(xs) - min(xs), max(ys) - min(ys)) * 4
        else:
            penalty = 10_000.0

    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    total = 0.0
    pairs = 0
    for u, v in combinations(nodes, 2):
        if u in graph and v in graph:
            total += float(lengths.get(u, {}).get(v, penalty))
        else:
            total += float(penalty)
        pairs += 1
    return total / max(pairs, 1)


def stress_test(graph: nx.Graph, disabled_nodes: list[int]) -> dict[str, float]:
    reference_nodes = list(graph.nodes)
    if graph.number_of_nodes():
        xs = [graph.nodes[node]["x"] for node in graph.nodes]
        ys = [graph.nodes[node]["y"] for node in graph.nodes]
        penalty = hypot(max(xs) - min(xs), max(ys) - min(ys)) * 4
    else:
        penalty = 10_000.0

    baseline = average_network_cost(graph, reference_nodes, penalty)
    perturbed = graph.copy()
    perturbed.remove_nodes_from([node for node in disabled_nodes if node in perturbed])
    damaged = average_network_cost(perturbed, reference_nodes, penalty)
    resilience = baseline / damaged if np.isfinite(baseline) and np.isfinite(damaged) and damaged > 0 else 0.0
    base_components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    damaged_components = nx.number_connected_components(perturbed) if perturbed.number_of_nodes() else 0
    return {
        "baseline_path": float(baseline),
        "damaged_path": float(damaged),
        "resilience_index": float(resilience),
        "component_delta": float(damaged_components - base_components),
        "remaining_nodes": float(perturbed.number_of_nodes()),
    }


def nearest_node_to_xy(graph: nx.Graph, x: float, y: float) -> int | None:
    if graph.number_of_nodes() == 0:
        return None
    return min(
        graph.nodes,
        key=lambda node: hypot(graph.nodes[node]["x"] - x, graph.nodes[node]["y"] - y),
    )


def path_length(graph: nx.Graph, path: list[int] | None) -> float:
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += float(graph.edges[u, v].get("weight", 1.0))
    return total


def shortest_route(graph: nx.Graph, start_node: int | None, end_node: int | None) -> tuple[list[int] | None, float]:
    if start_node is None or end_node is None or start_node not in graph or end_node not in graph:
        return None, float("inf")
    if start_node == end_node:
        return [start_node], 0.0
    try:
        path = nx.shortest_path(graph, start_node, end_node, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")
    return path, path_length(graph, path)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    out = image.copy()
    colored = np.zeros_like(out)
    colored[:, :] = color
    active = mask > 0
    out[active] = np.clip(out[active] * (1 - alpha) + colored[active] * alpha, 0, 255)
    return out.astype(np.uint8)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return (mask > 127).astype(np.uint8) * 255
