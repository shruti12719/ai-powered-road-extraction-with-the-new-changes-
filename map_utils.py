from __future__ import annotations

import folium
import networkx as nx

from icons import icon_for, label_for

TILE_OPTIONS: dict[str, dict[str, str | None]] = {
    "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": None},
    "Carto Voyager": {
        "tiles": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "attr": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "Esri World Imagery (Satellite)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
    },
}


def _best_edge(graph: nx.MultiDiGraph, u: int, v: int, k: int | None = None) -> dict:
    data = graph.get_edge_data(u, v)
    if not data:
        return {}
    if k is not None and k in data:
        return data[k]
    return min(data.values(), key=lambda d: d.get("length", float("inf")))


def _edge_coords(graph: nx.MultiDiGraph, u: int, v: int, k: int | None = None) -> list[tuple[float, float]]:
    data = _best_edge(graph, u, v, k)
    geom = data.get("geometry")
    if geom is not None:
        return [(lat, lon) for lon, lat in geom.coords]
    return [(graph.nodes[u]["y"], graph.nodes[u]["x"]), (graph.nodes[v]["y"], graph.nodes[v]["x"])]


def _path_coords(graph: nx.MultiDiGraph, path: list[int]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for u, v in zip(path[:-1], path[1:]):
        coords.extend(_edge_coords(graph, u, v))
    return coords


def build_emergency_map(
    graph: nx.MultiDiGraph,
    center: tuple[float, float],
    tile_choice: str,
    start_point: tuple[float, float] | None = None,
    end_point: tuple[float, float] | None = None,
    baseline_path: list[int] | None = None,
    emergency_path: list[int] | None = None,
    blocked_edges: list[tuple[int, int, int]] | None = None,
    amenities_gdf=None,
    zoom: int = 14,
) -> folium.Map:
    tile_cfg = TILE_OPTIONS.get(tile_choice, TILE_OPTIONS["OpenStreetMap"])
    fmap = folium.Map(location=center, zoom_start=zoom, tiles=tile_cfg["tiles"], attr=tile_cfg["attr"], control_scale=True)

    if baseline_path and len(baseline_path) > 1:
        folium.PolyLine(
            _path_coords(graph, baseline_path),
            color="#1455d9",
            weight=6,
            opacity=0.85,
            tooltip="Normal route",
        ).add_to(fmap)

    if emergency_path and len(emergency_path) > 1:
        folium.PolyLine(
            _path_coords(graph, emergency_path),
            color="#12a150",
            weight=6,
            opacity=0.95,
            tooltip="Emergency route",
        ).add_to(fmap)

    if blocked_edges:
        for u, v, k in blocked_edges:
            try:
                coords = _edge_coords(graph, u, v, k)
            except Exception:
                continue
            folium.PolyLine(
                coords,
                color="#dd2c2c",
                weight=5,
                opacity=0.9,
                dash_array="6,6",
                tooltip="Blocked road",
            ).add_to(fmap)

    if start_point:
        folium.Marker(start_point, tooltip="Start", icon=folium.Icon(color="blue", icon="play", prefix="fa")).add_to(fmap)
    if end_point:
        folium.Marker(end_point, tooltip="Destination", icon=folium.Icon(color="green", icon="flag", prefix="fa")).add_to(fmap)

    if amenities_gdf is not None and not amenities_gdf.empty:
        for _, row in amenities_gdf.iterrows():
            try:
                geom = row.geometry
                point = geom.centroid if geom.geom_type != "Point" else geom
                amenity_key = row.get("amenity") or row.get("aeroway") or row.get("emergency")
                _, color, icon_name = icon_for(amenity_key)
                folium.Marker(
                    (point.y, point.x),
                    tooltip=label_for(row),
                    icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
                ).add_to(fmap)
            except Exception:
                continue

    return fmap
