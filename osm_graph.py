from __future__ import annotations

import networkx as nx
import osmnx as ox
import streamlit as st

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 30  # fail fast instead of hanging if Overpass is slow/unreachable

AMENITY_TAGS = {
    "amenity": ["hospital", "police", "fire_station"],
    "aeroway": ["aerodrome"],
    "emergency": ["assembly_point", "shelter"],
}

# A full administrative-boundary graph for a metro like Mumbai or Delhi is
# hundreds of thousands of edges and routinely times out against the public
# Overpass API. A radius around a center point keeps downloads fast and
# reliable for a live routing demo, while still being 100% real OSM data.
CITY_CENTERS: dict[str, tuple[float, float]] = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
}


@st.cache_resource(show_spinner="Downloading real OpenStreetMap road network for the area...")
def load_city_graph(city_name: str, radius_m: int = 6000, network_type: str = "drive") -> nx.MultiDiGraph:
    """Download (and cache for the app session) a real, routable OSM road network
    within `radius_m` metres of the city's center point.

    Requires outbound internet access to the OpenStreetMap Overpass API, which
    is available on Streamlit Community Cloud by default.
    """
    center = CITY_CENTERS.get(city_name)
    if center is None:
        raise ValueError(f"Unknown city: {city_name}")
    graph = ox.graph_from_point(center, dist=radius_m, network_type=network_type, simplify=True)
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


@st.cache_resource(show_spinner="Loading hospitals, police stations, airports, and shelters...")
def load_city_amenities(city_name: str, radius_m: int = 6000):
    """Return a GeoDataFrame of emergency-relevant points of interest, or None on failure."""
    center = CITY_CENTERS.get(city_name)
    if center is None:
        return None
    try:
        gdf = ox.features_from_point(center, AMENITY_TAGS, dist=radius_m)
        return gdf
    except Exception:
        return None


def road_names(graph: nx.MultiDiGraph) -> list[str]:
    """Unique, human-readable street names available for the disaster simulator."""
    names: set[str] = set()
    for _, _, data in graph.edges(data=True):
        name = data.get("name")
        if isinstance(name, list):
            names.update(str(n) for n in name if n)
        elif name:
            names.add(str(name))
    return sorted(names)


def bridge_road_names(graph: nx.MultiDiGraph) -> list[str]:
    """Streets tagged as bridges/viaducts in OSM -- used to auto-suggest closures
    for the 'Bridge Collapse' disaster scenario."""
    names: set[str] = set()
    for _, _, data in graph.edges(data=True):
        if str(data.get("bridge", "")).lower() in {"yes", "viaduct"}:
            name = data.get("name")
            if isinstance(name, list):
                names.update(str(n) for n in name if n)
            elif name:
                names.add(str(name))
    return sorted(names)
