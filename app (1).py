from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import folium
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pipeline import (
    build_graph_bundle,
    centrality_scores,
    ensure_uint8_mask,
    estimate_road_mask_from_rgb,
    make_demo_scene,
    nearest_node_to_xy,
    overlay_mask,
    read_uploaded_image,
    segmentation_metrics,
    shortest_route as synthetic_shortest_route,
    stress_test,
)
from geocoder import Place, geocode_one, reverse_geocode, search_places
from icons import DISASTER_ICONS
from map_utils import TILE_OPTIONS, build_emergency_map
from osm_graph import CITY_CENTERS, bridge_road_names, load_city_amenities, load_city_graph, road_names
from report import build_pdf_report, generate_ai_summary
from routing import (
    RouteResult,
    block_roads_by_name,
    critical_roads,
    nearest_node,
    network_resilience_index,
    shortest_route,
)


APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Synthetic-demo city presets (used only by the Satellite Image extraction tab)
# ---------------------------------------------------------------------------
CITY_PRESETS = {
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Guwahati": (26.1445, 91.7362),
    "Custom": (12.9716, 77.5946),
}

# Curated landmark shortcuts for the Emergency Planner "quick pick" widget.
# These are just search queries -- they are still geocoded live via Nominatim,
# nothing here is hardcoded as a coordinate.
QUICK_PICKS: dict[str, dict[str, str]] = {
    "Mumbai": {
        "Kokilaben Hospital": "Kokilaben Dhirubhai Ambani Hospital, Mumbai",
        "Bandra Railway Station": "Bandra Railway Station, Mumbai",
        "Chhatrapati Shivaji Airport": "Chhatrapati Shivaji Maharaj International Airport, Mumbai",
        "Mumbai Police HQ": "Mumbai Police Commissioner Office, Mumbai",
    },
    "Delhi": {
        "AIIMS Hospital": "All India Institute of Medical Sciences, Delhi",
        "New Delhi Railway Station": "New Delhi Railway Station",
        "IGI Airport": "Indira Gandhi International Airport, Delhi",
        "Delhi Police HQ": "Delhi Police Headquarters",
    },
    "Bengaluru": {
        "Victoria Hospital": "Victoria Hospital, Bengaluru",
        "KSR Railway Station": "KSR Bengaluru City Railway Station",
        "Kempegowda Airport": "Kempegowda International Airport, Bengaluru",
        "Bengaluru Police HQ": "Bengaluru City Police Commissionerate",
    },
}


st.set_page_config(
    page_title="Route Resilience AI",
    page_icon="RR",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #0e1a17;
        --ink: #1b2722;
        --muted: #52615a;
        --line: #dde3dc;
        --accent: #dd563b;
        --accent-2: #24746b;
        --panel: #ffffff;
    }
    .main .block-container {
        padding-top: 1.2rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        letter-spacing: 0 !important;
        color: var(--ink);
    }
    .rr-hero {
        background: linear-gradient(120deg, #0e1a17 0%, #16342c 55%, #24746b 100%);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 14px;
        color: #f4f7f2;
        box-shadow: 0 6px 18px rgba(14, 26, 23, 0.28);
    }
    .rr-hero h1 {
        color: #ffffff;
        font-size: 2.1rem;
        margin: 0;
        letter-spacing: 0.02em !important;
    }
    .rr-hero p {
        color: #cfe3da;
        margin: 6px 0 0;
        font-size: 0.98rem;
    }
    .rr-banner {
        background: #fff4ef;
        border: 1px solid #f0c3b3;
        border-left: 5px solid var(--accent);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0 16px;
        color: #5c2d1d;
        font-size: 0.94rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cbd6ce;
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #33423b;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        color: #14211b;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 10px;
        margin: 16px 0 18px;
    }
    .kpi-card {
        min-height: 112px;
        background: #ffffff;
        border: 1px solid #cbd6ce;
        border-top: 5px solid var(--accent-2);
        border-radius: 8px;
        padding: 12px 12px 10px;
        box-shadow: 0 1px 2px rgba(27, 39, 34, 0.08);
    }
    .kpi-title {
        color: #31423b;
        font-size: 0.76rem;
        line-height: 1.15;
        text-transform: uppercase;
        font-weight: 700;
    }
    .kpi-value {
        color: #13211b;
        font-size: 1.72rem;
        line-height: 1.2;
        font-weight: 800;
        margin-top: 5px;
    }
    .kpi-help {
        color: #5c6a62;
        font-size: 0.78rem;
        line-height: 1.25;
        margin-top: 4px;
    }
    .route-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 16px;
    }
    .route-card {
        background: #ffffff;
        border: 1px solid #cbd6ce;
        border-radius: 8px;
        padding: 14px;
        min-height: 124px;
    }
    .route-card strong {
        color: #18261f;
        display: block;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .route-card span {
        color: #18261f;
        font-size: 1.45rem;
        font-weight: 800;
    }
    .route-card p {
        color: #5c6a62;
        margin: 6px 0 0;
        font-size: 0.86rem;
        line-height: 1.32;
    }
    .pipeline-flow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
    }
    .pipeline-step {
        background: #16342c;
        border: 1px solid #cbd6ce;
        border-left: 5px solid #dd563b;
        border-radius: 8px;
        padding: 12px;
        min-height: 86px;
    }
    .pipeline-step b {
        color: #ffffff !important;
        display: block;
        font-size: 0.92rem;
    }
    .pipeline-step small {
        color: #f0f0f0 !important;
        display: block;
        margin-top: 5px;
        line-height: 1.25;
    }
    @media (max-width: 1100px) {
        .kpi-grid, .pipeline-flow {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .route-grid {
            grid-template-columns: 1fr;
        }
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }
    .rr-caption {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: -0.5rem;
    }
    .rr-status {
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fbfcf8;
        color: var(--ink);
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Synthetic-demo helper functions (Satellite Image tab only)
# ---------------------------------------------------------------------------
def node_to_latlon(x: float, y: float, width: int, height: int, center_lat: float, center_lon: float, meters_per_pixel: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(center_lat))
    lat = center_lat - ((y - height / 2) * meters_per_pixel) / meters_per_deg_lat
    lon = center_lon + ((x - width / 2) * meters_per_pixel) / meters_per_deg_lon
    return float(lat), float(lon)


def latlon_to_xy(lat: float, lon: float, width: int, height: int, center_lat: float, center_lon: float, meters_per_pixel: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(center_lat))
    x = width / 2 + ((lon - center_lon) * meters_per_deg_lon) / meters_per_pixel
    y = height / 2 - ((lat - center_lat) * meters_per_deg_lat) / meters_per_pixel
    return float(x), float(y)


def criticality_color(value: float, vmax: float) -> str:
    ratio = 0 if vmax <= 0 else min(max(value / vmax, 0), 1)
    if ratio > 0.72:
        return "#d73027"
    if ratio > 0.42:
        return "#fc8d59"
    if ratio > 0.18:
        return "#fee08b"
    return "#2c7bb6"


def path_to_latlon(graph: nx.Graph, path: list[int] | None, center: tuple[float, float], image_shape: tuple[int, int], meters_per_pixel: float) -> list[tuple[float, float]]:
    if not path:
        return []
    height, width = image_shape
    return [
        node_to_latlon(graph.nodes[node]["x"], graph.nodes[node]["y"], width, height, center[0], center[1], meters_per_pixel)
        for node in path
        if node in graph
    ]


def format_distance(px_length: float, meters_per_pixel: float) -> str:
    if not np.isfinite(px_length):
        return "No route"
    km = px_length * meters_per_pixel / 1000.0
    return f"{km:.2f} km"


def kpi_cards(items: list[dict[str, str]]) -> None:
    html = "<div class='kpi-grid'>"
    for item in items:
        html += (
            f"<div class='kpi-card' style='border-top-color:{item.get('color', '#24746b')}'>"
            f"<div class='kpi-title'>{item['title']}</div>"
            f"<div class='kpi-value'>{item['value']}</div>"
            f"<div class='kpi-help'>{item['help']}</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def route_cards(items: list[dict[str, str]]) -> None:
    html = "<div class='route-grid'>"
    for item in items:
        html += (
            f"<div class='route-card' style='border-top:5px solid {item.get('color', '#24746b')}'>"
            f"<strong>{item['title']}</strong>"
            f"<span>{item['value']}</span>"
            f"<p>{item['help']}</p>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def build_synthetic_map(
    graph: nx.Graph,
    scores: dict[int, float],
    disabled_nodes: list[int],
    center: tuple[float, float],
    image_shape: tuple[int, int],
    meters_per_pixel: float,
    start_node: int | None = None,
    end_node: int | None = None,
    baseline_path: list[int] | None = None,
    emergency_path: list[int] | None = None,
) -> folium.Map:
    height, width = image_shape
    fmap = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron", control_scale=True)
    vmax = max(scores.values()) if scores else 0.0
    disabled = set(disabled_nodes)

    for u, v, data in graph.edges(data=True):
        pts = data.get("geometry") or [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
        latlon = [
            node_to_latlon(x, y, width, height, center[0], center[1], meters_per_pixel)
            for x, y in pts
        ]
        edge_score = max(scores.get(u, 0.0), scores.get(v, 0.0))
        color = "#a23b72" if data.get("healed") else criticality_color(edge_score, vmax)
        folium.PolyLine(
            latlon,
            color=color,
            weight=5 if data.get("healed") else 4,
            opacity=0.82,
            tooltip=f"criticality={edge_score:.3f} | healed={bool(data.get('healed'))}",
        ).add_to(fmap)

    if baseline_path:
        folium.PolyLine(
            path_to_latlon(graph, baseline_path, center, image_shape, meters_per_pixel),
            color="#1455d9",
            weight=7,
            opacity=0.86,
            tooltip="Shortest route before disaster",
        ).add_to(fmap)

    if emergency_path:
        folium.PolyLine(
            path_to_latlon(graph, emergency_path, center, image_shape, meters_per_pixel),
            color="#12a150",
            weight=7,
            opacity=0.94,
            tooltip="Emergency alternate route after disaster",
        ).add_to(fmap)

    for node, score in scores.items():
        x, y = graph.nodes[node]["pos"]
        lat, lon = node_to_latlon(x, y, width, height, center[0], center[1], meters_per_pixel)
        is_disabled = node in disabled
        is_terminal = node == start_node or node == end_node
        folium.CircleMarker(
            location=(lat, lon),
            radius=9 if is_terminal else 7 if is_disabled else 4 + min(score * 30, 7),
            color="#1455d9" if node == start_node else "#12a150" if node == end_node else "#111111" if is_disabled else "#1b2722",
            fill=True,
            fill_color="#1455d9" if node == start_node else "#12a150" if node == end_node else "#111111" if is_disabled else criticality_color(score, vmax),
            fill_opacity=0.95,
            tooltip=f"node {node} | betweenness={score:.4f}",
        ).add_to(fmap)

    return fmap


def to_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Emergency Planner helpers (real OpenStreetMap data)
# ---------------------------------------------------------------------------
def resolve_current_location() -> Place | None:
    try:
        from streamlit_js_eval import get_geolocation
    except ImportError:
        st.warning(
            "'Use current location' needs the `streamlit-js-eval` package (already listed in "
            "requirements.txt) -- install it and redeploy to enable this button."
        )
        return None
    loc = get_geolocation()
    if not loc or "coords" not in loc:
        st.info("Waiting for the browser to share your location -- allow the permission prompt and click again.")
        return None
    lat = float(loc["coords"]["latitude"])
    lon = float(loc["coords"]["longitude"])
    name = reverse_geocode(lat, lon)
    return Place(display_name=name, lat=lat, lon=lon)


def place_picker(label: str, key_prefix: str, city: str, allow_geolocation: bool = False) -> Place | None:
    st.markdown(f"**{label}**")
    quick_options = ["Custom search..."] + list(QUICK_PICKS.get(city, {}).keys())
    quick_choice = st.selectbox("Quick pick or custom", quick_options, key=f"{key_prefix}_quick_{city}", label_visibility="collapsed")
    place: Place | None = st.session_state.get(f"{key_prefix}_place")

    if quick_choice != "Custom search...":
        query = QUICK_PICKS[city][quick_choice]
        place = geocode_one(query, city_hint=city)
        st.session_state[f"{key_prefix}_place"] = place
    else:
        query = st.text_input(
            "Type a place name",
            key=f"{key_prefix}_query",
            placeholder="e.g. Bandra Railway Station",
            label_visibility="collapsed",
        )
        btn_cols = st.columns([1, 1]) if allow_geolocation else st.columns([1])
        if btn_cols[0].button("🔍 Search", key=f"{key_prefix}_search_btn") and query.strip():
            st.session_state[f"{key_prefix}_suggestions"] = search_places(query, city_hint=city)
        if allow_geolocation and btn_cols[1].button("📍 Use current location", key=f"{key_prefix}_geoloc_btn"):
            located = resolve_current_location()
            if located:
                place = located
                st.session_state[f"{key_prefix}_place"] = place
                st.session_state[f"{key_prefix}_suggestions"] = []

        suggestions = st.session_state.get(f"{key_prefix}_suggestions", [])
        if suggestions:
            labels = [p.display_name for p in suggestions]
            picked_label = st.selectbox("Matches found", labels, key=f"{key_prefix}_pick")
            place = suggestions[labels.index(picked_label)]
            st.session_state[f"{key_prefix}_place"] = place

    if place:
        st.caption(f"📍 {place.display_name}")
    else:
        st.caption("No place selected yet.")
    return place


# ---------------------------------------------------------------------------
# Sidebar: Satellite Image extraction demo controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Satellite Extraction Controls")
    mode = st.radio("Input", ["Synthetic city tile", "Upload image or mask"], index=0)
    city_name = st.selectbox("City / AOI center (synthetic demo)", list(CITY_PRESETS.keys()), index=0)
    preset_lat, preset_lon = CITY_PRESETS[city_name]
    if city_name == "Custom":
        selected_center_lat = st.number_input("Center latitude", value=float(preset_lat), format="%.6f")
        selected_center_lon = st.number_input("Center longitude", value=float(preset_lon), format="%.6f")
    else:
        selected_center_lat, selected_center_lon = preset_lat, preset_lon

    occlusion_strength = st.slider("Occlusion severity", 0.10, 0.95, 0.58, 0.05)
    seed = st.number_input("Demo seed", min_value=1, max_value=999, value=7, step=1)
    max_gap = st.slider("Healing gap", 20, 120, 62, 2)
    angle_tolerance = st.slider("Alignment tolerance", 10, 80, 46, 2)
    disable_count = st.slider("Auto-disable top nodes", 0, 8, 2, 1)
    synthetic_disaster_scenario = st.selectbox(
        "Synthetic disaster scenario",
        ["Urban flood", "Major accident", "Bridge failure", "Landslide", "Emergency convoy"],
        index=0,
    )

    st.divider()
    st.subheader("Upload")
    upload_image = st.file_uploader("Satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    upload_mask = st.file_uploader("Road mask", type=["png", "jpg", "jpeg", "tif", "tiff"])


st.markdown(
    """
    <div class='rr-hero'>
        <h1>ROUTE RESILIENCE AI</h1>
        <p>ISRO Disaster Decision Support Dashboard &mdash; occlusion-robust road extraction, real-world
        emergency routing, and network resilience analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Run the synthetic-demo segmentation + graph pipeline (feeds the
# Satellite Image, Topology, Pipeline, and Dataset tabs)
# ---------------------------------------------------------------------------
if mode == "Upload image or mask" and (upload_image or upload_mask):
    if upload_image:
        image = read_uploaded_image(upload_image)
    else:
        mask_preview = read_uploaded_image(upload_mask)
        image = np.dstack([mask_preview[:, :, 0]] * 3)

    if upload_mask:
        mask_rgb = read_uploaded_image(upload_mask)
        broken_mask = ensure_uint8_mask(mask_rgb)
    else:
        broken_mask = estimate_road_mask_from_rgb(image)

    ground_truth = broken_mask
    occlusion = np.zeros(broken_mask.shape, dtype=np.uint8)
    center_lat, center_lon, meters_per_pixel = float(selected_center_lat), float(selected_center_lon), 2.8
else:
    scene = make_demo_scene(occlusion_strength=occlusion_strength, seed=int(seed))
    image = scene.image
    ground_truth = scene.ground_truth_mask
    broken_mask = scene.broken_mask
    occlusion = scene.occlusion_mask
    center_lat, center_lon, meters_per_pixel = float(selected_center_lat), float(selected_center_lon), scene.meters_per_pixel


bundle = build_graph_bundle(broken_mask, max_gap=float(max_gap), angle_tolerance=float(angle_tolerance))
scores = centrality_scores(bundle.healed_graph)
top_nodes = [node for node, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:disable_count]]
synthetic_stress = stress_test(bundle.healed_graph, top_nodes)
broken_metrics = segmentation_metrics(broken_mask, ground_truth, occlusion)
healed_metrics = segmentation_metrics(bundle.healed_mask, ground_truth, occlusion)

component_gain = 0.0
if bundle.raw_components:
    component_gain = (bundle.raw_components - bundle.healed_components) / bundle.raw_components


tab_satellite, tab_planner, tab_analytics, tab_pipeline, tab_dataset = st.tabs(
    ["Satellite Image", "Emergency Planner", "Analytics", "Pipeline", "Dataset"]
)

# ---------------------------------------------------------------------------
# TAB 1: Satellite Image (extraction + topology healing on the demo scene)
# ---------------------------------------------------------------------------
with tab_satellite:
    kpi_cards(
        [
            {
                "title": "Road Mask Quality",
                "value": f"{healed_metrics['Dice']:.3f}",
                "help": "Dice score after mask-to-graph healing.",
                "color": "#24746b",
            },
            {
                "title": "Occlusion Recovery",
                "value": f"{healed_metrics.get('Occlusion recall', 0):.3f}",
                "help": "Recovered road pixels under shadow or canopy.",
                "color": "#557c2b",
            },
            {
                "title": "Connected Components",
                "value": f"{bundle.healed_components}",
                "help": f"Reduced from {bundle.raw_components} fragmented components.",
                "color": "#2c7bb6",
            },
            {
                "title": "Healed Road Gaps",
                "value": f"{len(bundle.bridged_edges)}",
                "help": "MST/disjoint-set bridges added to restore topology.",
                "color": "#a23b72",
            },
            {
                "title": "Gatekeeper Nodes",
                "value": f"{len(top_nodes)}",
                "help": "High-betweenness nodes selected for the synthetic demo.",
                "color": "#dd563b",
            },
            {
                "title": "Synthetic Resilience Index",
                "value": f"{synthetic_stress['resilience_index']:.3f}",
                "help": "Baseline network cost divided by disaster network cost.",
                "color": "#8a6f14",
            },
        ]
    )
    st.subheader("Road Extraction from Satellite Imagery")
    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Satellite tile", width="stretch")
    c2.image(overlay_mask(image, broken_mask, (221, 86, 59), 0.62), caption="Broken road mask under occlusion", width="stretch")
    c3.image(overlay_mask(image, bundle.healed_mask, (36, 116, 107), 0.62), caption="Healed routable topology", width="stretch")

    table = pd.DataFrame(
        [
            {"Layer": "Broken mask", **broken_metrics},
            {"Layer": "Healed graph mask", **healed_metrics},
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Topology & Gatekeeper Nodes")
    left, right = st.columns([1.45, 1])
    with left:
        fmap = build_synthetic_map(
            bundle.healed_graph,
            scores,
            top_nodes,
            (center_lat, center_lon),
            broken_mask.shape,
            meters_per_pixel,
        )
        st.iframe(fmap.get_root().render(), height=560)
    with right:
        st.markdown("**Critical Nodes**")
        centrality_table = pd.DataFrame(
            [
                {
                    "Node": node,
                    "Betweenness": score,
                    "Degree": int(bundle.healed_graph.degree[node]),
                    "Disabled": node in top_nodes,
                }
                for node, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:15]
            ]
        )
        st.dataframe(centrality_table, hide_index=True, width="stretch")
        st.markdown(
            f"<div class='rr-status'>Connectivity ratio improvement: {component_gain:.1%}<br>"
            f"Raw components: {bundle.raw_components}<br>"
            f"Healed components: {bundle.healed_components}</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "This tab runs entirely on the synthetic/uploaded demo scene from the sidebar. "
        "It shows how the road-extraction + topology-healing pipeline works, independent of "
        "the real-world routing in the Emergency Planner tab."
    )

# ---------------------------------------------------------------------------
# TAB 2: Emergency Planner (real OpenStreetMap road network)
# ---------------------------------------------------------------------------
with tab_planner:
    st.markdown(
        "<div class='rr-banner'>🚨 <b>Emergency Planner</b> &mdash; real OpenStreetMap roads, live place "
        "search, and disaster-aware routing for first responders. This uses actual street data, separate "
        "from the synthetic demo in the Satellite Image tab.</div>",
        unsafe_allow_html=True,
    )

    top_cols = st.columns([1.3, 1, 1.3])
    ep_city = top_cols[0].selectbox("City", list(CITY_CENTERS.keys()), key="ep_city")
    ep_radius_km = top_cols[1].slider("Coverage radius (km)", 3, 12, 6, 1, key="ep_radius_km")
    ep_tile = top_cols[2].selectbox("Map style", list(TILE_OPTIONS.keys()), index=1, key="ep_tile")

    ep_graph = None
    ep_amenities = None
    ep_error = None
    try:
        ep_graph = load_city_graph(ep_city, radius_m=ep_radius_km * 1000)
        ep_amenities = load_city_amenities(ep_city, radius_m=ep_radius_km * 1000)
    except Exception as exc:  # noqa: BLE001
        ep_error = str(exc)

    if ep_error:
        st.error(
            f"Could not download the OpenStreetMap road network for {ep_city}: {ep_error}\n\n"
            "This tab needs outbound internet access to the OSM Overpass API. It will work once "
            "deployed somewhere with internet access (e.g. Streamlit Community Cloud)."
        )
    else:
        st.caption(
            f"Loaded {ep_graph.number_of_nodes():,} real intersections and {ep_graph.number_of_edges():,} "
            f"real road segments within {ep_radius_km} km of {ep_city} center."
        )

        st.markdown("#### Plan a Route")
        start_col, dest_col = st.columns(2)
        with start_col:
            start_place = place_picker("Start", "ep_start", ep_city, allow_geolocation=True)
        with dest_col:
            dest_place = place_picker("Destination", "ep_dest", ep_city, allow_geolocation=False)

        st.markdown("#### Disaster Scenario")
        disaster_cols = st.columns(len(DISASTER_ICONS))
        if "ep_disaster" not in st.session_state:
            st.session_state["ep_disaster"] = "Flood"
        for i, (dname, demoji) in enumerate(DISASTER_ICONS.items()):
            if disaster_cols[i].button(f"{demoji} {dname}", key=f"ep_disaster_btn_{dname}", width="stretch"):
                st.session_state["ep_disaster"] = dname
        disaster_scenario = st.session_state["ep_disaster"]
        st.caption(f"Selected scenario: **{disaster_scenario}**")

        all_road_names = road_names(ep_graph)
        default_blocked: list[str] = []
        if disaster_scenario == "Bridge Collapse":
            default_blocked = [r for r in bridge_road_names(ep_graph) if r in all_road_names][:3]

        blocked_roads = st.multiselect(
            "Blocked road(s) -- type to filter the list",
            options=all_road_names,
            default=default_blocked,
            key="ep_blocked_roads",
        )

        find_route_clicked = st.button("🚑 FIND ROUTE", type="primary", width="stretch")

        if start_place and dest_place:
            orig_node = nearest_node(ep_graph, start_place.lat, start_place.lon)
            dest_node = nearest_node(ep_graph, dest_place.lat, dest_place.lon)
            damaged_graph, removed_edges = block_roads_by_name(ep_graph, blocked_roads)

            baseline: RouteResult = shortest_route(ep_graph, orig_node, dest_node)
            emergency: RouteResult = shortest_route(damaged_graph, orig_node, dest_node)

            metrics_key = f"{ep_city}|{ep_radius_km}|{start_place.display_name}|{dest_place.display_name}|{tuple(sorted(blocked_roads))}"
            if find_route_clicked:
                with st.spinner("Computing network-wide resilience and critical roads..."):
                    st.session_state["ep_resilience"] = network_resilience_index(ep_graph, damaged_graph)
                    st.session_state["ep_critical_roads"] = critical_roads(ep_graph)
                    st.session_state["ep_metrics_key"] = metrics_key

            has_metrics = st.session_state.get("ep_metrics_key") == metrics_key
            resilience = st.session_state.get("ep_resilience") if has_metrics else None
            crit_roads = st.session_state.get("ep_critical_roads") if has_metrics else []

            baseline_km = baseline.length_m / 1000.0
            emergency_km = emergency.length_m / 1000.0
            delay_min = None
            if baseline.path and emergency.path and np.isfinite(baseline.time_min) and np.isfinite(emergency.time_min):
                delay_min = emergency.time_min - baseline.time_min

            route_cards(
                [
                    {
                        "title": "Normal Route",
                        "value": f"{baseline_km:.2f} km" if baseline.path else "No route",
                        "help": f"{baseline.time_min:.1f} min at typical road speeds." if baseline.path else "Start or destination unreachable.",
                        "color": "#1455d9",
                    },
                    {
                        "title": "Emergency Route",
                        "value": f"{emergency_km:.2f} km" if emergency.path else "No route",
                        "help": f"{emergency.time_min:.1f} min with {len(blocked_roads)} road(s) blocked." if emergency.path else f"{disaster_scenario} disconnects the destination.",
                        "color": "#12a150" if emergency.path else "#dd563b",
                    },
                    {
                        "title": "Delay",
                        "value": f"{delay_min:+.1f} min" if delay_min is not None else "N/A",
                        "help": "Extra travel time caused by the disaster scenario.",
                        "color": "#8a6f14",
                    },
                ]
            )

            map_col, side_col = st.columns([1.4, 1])
            with map_col:
                fmap = build_emergency_map(
                    ep_graph,
                    center=(start_place.lat, start_place.lon),
                    tile_choice=ep_tile,
                    start_point=(start_place.lat, start_place.lon),
                    end_point=(dest_place.lat, dest_place.lon),
                    baseline_path=baseline.path,
                    emergency_path=emergency.path,
                    blocked_edges=removed_edges,
                    amenities_gdf=ep_amenities,
                )
                st.iframe(fmap.get_root().render(), height=560)
                st.caption("Blue = normal route · Green = emergency route · Red dashed = blocked road · Icons = hospital/police/airport/shelter")

            with side_col:
                st.markdown("**AI Summary**")
                alt_hint = crit_roads[0][0] if crit_roads else None
                ai_summary = generate_ai_summary(disaster_scenario, blocked_roads, bool(emergency.path), alt_hint, delay_min)
                st.markdown(f"<div class='rr-status'>{ai_summary}</div>", unsafe_allow_html=True)

                st.markdown("**Critical Roads**")
                if crit_roads:
                    st.dataframe(
                        pd.DataFrame([{"Road": name, "Impact score": f"{score:.3f}"} for name, score in crit_roads]),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.caption("Press FIND ROUTE to compute network resilience and critical roads.")

                if has_metrics and resilience is not None:
                    st.metric("Network Resilience Index", f"{resilience:.3f}", help="1.0 = no degradation. Sampled across the loaded network.")

                    pdf_bytes = build_pdf_report(
                        city=ep_city,
                        scenario=disaster_scenario,
                        start_name=start_place.display_name,
                        dest_name=dest_place.display_name,
                        blocked_roads=blocked_roads,
                        baseline_km=baseline_km,
                        baseline_min=baseline.time_min,
                        emergency_km=emergency_km,
                        emergency_min=emergency.time_min,
                        delay_min=delay_min,
                        resilience_index=resilience,
                        critical_roads=crit_roads,
                        ai_summary=ai_summary,
                    )
                    st.download_button(
                        "📄 Download Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"route_resilience_{ep_city.lower()}_{disaster_scenario.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )
        else:
            st.info("Pick a Start and a Destination above (quick pick or search), then press FIND ROUTE.")

# ---------------------------------------------------------------------------
# TAB 3: Analytics
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Analytics")

    st.markdown("**Emergency Planner (real-world) resilience**")
    has_ep_metrics = st.session_state.get("ep_resilience") is not None
    if has_ep_metrics:
        crit_roads = st.session_state.get("ep_critical_roads", [])
        analytics_cols = st.columns(3)
        analytics_cols[0].metric("Network Resilience Index", f"{st.session_state['ep_resilience']:.3f}")
        analytics_cols[1].metric("Blocked Roads", str(len(st.session_state.get("ep_blocked_roads", []))))
        analytics_cols[2].metric("Critical Roads Tracked", str(len(crit_roads)))
        if crit_roads:
            chart_df = pd.DataFrame(crit_roads, columns=["Road", "Impact score"]).set_index("Road")
            st.bar_chart(chart_df)
    else:
        st.info("Run a scenario in the Emergency Planner tab (press FIND ROUTE) to populate real-world analytics here.")

    st.divider()
    st.markdown("**Satellite extraction (synthetic demo) resilience**")
    synth_cols = st.columns(4)
    synth_cols[0].metric("Resilience Index", f"{synthetic_stress['resilience_index']:.3f}")
    synth_cols[1].metric("Baseline Network Cost", f"{synthetic_stress['baseline_path']:.2f}")
    synth_cols[2].metric("Damaged Network Cost", f"{synthetic_stress['damaged_path']:.2f}")
    synth_cols[3].metric("Component Increase", str(int(synthetic_stress["component_delta"])))
    st.dataframe(
        pd.DataFrame(
            [
                {"Layer": "Broken mask", **broken_metrics},
                {"Layer": "Healed graph mask", **healed_metrics},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------------------------
# TAB 4: Pipeline
# ---------------------------------------------------------------------------
with tab_pipeline:
    st.subheader("Model Pipeline")
    st.markdown(
        """
        <div class='pipeline-flow'>
            <div class='pipeline-step'><b>Satellite Image</b><small>Cartosat, Resourcesat, Sentinel, drone, or uploaded RGB tile.</small></div>
            <div class='pipeline-step'><b>Preprocessing</b><small>Tiling, contrast normalization, denoising, and occlusion simulation.</small></div>
            <div class='pipeline-step'><b>U-Net Road Extraction</b><small>Current GitHub-style UNet baseline, with DeepLab/SegFormer upgrade path.</small></div>
            <div class='pipeline-step'><b>Road Mask</b><small>Binary road probability converted into routable evidence.</small></div>
            <div class='pipeline-step'><b>Morphological Processing</b><small>Closing, thinning cleanup, and small-fragment suppression.</small></div>
            <div class='pipeline-step'><b>Skeletonization</b><small>CRESI-inspired centerline extraction from road mask.</small></div>
            <div class='pipeline-step'><b>Road Graph Generation</b><small>CRESI-inspired endpoints, intersections, edge weights, and healed gaps.</small></div>
            <div class='pipeline-step'><b>Network Analysis</b><small>Betweenness centrality and graph efficiency for gatekeeper discovery.</small></div>
            <div class='pipeline-step'><b>Critical Roads</b><small>Road segments and nodes with high systemic impact.</small></div>
            <div class='pipeline-step'><b>Disaster Simulation</b><small>Flood, accident, bridge failure, landslide, or convoy closure scenarios.</small></div>
            <div class='pipeline-step'><b>Alternate Route Planning</b><small>Dijkstra shortest path before and after blocked roads.</small></div>
            <div class='pipeline-step'><b>Interactive GIS Map</b><small>Folium/Leaflet map embedded in the Streamlit dashboard.</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ```text
        Satellite Image
                |
                v
        Preprocessing
                |
                v
        U-Net Road Extraction
                |
                v
        Road Mask
                |
                v
        Morphological Processing
                |
                v
        Skeletonization
                |
                v
        Road Graph Generation
                |
                v
        Network Analysis
           |                 |
           v                 v
        Critical Roads   Disaster Simulation
           |                 |
           +--------+--------+
                    v
        Alternate Route Planning
        (Dijkstra / A*)
                    |
                    v
        Interactive GIS Map
        (Folium / Leaflet)
                    |
                    v
        Streamlit Dashboard
        ```
        """
    )
    st.caption(
        "The Emergency Planner tab swaps the synthetic mask/graph in this flow for a real "
        "OpenStreetMap road network (via osmnx), so 'Road Graph Generation' becomes 'Download "
        "real OSM graph' and routing/critical-roads run on real streets."
    )

# ---------------------------------------------------------------------------
# TAB 5: Dataset
# ---------------------------------------------------------------------------
with tab_dataset:
    st.subheader("Hackathon Data Path")
    st.markdown(
        """
        Primary training can start with DeepGlobe road extraction tiles, then fine-tune with SpaceNet Roads or
        OSM-derived masks around Indian cities. The dashboard accepts uploaded masks immediately, so a trained
        model can replace the demo estimator without changing the graph analytics.

        Recommended flow:

        1. Download DeepGlobe from Kaggle: `balraj98/deepglobe-road-extraction-dataset`.
        2. Build `data/train/images`, `data/train/masks`, `data/val/images`, and `data/val/masks`.
        3. Train UNet/DeepLab/SegFormer with Dice + BCE + boundary/connectivity loss.
        4. Export prediction masks and upload them here for graph healing and stress testing.
        """
    )
    st.code("python scripts/prepare_deepglobe.py --kaggle-dataset balraj98/deepglobe-road-extraction-dataset", language="bash")
    st.code("python scripts/train_segmentation.py --data data --model unet --epochs 30", language="bash")

    out_col1, out_col2 = st.columns(2)
    out_col1.download_button(
        "Download healed mask PNG",
        data=to_png_bytes(bundle.healed_mask),
        file_name="healed_road_mask.png",
        mime="image/png",
    )
    graph_json = nx.node_link_data(bundle.healed_graph, edges="edges")
    out_col2.download_button(
        "Download graph JSON",
        data=json.dumps(graph_json, indent=2),
        file_name="healed_graph.json",
        mime="application/json",
    )
