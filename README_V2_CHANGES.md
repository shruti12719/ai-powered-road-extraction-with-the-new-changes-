# Route Resilience AI — v2.0 changes

## What changed and why

Your original app builds roads from a **synthetic demo image** (or an uploaded
mask) — there's no real map underneath it, so it has no concept of "SV Road"
or "Kokilaben Hospital." To get real place search, real road names, and real
routing, the new **Emergency Planner** tab runs on an actual **OpenStreetMap**
road network (via `osmnx`) for the selected city, completely independent from
the satellite-segmentation demo. Nothing about your original extraction
pipeline was removed — it now lives in the **Satellite Image** tab, unchanged
under the hood.

## New tab layout
`Satellite Image` · `Emergency Planner` · `Analytics` · `Pipeline` · `Dataset`

- **Satellite Image** — your existing road-extraction + topology-healing demo (was "Extraction" + "Topology").
- **Emergency Planner** — new, real-world: city → coverage radius → live place search → disaster buttons → blocked-road picker → FIND ROUTE → map + AI summary + critical roads + PDF report.
- **Analytics** — resilience index, delay, and critical-road charts for whichever scenario you last ran in the Planner, plus the synthetic-demo metrics.
- **Pipeline / Dataset** — unchanged.

## Files
```
app.py                          — rewritten, ties everything together
pipeline.py                     — unchanged (synthetic segmentation/graph demo)
route_resilience/
  geocoder.py                   — Nominatim place search + reverse geocoding
  osm_graph.py                  — real OSM road network + amenity download (cached)
  routing.py                    — nearest-node, block-by-road-name, shortest path,
                                   sampled resilience index, sampled critical roads
  icons.py                      — hospital/police/airport/shelter icon mapping
  map_utils.py                  — folium map builder + 3 tile styles
  report.py                     — PDF report + rule-based "AI summary" text
requirements.txt                — added osmnx, geopy, fpdf2, streamlit-js-eval, geopandas, shapely
```

## Implementation notes / honest caveats

1. **Radius, not full city.** A full administrative-boundary graph for a
   metro like Mumbai is hundreds of thousands of edges and routinely times
   out against the public Overpass API. The Planner instead downloads a
   `graph_from_point` within a slider-controlled radius (3–12 km) of the
   city center — still 100% real OSM data, just scoped for a responsive demo.

2. **No native JS autocomplete.** Streamlit can't render a live Google-style
   dropdown-as-you-type. The practical equivalent used here: type a place →
   press Search → pick the right match from a short results list (or use a
   one-click "Quick pick" for common landmarks per city). Blocked-road
   selection *does* get real type-to-filter behavior, since Streamlit's
   `multiselect` supports that natively once given the real street-name list.

3. **"Use current location"** uses `streamlit-js-eval` to read the browser's
   geolocation API, then reverse-geocodes it to a place name. It needs the
   browser's location permission and only works once deployed (not in a
   sandboxed preview).

4. **Resilience index & critical roads are sampled, not exhaustive.**
   All-pairs shortest paths don't scale to a real street network, so the
   resilience index samples ~25 random nodes and critical-road ranking uses
   sampled edge-betweenness centrality (`k=60`). This is a standard, honest
   approximation — fast enough to run on every "Find Route" click.

5. **AI Summary is rule-based**, not an LLM call — it reads like a dispatcher
   briefing but is deterministic. If you want a real Claude-generated summary
   instead, that's a small follow-up change (an API call from `report.py`)
   and I'm happy to wire it up.

6. **Bridge Collapse auto-suggests real bridges** (OSM `bridge=yes/viaduct`
   tags) as default blocked roads. Flood/Landslide/Fire/Major Accident don't
   have a hazard-layer data source wired in, so those default to no
   pre-selected roads — pick them yourself from the filterable list.

7. **This needs real internet access to work** (OpenStreetMap Overpass API +
   Nominatim). It will run fine on Streamlit Community Cloud, which has
   outbound internet by default. I verified all API calls against the
   installed `osmnx` version and ran a full syntax/import check on every
   file, but I could not execute a live Overpass download from this sandbox
   (Overpass isn't on the sandbox's allowed domain list) — so test the
   Planner tab first thing after you deploy.

## Deploying
1. Replace `app.py`, `pipeline.py` (same as before), and add the
   `route_resilience/` folder to your repo.
2. Replace `requirements.txt` with the one here.
3. Push to GitHub → Streamlit Cloud will redeploy automatically.
