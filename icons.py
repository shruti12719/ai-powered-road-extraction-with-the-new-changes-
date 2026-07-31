from __future__ import annotations

# key -> (emoji for text/legend use, folium.Icon color, Font Awesome icon name)
AMENITY_ICONS: dict[str, tuple[str, str, str]] = {
    "hospital": ("🏥", "red", "plus-square"),
    "police": ("🚓", "darkblue", "shield-alt"),
    "fire_station": ("🚒", "orange", "fire"),
    "aerodrome": ("✈️", "purple", "plane"),
    "shelter": ("🏕️", "green", "home"),
    "assembly_point": ("🏕️", "green", "home"),
}

DISASTER_ICONS: dict[str, str] = {
    "Flood": "🌊",
    "Landslide": "⛰️",
    "Bridge Collapse": "🌉",
    "Major Accident": "🚧",
    "Fire": "🔥",
}


def icon_for(amenity_key: str | None) -> tuple[str, str, str]:
    if not amenity_key:
        return ("📍", "gray", "map-marker")
    return AMENITY_ICONS.get(str(amenity_key).lower(), ("📍", "gray", "map-marker"))


def label_for(row) -> str:
    amenity_key = row.get("amenity") or row.get("aeroway") or row.get("emergency")
    emoji, _, _ = icon_for(amenity_key)
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return f"{emoji} {name}"
    return f"{emoji} {str(amenity_key).replace('_', ' ').title() if amenity_key else 'Point of interest'}"
