from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from geopy.geocoders import Nominatim

_USER_AGENT = "route-resilience-ai/2.0 (isro-disaster-dashboard-demo)"


@dataclass(frozen=True)
class Place:
    """A single geocoded result. No raw lat/lon is ever shown to the end user."""

    display_name: str
    lat: float
    lon: float


def _geolocator() -> Nominatim:
    return Nominatim(user_agent=_USER_AGENT, timeout=8)


@st.cache_data(show_spinner=False, ttl=3600)
def search_places(query: str, city_hint: str | None = None, limit: int = 5) -> list[Place]:
    """Google-Maps-style search: free text in, ranked candidate places out.

    Streamlit has no native JS autocomplete dropdown, so the UI pattern is:
    type a query -> press Search -> pick the right match from a short list.
    This function powers that "search then confirm" step.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return []

    geolocator = _geolocator()
    attempts = []
    if city_hint:
        attempts.append(f"{query}, {city_hint}, India")
    attempts.append(f"{query}, India")
    attempts.append(query)

    for attempt in attempts:
        try:
            results = geolocator.geocode(attempt, exactly_one=False, limit=limit, addressdetails=False)
        except Exception:
            results = None
        if results:
            return [Place(display_name=r.address, lat=float(r.latitude), lon=float(r.longitude)) for r in results]
    return []


def geocode_one(query: str, city_hint: str | None = None) -> Place | None:
    matches = search_places(query, city_hint=city_hint, limit=1)
    return matches[0] if matches else None


@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat: float, lon: float) -> str:
    """Turn a raw lat/lon (e.g. from browser geolocation) into a readable place name."""
    try:
        location = _geolocator().reverse((lat, lon), exactly_one=True)
        if location:
            return location.address
    except Exception:
        pass
    return f"Current location ({lat:.4f}, {lon:.4f})"
