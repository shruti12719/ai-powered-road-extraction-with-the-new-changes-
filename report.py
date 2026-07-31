from __future__ import annotations

from fpdf import FPDF


def generate_ai_summary(
    scenario: str,
    blocked_roads: list[str],
    route_exists: bool,
    alt_road_hint: str | None,
    delay_min: float | None,
) -> str:
    """Rule-based, template-driven summary (no external LLM call). Reads like
    a dispatcher briefing, matching the 'AI Summary' block in the dashboard."""
    if not blocked_roads:
        return "No roads are currently blocked. The emergency vehicle can use the normal route directly."

    roads_txt = " and ".join(blocked_roads[:3])
    if len(blocked_roads) > 3:
        roads_txt += f", and {len(blocked_roads) - 3} more road(s)"

    if not route_exists:
        return (
            f"{scenario} has closed {roads_txt}. The origin and destination are no longer connected "
            f"through the road network. Recommend reopening one of the blocked roads or choosing a "
            f"different staging point."
        )

    delay_txt = f"Estimated delay: {delay_min:.1f} minutes." if delay_min is not None else ""
    alt_txt = f"Emergency vehicles should use {alt_road_hint}." if alt_road_hint else "An alternate route is available (shown in green)."
    return f"{scenario} has disconnected the direct path via {roads_txt}. {alt_txt} {delay_txt}".strip()


def build_pdf_report(
    city: str,
    scenario: str,
    start_name: str,
    dest_name: str,
    blocked_roads: list[str],
    baseline_km: float,
    baseline_min: float,
    emergency_km: float,
    emergency_min: float,
    delay_min: float | None,
    resilience_index: float,
    critical_roads: list[tuple[str, float]],
    ai_summary: str,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Route Resilience AI - Emergency Route Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"City: {city}    Scenario: {scenario}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Route Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    delay_txt = f"{delay_min:+.1f} min" if delay_min is not None else "N/A (no route)"
    pdf.multi_cell(
        0,
        7,
        f"Start: {start_name}\n"
        f"Destination: {dest_name}\n"
        f"Blocked roads: {', '.join(blocked_roads) if blocked_roads else 'None'}\n\n"
        f"Normal route: {baseline_km:.2f} km / {baseline_min:.1f} min\n"
        f"Emergency route: {emergency_km:.2f} km / {emergency_min:.1f} min\n"
        f"Delay: {delay_txt}\n"
        f"Network resilience index: {resilience_index:.3f} (1.0 = no degradation)",
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "AI Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, ai_summary)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Critical Roads (highest systemic impact)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    if critical_roads:
        for i, (road, score) in enumerate(critical_roads, start=1):
            pdf.cell(0, 7, f"{i}. {road}  (impact score {score:.3f})", ln=True)
    else:
        pdf.cell(0, 7, "Not enough graph data to rank critical roads.", ln=True)

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="replace")
    return bytes(raw)
