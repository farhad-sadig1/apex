from __future__ import annotations

from typing import Any, Literal

LiveStatus = Literal["clear", "awareness", "risk", "critical"]
IncidentTier = Literal["awareness", "risk", "critical"]

TIER_RANK: dict[str, int] = {
    "clear": 0,
    "awareness": 1,
    "risk": 2,
    "critical": 3,
}

# Fallback when an incident has no severity_tier (legacy type-only payloads).
INCIDENT_TYPE_TIER: dict[str, IncidentTier] = {
    "pedestrian_ahead": "awareness",
    "sustained_side_proximity": "risk",
    "pedestrian_critical": "critical",
    "dynamic_side_closure": "critical",
}


def incident_tier(incident: dict[str, Any]) -> IncidentTier:
    """Peak tier for one incident, taken from the detectors when present."""
    tier = incident.get("severity_tier")
    if tier in TIER_RANK and tier != "clear":
        return tier  # type: ignore[return-value]
    mapped = INCIDENT_TYPE_TIER.get(str(incident.get("incident_type", "")))
    return mapped if mapped is not None else "awareness"


def _frame_span(incident: dict[str, Any], total_frames: int) -> range:
    start = int(incident.get("start_frame", 0))
    end = int(incident.get("end_frame", start))
    lo = max(0, min(start, end))
    hi = min(total_frames - 1, max(start, end))
    if total_frames <= 0 or lo > hi:
        return range(0)
    return range(lo, hi + 1)


def live_status_by_frame(
    incidents: list[dict[str, Any]],
    total_frames: int,
) -> list[LiveStatus]:
    """Highest-severity active incident at each frame. No carry-over."""
    if total_frames <= 0:
        return []

    status: list[LiveStatus] = ["clear"] * total_frames
    for incident in incidents:
        tier = incident_tier(incident)
        rank = TIER_RANK[tier]
        for frame in _frame_span(incident, total_frames):
            if rank > TIER_RANK[status[frame]]:
                status[frame] = tier
    return status


def _event_timestamp_seconds(incident: dict[str, Any], fps: float) -> float:
    start_frame = int(incident.get("start_frame", 0))
    if fps <= 0:
        return 0.0
    return start_frame / fps


def summarize_incidents(
    incidents: list[dict[str, Any]],
    ride_duration_seconds: float,
    fps: float,
) -> dict[str, Any]:
    """Ride-level counts, jump-to list, and risk+critical rate."""
    tier_counts: dict[str, int] = {"awareness": 0, "risk": 0, "critical": 0}
    notable_events: list[dict[str, Any]] = []

    for incident in incidents:
        tier = incident_tier(incident)
        tier_counts[tier] += 1
        if tier not in ("risk", "critical"):
            continue
        start_frame = int(incident.get("start_frame", 0))
        notable_events.append(
            {
                "tier": tier,
                "incident_type": incident.get("incident_type"),
                "start_frame": start_frame,
                "timestamp_seconds": _event_timestamp_seconds(incident, fps),
            }
        )

    notable_events.sort(
        key=lambda event: (-TIER_RANK[event["tier"]], event["timestamp_seconds"])
    )

    ride_minutes = ride_duration_seconds / 60.0 if ride_duration_seconds > 0 else 0.0
    risk_critical_count = tier_counts["risk"] + tier_counts["critical"]
    risk_critical_per_minute = (
        risk_critical_count / ride_minutes if ride_minutes > 0 else 0.0
    )

    return {
        "tier_counts": tier_counts,
        "risk_critical_events": notable_events,
        "risk_critical_per_minute": risk_critical_per_minute,
        "total_incidents_detected": len(incidents),
        "ride_duration_seconds": ride_duration_seconds,
    }


def calculate_ride_metrics(
    incidents: list[dict[str, Any]],
    total_frames: float,
    fps: float,
) -> dict[str, Any]:
    """Per-frame live status plus end-of-ride aggregates. JSON-ready."""
    safe_fps = float(fps) if fps and fps > 0 else 0.0
    frame_count = max(0, int(total_frames))
    ride_duration_seconds = frame_count / safe_fps if safe_fps > 0 else 0.0

    return {
        "frame_status": live_status_by_frame(incidents, frame_count),
        "summary": summarize_incidents(incidents, ride_duration_seconds, safe_fps),
        "incidents": incidents,
    }
