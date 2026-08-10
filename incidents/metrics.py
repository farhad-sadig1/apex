from __future__ import annotations


def calculate_ride_metrics(
    incidents: list[dict],
    total_frames: float,
    fps: float,
) -> dict:
    ride_duration_seconds = total_frames / fps

    safety_score = 100.0
    for incident in incidents:
        incident_type = incident.get("incident_type")
        if incident_type == "dynamic_side_closure":
            safety_score -= 5.0
        elif incident_type == "sustained_side_proximity":
            safety_score -= 10.0

    final_safety_score = max(0.0, safety_score)

    return {
        "summary": {
            "initial_score": 100.0,
            "final_safety_score": final_safety_score,
            "total_incidents_detected": len(incidents),
            "ride_duration_seconds": ride_duration_seconds,
        },
        "incidents": incidents,
    }
