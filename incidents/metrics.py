from __future__ import annotations

import math


def calculate_ride_metrics(
    incidents: list[dict],
    total_frames: float,
    fps: float,
) -> dict:
    total_ride_duration_seconds = total_frames / fps
    total_penalty_weight = 0.0

    for incident in incidents:
        duration_frames = incident["end_frame"] - incident["start_frame"] + 1

        if incident["incident_type"] == "dynamic_side_closure":
            intensity = math.tanh(incident.get("max_severity_score", 0.0) * 10.0)
            incident_weight = duration_frames * intensity * 1.5
        else:
            gap = incident.get("min_normalized_gap", 0.4)
            proximity_factor = (0.4 - gap) / 0.4 if gap < 0.4 else 0.0
            incident_weight = duration_frames * (1.0 + proximity_factor)

        total_penalty_weight += incident_weight

    exposure_factor = total_penalty_weight / max(total_frames, 1.0)
    final_safety_score = max(0.0, 100.0 - (exposure_factor * 20.0))

    return {
        "summary": {
            "initial_score": 100.0,
            "final_safety_score": final_safety_score,
            "total_incidents_detected": len(incidents),
            "ride_duration_seconds": total_ride_duration_seconds,
        },
        "incidents": incidents,
    }
