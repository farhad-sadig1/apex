from __future__ import annotations

import math
from typing import Any

from incidents.lateral_safety import LATERAL_PROXIMITY_THRESHOLD


def calculate_ride_metrics(
    incidents: list[dict[str, Any]],
    total_frames: float,
    fps: float,
) -> dict[str, Any]:
    """Start at 100, deduct per incident, clamp at 0."""
    ride_duration_seconds = total_frames / fps

    for incident in incidents:
        incident_type = incident.get("incident_type")
        deduction = 0.0

        if incident_type == "dynamic_side_closure":
            intensity = math.tanh(incident.get("max_severity_score", 0.0) * 10.0)
            deduction = 15.0 * intensity

        elif incident_type == "sustained_side_proximity":
            gap = incident.get("min_normalized_gap", LATERAL_PROXIMITY_THRESHOLD)
            proximity_multiplier = (
                (LATERAL_PROXIMITY_THRESHOLD - gap) / LATERAL_PROXIMITY_THRESHOLD
                if gap < LATERAL_PROXIMITY_THRESHOLD
                else 0.0
            )
            deduction = 10.0 * (1.0 + proximity_multiplier)

        elif incident_type in ("pedestrian_ahead", "pedestrian_critical"):
            duration_seconds = (
                incident["end_frame"] - incident["start_frame"] + 1
            ) / fps
            max_proximity_index = incident.get("max_proximity_index", 0.5)
            duration_factor = math.tanh(duration_seconds / 3.0)
            base = 8.0 if incident_type == "pedestrian_ahead" else 15.0
            deduction = (
                base
                * (0.5 + 0.5 * duration_factor)
                * (0.5 + 0.5 * max_proximity_index)
            )

        safety_score -= deduction

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
