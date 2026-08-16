from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

VEHICLE_CLASSES = frozenset({"car", "truck", "bus"})

# keep in sync with detect.py overlay
LATERAL_PROXIMITY_THRESHOLD = 0.25
CRITICAL_PROXIMITY_THRESHOLD = 0.15
MAX_GAP_PIXEL_FRACTION = 0.15

# skip tiny boxes; gap/height blows up at distance
MIN_RELEVANT_SIZE_FRACTION = 0.12


@dataclass
class _TrackState:
    sustained_violation_frames: int = 0
    dynamic_event: dict[str, Any] | None = None
    sustained_event: dict[str, Any] | None = None


def _parse_object(obj: Any) -> tuple[int, str, tuple[int, int, int, int]]:
    if isinstance(obj, dict):
        track_id = int(obj["track_id"])
        cls_name = str(obj["class"])
        box = obj["box"]
    else:
        track_id, cls_name, box = obj
        track_id = int(track_id)
        cls_name = str(cls_name)

    x1, y1, x2, y2 = (int(v) for v in box)
    return track_id, cls_name, (x1, y1, x2, y2)


def _is_in_side_zone(
    box: tuple[int, int, int, int],
    frame_width: int,
    side_fraction: float,
) -> bool:
    x1, _, x2, _ = box
    left_boundary = frame_width * side_fraction
    right_boundary = frame_width * (1.0 - side_fraction)
    center_x = (x1 + x2) / 2.0
    return center_x <= left_boundary or center_x >= right_boundary


def _inner_edge_and_gap(
    box: tuple[int, int, int, int],
    frame_width: int,
) -> tuple[float, str]:
    x1, _, x2, _ = box
    frame_center = frame_width / 2.0
    center_x = (x1 + x2) / 2.0

    if center_x <= frame_center:
        inner_edge = x2
        pixel_gap = frame_center - inner_edge
        side = "left"
    else:
        inner_edge = x1
        pixel_gap = inner_edge - frame_center
        side = "right"

    return pixel_gap, side


def _normalized_gap(pixel_gap: float, box: tuple[int, int, int, int]) -> float:
    _, y1, _, y2 = box
    height = max(y2 - y1, 1)
    return pixel_gap / height


def _box_height_fraction(box: tuple[int, int, int, int], frame_height: int) -> float:
    """Bbox height as a fraction of frame height."""
    if frame_height <= 0:
        return 0.0
    _, y1, _, y2 = box
    return max(y2 - y1, 0) / float(frame_height)


def _dual_gate(
    normalized_gap: float,
    gap_fraction: float,
    proximity_threshold: float,
    max_gap_pixel_fraction: float,
) -> bool:
    """True when both the box-relative and frame-relative gap gates pass."""
    return (
        normalized_gap < proximity_threshold
        and gap_fraction < max_gap_pixel_fraction
    )


def _log_event(action: str, incident_type: str, track_id: int, frame: int, **details: Any) -> None:
    detail_str = " ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" {detail_str}" if detail_str else ""
    print(f"[LateralSafety] {action} {incident_type} track_id={track_id} frame={frame}{suffix}")


def _start_event(
    incident_type: str,
    track_id: int,
    frame_idx: int,
    normalized_gap: float,
    side: str,
) -> dict[str, Any]:
    _log_event(
        "START",
        incident_type,
        track_id,
        frame_idx,
        normalized_gap=f"{normalized_gap:.3f}",
        side=side,
    )
    return {
        "track_id": track_id,
        "incident_type": incident_type,
        "start_frame": frame_idx,
        "end_frame": frame_idx,
        "max_severity_score": 0.0,
        "min_normalized_gap": normalized_gap,
        "side": side,
    }


def _update_event(
    event: dict[str, Any],
    frame_idx: int,
    normalized_gap: float,
    severity_score: float,
) -> None:
    event["end_frame"] = frame_idx
    event["min_normalized_gap"] = min(event["min_normalized_gap"], normalized_gap)
    event["max_severity_score"] = max(event["max_severity_score"], severity_score)


def _finalize_event(event: dict[str, Any]) -> dict[str, Any]:
    _log_event(
        "END",
        event["incident_type"],
        event["track_id"],
        event["end_frame"],
        start_frame=event["start_frame"],
        max_severity=f"{event['max_severity_score']:.3f}",
        min_normalized_gap=f"{event['min_normalized_gap']:.3f}",
        side=event["side"],
    )
    return {
        "track_id": event["track_id"],
        "incident_type": event["incident_type"],
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "max_severity_score": event["max_severity_score"],
        "min_normalized_gap": event["min_normalized_gap"],
        "side": event["side"],
    }


def _close_active_event(
    event: dict[str, Any] | None,
    incidents: list[dict[str, Any]],
) -> None:
    if event is not None:
        incidents.append(_finalize_event(event))


def analyze_lateral_safety(
    tracked_objects: Sequence[Sequence[Any]],
    frame_width: int,
    frame_height: int,
    *,
    side_fraction: float = 0.35,
    proximity_threshold: float = LATERAL_PROXIMITY_THRESHOLD,
    critical_proximity_threshold: float = CRITICAL_PROXIMITY_THRESHOLD,
    max_gap_pixel_fraction: float = MAX_GAP_PIXEL_FRACTION,
    sustained_frame_threshold: int = 30,
    min_relevant_size_fraction: float = MIN_RELEVANT_SIZE_FRACTION,
) -> list[dict[str, Any]]:
    """Detect close-pass and sustained side-proximity incidents from tracked vehicles."""
    track_states: dict[int, _TrackState] = defaultdict(_TrackState)
    incidents: list[dict[str, Any]] = []

    for frame_idx, frame_objects in enumerate(tracked_objects):
        active_track_ids: set[int] = set()

        for obj in frame_objects:
            track_id, cls_name, box = _parse_object(obj)
            if cls_name not in VEHICLE_CLASSES:
                continue
            if not _is_in_side_zone(box, frame_width, side_fraction):
                continue
            if _box_height_fraction(box, frame_height) <= min_relevant_size_fraction:
                continue

            active_track_ids.add(track_id)
            state = track_states[track_id]

            pixel_gap, side = _inner_edge_and_gap(box, frame_width)
            gap_fraction = pixel_gap / float(frame_width)
            normalized_gap = _normalized_gap(pixel_gap, box)

            critical_active = _dual_gate(
                normalized_gap,
                gap_fraction,
                critical_proximity_threshold,
                max_gap_pixel_fraction,
            )
            caution_active = _dual_gate(
                normalized_gap,
                gap_fraction,
                proximity_threshold,
                max_gap_pixel_fraction,
            )

            if critical_active:
                dynamic_severity = critical_proximity_threshold - normalized_gap
                if state.dynamic_event is None:
                    state.dynamic_event = _start_event(
                        "dynamic_side_closure",
                        track_id,
                        frame_idx,
                        normalized_gap,
                        side,
                    )
                _update_event(
                    state.dynamic_event,
                    frame_idx,
                    normalized_gap,
                    dynamic_severity,
                )
            else:
                _close_active_event(state.dynamic_event, incidents)
                state.dynamic_event = None

            if caution_active:
                state.sustained_violation_frames += 1
            else:
                state.sustained_violation_frames = 0
                _close_active_event(state.sustained_event, incidents)
                state.sustained_event = None

            sustained_active = (
                state.sustained_violation_frames > sustained_frame_threshold
            )
            if sustained_active:
                sustained_severity = (
                    proximity_threshold - normalized_gap
                ) * state.sustained_violation_frames
                if state.sustained_event is None:
                    state.sustained_event = _start_event(
                        "sustained_side_proximity",
                        track_id,
                        frame_idx,
                        normalized_gap,
                        side,
                    )
                _update_event(
                    state.sustained_event,
                    frame_idx,
                    normalized_gap,
                    sustained_severity,
                )

        for track_id, state in track_states.items():
            if track_id in active_track_ids:
                continue

            state.sustained_violation_frames = 0
            _close_active_event(state.dynamic_event, incidents)
            state.dynamic_event = None
            _close_active_event(state.sustained_event, incidents)
            state.sustained_event = None

    for state in track_states.values():
        _close_active_event(state.dynamic_event, incidents)
        _close_active_event(state.sustained_event, incidents)

    return incidents
