from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Sequence

PEDESTRIAN_CLASSES = frozenset({"person"})

# in-lane proximity that counts as critical
CRITICAL_PROXIMITY_THRESHOLD = 0.65


@dataclass
class _PedestrianState:
    """Per-track state for forward pedestrian threats."""

    ground_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    consecutive_threat_frames: int = 0
    consecutive_out_of_lane_frames: int = 0
    active_event: dict[str, Any] | None = None


def _parse_object(obj: Any) -> tuple[int, str, tuple[int, int, int, int]]:
    """Unpack a detection into track_id, class, box."""
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


def _ground_contact_point(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Bottom-center of the box (approx. feet)."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _forward_lane_bounds(
    frame_width: int,
    lane_width_fraction: float,
) -> tuple[float, float]:
    """Centered corridor edges used by in-lane tests and lateral_position."""
    half_lane = (frame_width * lane_width_fraction) / 2.0
    lane_center = frame_width / 2.0
    return lane_center - half_lane, lane_center + half_lane


def _is_in_forward_lane(
    ground_x: float,
    ground_y: float,
    frame_width: int,
    frame_height: int,
    lane_width_fraction: float,
    horizon_fraction: float,
) -> bool:
    """True if the ground point is in the centered lane below the horizon."""
    horizon_y = frame_height * horizon_fraction
    if ground_y < horizon_y:
        return False

    lane_left_x, lane_right_x = _forward_lane_bounds(
        frame_width, lane_width_fraction
    )
    return lane_left_x <= ground_x <= lane_right_x


def _lateral_position(
    ground_x: float,
    frame_width: int,
    lane_width_fraction: float,
) -> float:
    """0.0 left corridor edge, 0.5 center, 1.0 right corridor edge."""
    lane_left_x, lane_right_x = _forward_lane_bounds(
        frame_width, lane_width_fraction
    )
    span = lane_right_x - lane_left_x
    if span <= 0:
        return 0.5
    return _clamp01((ground_x - lane_left_x) / span)


def _proximity_index(ground_y: float, frame_height: int) -> float:
    """How close: 0 at the top of the frame, 1 at the bottom."""
    if frame_height <= 0:
        return 0.0
    return _clamp01(ground_y / float(frame_height))


def _log_event(action: str, incident_type: str, track_id: int, frame: int, **details: Any) -> None:
    detail_str = " ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" {detail_str}" if detail_str else ""
    print(f"[PedestrianSafety] {action} {incident_type} track_id={track_id} frame={frame}{suffix}")


def _start_event(
    incident_type: str,
    track_id: int,
    frame_idx: int,
    proximity_index: float,
    lateral_position: float,
) -> dict[str, Any]:
    _log_event(
        "START",
        incident_type,
        track_id,
        frame_idx,
        proximity_index=f"{proximity_index:.3f}",
        lateral_position=f"{lateral_position:.3f}",
    )
    return {
        "track_id": track_id,
        "incident_type": incident_type,
        "start_frame": frame_idx,
        "end_frame": frame_idx,
        "max_severity_score": 0.0,
        "max_proximity_index": proximity_index,
        "lateral_position": lateral_position,
    }


def _update_event(
    event: dict[str, Any],
    frame_idx: int,
    proximity_index: float,
    severity_score: float,
    incident_type: str,
    lateral_position: float,
) -> None:
    """Update the open event. Escalates ahead -> critical, never the reverse."""
    event["end_frame"] = frame_idx
    if proximity_index >= event["max_proximity_index"]:
        event["lateral_position"] = lateral_position
    event["max_proximity_index"] = max(event["max_proximity_index"], proximity_index)
    event["max_severity_score"] = max(event["max_severity_score"], severity_score)
    if (
        incident_type == "pedestrian_critical"
        and event["incident_type"] != "pedestrian_critical"
    ):
        event["incident_type"] = "pedestrian_critical"
        _log_event(
            "ESCALATE",
            "pedestrian_critical",
            event["track_id"],
            frame_idx,
            proximity_index=f"{proximity_index:.3f}",
        )


def _finalize_event(event: dict[str, Any]) -> dict[str, Any]:
    _log_event(
        "END",
        event["incident_type"],
        event["track_id"],
        event["end_frame"],
        start_frame=event["start_frame"],
        max_severity=f"{event['max_severity_score']:.3f}",
        max_proximity_index=f"{event['max_proximity_index']:.3f}",
        lateral_position=f"{event['lateral_position']:.3f}",
    )
    return {
        "track_id": event["track_id"],
        "incident_type": event["incident_type"],
        "start_frame": event["start_frame"],
        "end_frame": event["end_frame"],
        "max_severity_score": event["max_severity_score"],
        "max_proximity_index": event["max_proximity_index"],
        "lateral_position": event["lateral_position"],
    }


def _close_active_event(
    event: dict[str, Any] | None,
    incidents: list[dict[str, Any]],
) -> None:
    if event is not None:
        incidents.append(_finalize_event(event))


def _reset_track_state(
    state: _PedestrianState,
    incidents: list[dict[str, Any]],
) -> None:
    """Close the open event and reset counters."""
    _close_active_event(state.active_event, incidents)
    state.active_event = None
    state.consecutive_threat_frames = 0
    state.consecutive_out_of_lane_frames = 0
    state.ground_history.clear()


def analyze_pedestrian_safety(
    tracked_objects: Sequence[Sequence[Any]],
    frame_width: int,
    frame_height: int,
    *,
    lane_width_fraction: float = 0.40,
    horizon_fraction: float = 0.35,
    rolling_window: int = 10,
    consecutive_threat_threshold: int = 5,
    critical_proximity_threshold: float = CRITICAL_PROXIMITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detect in-lane pedestrian threats (ahead vs critical) from tracked objects."""
    track_states: dict[int, _PedestrianState] = defaultdict(
        lambda: _PedestrianState(ground_history=deque(maxlen=rolling_window))
    )
    incidents: list[dict[str, Any]] = []

    for frame_idx, frame_objects in enumerate(tracked_objects):
        active_track_ids: set[int] = set()

        for obj in frame_objects:
            track_id, cls_name, box = _parse_object(obj)
            if cls_name not in PEDESTRIAN_CLASSES:
                continue

            ground_x, ground_y = _ground_contact_point(box)
            in_lane = _is_in_forward_lane(
                ground_x,
                ground_y,
                frame_width,
                frame_height,
                lane_width_fraction,
                horizon_fraction,
            )

            # still tracked even if out of lane
            active_track_ids.add(track_id)
            state = track_states[track_id]

            if not in_lane:
                if state.active_event is not None:
                    # close after N consecutive out-of-lane frames
                    state.consecutive_out_of_lane_frames += 1
                    if (
                        state.consecutive_out_of_lane_frames
                        >= consecutive_threat_threshold
                    ):
                        _reset_track_state(state, incidents)
                else:
                    # no open event — drop the entry counter
                    state.consecutive_threat_frames = 0
                    state.consecutive_out_of_lane_frames = 0
                continue

            state.consecutive_out_of_lane_frames = 0
            state.ground_history.append((ground_x, ground_y))
            proximity = _proximity_index(ground_y, frame_height)
            lateral_position = _lateral_position(
                ground_x, frame_width, lane_width_fraction
            )
            incident_type = (
                "pedestrian_critical"
                if proximity > critical_proximity_threshold
                else "pedestrian_ahead"
            )

            state.consecutive_threat_frames += 1
            threat_confirmed = (
                state.consecutive_threat_frames >= consecutive_threat_threshold
            )

            if not threat_confirmed:
                continue

            severity_score = proximity * state.consecutive_threat_frames

            if state.active_event is None:
                state.active_event = _start_event(
                    incident_type,
                    track_id,
                    frame_idx,
                    proximity,
                    lateral_position,
                )
            _update_event(
                state.active_event,
                frame_idx,
                proximity,
                severity_score,
                incident_type,
                lateral_position,
            )

        # tracks missing this frame
        for track_id, state in list(track_states.items()):
            if track_id in active_track_ids:
                continue
            _reset_track_state(state, incidents)

    for state in track_states.values():
        _close_active_event(state.active_event, incidents)
        state.active_event = None

    return incidents
