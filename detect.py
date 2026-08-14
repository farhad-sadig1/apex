import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from incidents.lateral_safety import analyze_lateral_safety
from incidents.metrics import calculate_ride_metrics
from incidents.pedestrian_safety import analyze_pedestrian_safety

INPUT_PATH = Path("test_clips/ride1.mp4")
OUTPUT_PATH = Path("test_clips/ride1_detected.mp4")
TELEMETRY_PATH = Path("test_clips/ride1_telemetry.json")
# COCO class IDs: person=0, bicycle=1, car=2, bus=5, truck=7
TARGET_CLASSES = [0, 1, 2, 5, 7]

# match analyzer defaults
HORIZON_FRACTION = 0.35
LANE_WIDTH_FRACTION = 0.40
SIDE_FRACTION = 0.35
LATERAL_PROXIMITY_THRESHOLD = 0.25
CRITICAL_PROXIMITY_THRESHOLD = 0.15
MAX_GAP_PIXEL_FRACTION = 0.15
MIN_RELEVANT_SIZE_FRACTION = 0.12  # ignore distant / tiny vehicle boxes
PEDESTRIAN_CRITICAL_DEPTH = 0.65  # bottom 35% of the frame

CYAN = (255, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
GREEN = (0, 255, 0)


def extract_visible_tracks(result) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    visible = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return visible

    for box in boxes:
        track_id = int(box.id.item())
        cls_name = result.names[int(box.cls.item())]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        visible.append((track_id, cls_name, (x1, y1, x2, y2)))
    return visible


def _draw_label(
    frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 2, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def render_debug_overlay(
    frame,
    visible: list[tuple[int, str, tuple[int, int, int, int]]],
    width: int,
    height: int,
):
    """Per-frame overlay. Incidents and scores are computed after the video loop."""
    annotated_frame = frame.copy()

    horizon_y = int(height * HORIZON_FRACTION)
    lane_half = int((width * LANE_WIDTH_FRACTION) / 2.0)
    lane_center = width // 2
    left_x = lane_center - lane_half
    right_x = lane_center + lane_half

    cv2.line(annotated_frame, (left_x, horizon_y), (left_x, height), CYAN, 2)
    cv2.line(annotated_frame, (right_x, horizon_y), (right_x, height), CYAN, 2)
    cv2.line(annotated_frame, (left_x, horizon_y), (right_x, horizon_y), CYAN, 2)

    left_boundary = width * SIDE_FRACTION
    right_boundary = width * (1.0 - SIDE_FRACTION)

    for track_id, cls_name, (x1, y1, x2, y2) in visible:
        color = GREEN
        label = f"id={track_id} {cls_name}"

        if cls_name == "person":
            ground_x = (x1 + x2) / 2.0
            ground_y = float(y2)
            in_lane = (
                ground_y >= horizon_y
                and left_x <= ground_x <= right_x
            )
            depth = ground_y / float(height) if height > 0 else 0.0
            if in_lane and depth >= PEDESTRIAN_CRITICAL_DEPTH:
                color = RED
                label = f"id={track_id} [CRITICAL]"
            elif in_lane:
                color = ORANGE
                label = f"id={track_id} [AWARENESS]"
            else:
                color = GREEN
                label = f"id={track_id} person"

        elif cls_name in ("car", "truck", "bus"):
            center_x = (x1 + x2) / 2.0
            in_side_zone = center_x <= left_boundary or center_x >= right_boundary
            box_height = max(y2 - y1, 1)
            box_height_fraction = box_height / float(height) if height > 0 else 0.0
            # tiny boxes stay green — gap/height is noisy
            if in_side_zone and box_height_fraction > MIN_RELEVANT_SIZE_FRACTION:
                frame_center = width / 2.0
                if center_x <= frame_center:
                    pixel_gap = frame_center - x2
                else:
                    pixel_gap = x1 - frame_center
                gap_fraction = pixel_gap / float(width)
                normalized_gap = pixel_gap / box_height

                # same gates as lateral_safety.py
                if (
                    normalized_gap < CRITICAL_PROXIMITY_THRESHOLD
                    and gap_fraction < MAX_GAP_PIXEL_FRACTION
                ):
                    color = RED
                    label = f"id={track_id} [CLOSE]"
                elif (
                    normalized_gap < LATERAL_PROXIMITY_THRESHOLD
                    and gap_fraction < MAX_GAP_PIXEL_FRACTION
                ):
                    color = ORANGE
                    label = f"id={track_id} [CAUTION]"
                else:
                    color = GREEN
                    label = f"id={track_id} {cls_name}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated_frame, label, (x1, y1), color)

    return annotated_frame


def main() -> None:
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(INPUT_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

    tracked_objects: list[list[tuple[int, str, tuple[int, int, int, int]]]] = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=TARGET_CLASSES,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]
        visible = extract_visible_tracks(result)
        tracked_objects.append(visible)
        print(f"Frame {frame_count}: {visible}")

        # overlay only; incidents scored after the loop
        annotated_frame = render_debug_overlay(frame, visible, width, height)
        writer.write(annotated_frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_count} frames -> Visual logs stored in {OUTPUT_PATH}")

    # score the whole ride once
    lateral_incidents = analyze_lateral_safety(tracked_objects, width, height)
    pedestrian_incidents = analyze_pedestrian_safety(tracked_objects, width, height)

    all_incidents = lateral_incidents + pedestrian_incidents
    ride_data = calculate_ride_metrics(all_incidents, frame_count, fps)

    with open(TELEMETRY_PATH, "w", encoding="utf-8") as f:
        json.dump(ride_data, f, indent=4)
    print(f"Exported ride telemetry -> {TELEMETRY_PATH}")


if __name__ == "__main__":
    main()
