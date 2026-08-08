import cv2
from pathlib import Path

from ultralytics import YOLO

INPUT_PATH = Path("test_clips/ride1.mp4")
OUTPUT_PATH = Path("test_clips/ride1_detected.mp4")
# COCO class IDs: person=0, bicycle=1, car=2
TARGET_CLASSES = [0, 1, 2]


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
        print(f"Frame {frame_count}: {visible}")

        annotated = result.plot()
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_count} frames -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
