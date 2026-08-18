# Apex ([Cyclist Safety Auditor](https://apex-cyclist-safety.vercel.app))

A computer vision pipeline that watches cyclist ride footage and flags close passes, sustained unsafe proximity, and pedestrian conflicts. Plays the ride back through a dashboard showing exactly when and where each one happened.

<img src="demo_image.png" alt="Apex Dashboard" width="800" />

## Detection Pipeline

- YOLOv8 object detection (person, bicycle, car, truck, bus) with ByteTrack assigning persistent IDs across frames
- Dual gate lateral proximity check per vehicle, using gap relative to the vehicle's own box height to account for depth, and gap as an absolute fraction of frame width to filter out vehicles that are nearby in pixels but a full lane away in reality
- Two lateral incident tiers, `dynamic_side_closure` for a fast closing gap and `sustained_side_proximity` for prolonged closeness
- Pedestrian forward collision detection using ground contact point as a distance proxy, restricted to the cyclist's forward corridor
- Hysteresis on both entry and exit for every incident type, so tracking jitter doesn't fragment one encounter into several
- Live status computed for each frame (`clear`, `awareness`, `risk`, `critical`) with no memory carried over, plus an end of ride summary: tier counts, a timestamped list of risk and critical events, and a hazard rate in events per minute



## Dashboard

- Dual pane view, the annotated ride video on one side, an overhead radar HUD on the other
- Video playback position drives the HUD's current frame, not an independent timer, so the two stay frame accurate against each other
- Projected road corridor in perspective with converging rails. The zone on either side glows when a lateral incident is active on that side
- Pedestrian marker positioned continuously from real detected coordinates, not snapped to fixed points
- End of ride report: tier counts, hazard rate, and a clickable list of risk and critical moments that seeks the video to that timestamp



## Repository Structure

- **Backend** (`detect.py`, `incidents/`) - Python CV pipeline: YOLOv8 detection, ByteTrack tracking, and incident classification
  - `incidents/lateral_safety.py` - dual gate vehicle proximity detection
  - `incidents/pedestrian_safety.py` - forward corridor pedestrian detection
  - `incidents/metrics.py` - per frame status and end of ride aggregation
- **Frontend** (`cyclist-aras/`) - Expo / React Native Web dashboard, video playback synced to the radar HUD
- **Sample data** (`test_clips/`) - ride footage used to generate the bundled demo telemetry



## Running Locally

1. Clone the repository:

```bash
git clone https://github.com/farhad-sadig1/apex.git
cd apex
```

1. Set up the backend (requires Python 3.10+):

```bash
pip install -r requirements.txt
python3 detect.py
```

Reads `test_clips/ride1.mp4` and writes an annotated video plus `ride1_telemetry.json` alongside it.

Re-encode the output so it plays correctly in the browser:

```bash
ffmpeg -i test_clips/ride1_detected.mp4 -c:v libx264 -pix_fmt yuv420p -c:a aac cyclist-aras/public/ride1_detected.mp4
```

1. Set up the frontend (requires Node.js 18+):

```bash
cd cyclist-aras
npm install
npx expo start --web
```

Points at the sample telemetry and video bundled in `cyclist-aras/assets` and `cyclist-aras/public`.
