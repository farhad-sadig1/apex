/**
 * Cyclist ARAS cockpit — dual-screen ride auditor.
 *
 * Left: processed detection video (timing source), served from
 * `public/ride1_detected.mp4` as `/ride1_detected.mp4`.
 * Right: Tesla-style sonar HUD driven by the video clock.
 *
 * Frame index is derived from native playback position:
 *   currentFrame = floor(videoCurrentTimeSeconds * summary.fps)
 * Web uses HTML5 <video> timeupdate (+ rAF while playing).
 * Native uses expo-video timeUpdate (SDK 57 successor to
 * expo-av onPlaybackStatusUpdate). The video is never seeked
 * from a radar interval.
 */
import { useEventListener } from 'expo';
import { StatusBar } from 'expo-status-bar';
import { useVideoPlayer, VideoView } from 'expo-video';
import {
  createElement,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import Svg, {
  Circle,
  Defs,
  G,
  LinearGradient,
  Path,
  Rect,
  Stop,
  Text as SvgText,
} from 'react-native-svg';

import telemetryJson from './assets/telemetry/ride1_telemetry.json';

// ---------------------------------------------------------------------------
// Domain types — mirrors the backend ride telemetry payload
// ---------------------------------------------------------------------------

/** Lateral / pedestrian incident classes emitted by the Python pipeline. */
type IncidentType =
  | 'dynamic_side_closure'
  | 'sustained_side_proximity'
  | 'pedestrian_ahead'
  | 'pedestrian_critical';

type FlankSide = 'left' | 'right';

interface TelemetryIncident {
  track_id: number;
  incident_type: IncidentType | string;
  start_frame: number;
  end_frame: number;
  max_severity_score: number;
  min_normalized_gap?: number;
  max_proximity_index?: number;
  /** Optional until the exporter stamps overtake side onto lateral events. */
  side?: FlankSide | string;
}

interface RideSummary {
  initial_score: number;
  final_safety_score: number;
  total_incidents_detected: number;
  ride_duration_seconds: number;
  fps?: number;
}

interface TelemetryLog {
  summary: RideSummary;
  incidents: TelemetryIncident[];
}

/** Per-tick boolean raster that lights the four cockpit sectors. */
interface SectorThreats {
  leftCaution: boolean;
  leftClose: boolean;
  rightCaution: boolean;
  rightClose: boolean;
  pedestrianAhead: boolean;
  pedestrianCritical: boolean;
}

type FlankLevel = 'safe' | 'caution' | 'close';
type CorridorLevel = 'safe' | 'ahead' | 'critical';
type AlertTone = 'safe' | 'caution' | 'critical';

interface PrimaryAlert {
  title: string;
  tone: AlertTone;
}

interface CanvasSize {
  width: number;
  height: number;
}

interface PlaybackController {
  play: () => void;
  pause: () => void;
  reset: () => void;
}

// ---------------------------------------------------------------------------
// Design tokens — Tesla-style dark cockpit
// ---------------------------------------------------------------------------

const COLOR = {
  bg: '#000000',
  text: '#F4F6F8',
  muted: '#8B9198',
  hairline: '#1C1F24',
  cyan: '#6EE7F9',
  cyanDeep: '#143A42',
  greyBand: '#2A3036',
  orange: '#FF8A3D',
  red: '#FF2A2A',
  island: '#0B0D10',
  vehicle: '#3A4046',
  atmosphere: '#1A2228',
} as const;

/** Statically served from `cyclist-aras/public/` — not a Metro `require()`. */
const RIDE_VIDEO_URI = '/ride1_detected.mp4';

/**
 * Minimum on-glass persistence for a threat. A 1–2 frame backend event is
 * otherwise invisible (~33–66 ms). 12 frames ≈ 400 ms, the lower bound for
 * a glanceable HUD cue.
 */
const MIN_HUD_FRAMES = 12;

/** Keep in lockstep with incidents/lateral_safety.py. */
const LATERAL_PROXIMITY_THRESHOLD = 0.25;

const VIEW_W = 400;
const VIEW_H = 700;
const BIKE_X = 200;
const BIKE_Y = 502;

/** Trapezoid road: near and far widths stay close; gradual perspective only. */
const HORIZON_X = BIKE_X;
const HORIZON_Y = 64;
const PATH_NEAR_Y = BIKE_Y - 36;
const PATH_NEAR_HALF = 100;
const PATH_FAR_HALF = 82;
/** Translucent flank panels sit just outside each rail. */
const ZONE_PAD_NEAR = 42;
const ZONE_PAD_FAR = 30;

const PERSPECTIVE_LINES = 14;

interface Point {
  x: number;
  y: number;
}

// ---------------------------------------------------------------------------
// Telemetry bootstrap (module scope — parse once)
// ---------------------------------------------------------------------------

function loadTelemetry(raw: unknown): TelemetryLog {
  const data = raw as TelemetryLog;
  if (!data?.summary || !Array.isArray(data.incidents)) {
    throw new Error('Invalid ARAS telemetry payload.');
  }
  return data;
}

const TELEMETRY = loadTelemetry(telemetryJson);
const SUMMARY = TELEMETRY.summary;
const INCIDENTS: readonly TelemetryIncident[] = TELEMETRY.incidents;

const PLAYBACK_FPS =
  SUMMARY.fps && SUMMARY.fps > 0 ? SUMMARY.fps : 30;

const MAX_INCIDENT_FRAME = INCIDENTS.reduce(
  (max, incident) => Math.max(max, incident.end_frame),
  0,
);

/** Inclusive playback length: frames [0, FRAME_COUNT - 1]. */
const FRAME_COUNT = Math.max(
  1,
  Math.round(SUMMARY.ride_duration_seconds * PLAYBACK_FPS),
  MAX_INCIDENT_FRAME + 1,
);

const LAST_FRAME = FRAME_COUNT - 1;

function incidentDeduction(incident: TelemetryIncident, fps: number): number {
  switch (incident.incident_type) {
    case 'dynamic_side_closure': {
      const intensity = Math.tanh(incident.max_severity_score * 10);
      return 15 * intensity;
    }
    case 'sustained_side_proximity': {
      const gap = incident.min_normalized_gap ?? LATERAL_PROXIMITY_THRESHOLD;
      const proximityMultiplier =
        gap < LATERAL_PROXIMITY_THRESHOLD
          ? (LATERAL_PROXIMITY_THRESHOLD - gap) / LATERAL_PROXIMITY_THRESHOLD
          : 0;
      return 10 * (1 + proximityMultiplier);
    }
    case 'pedestrian_ahead':
    case 'pedestrian_critical': {
      const durationSeconds =
        (incident.end_frame - incident.start_frame + 1) / fps;
      const maxProximity = incident.max_proximity_index ?? 0.5;
      const durationFactor = Math.tanh(durationSeconds / 3);
      const base = incident.incident_type === 'pedestrian_ahead' ? 8 : 15;
      return base * (0.5 + 0.5 * durationFactor) * (0.5 + 0.5 * maxProximity);
    }
    default:
      return 0;
  }
}

const RAW_DEDUCTIONS = INCIDENTS.map((incident) =>
  incidentDeduction(incident, PLAYBACK_FPS),
);
const RAW_TOTAL = RAW_DEDUCTIONS.reduce((sum, value) => sum + value, 0);
const TARGET_DROP = SUMMARY.initial_score - SUMMARY.final_safety_score;
const SCALED_DEDUCTIONS = RAW_DEDUCTIONS.map((value) =>
  RAW_TOTAL > 0 ? (value / RAW_TOTAL) * TARGET_DROP : 0,
);

/** `currentFrame = floor(video.currentTime * summary.fps)`. */
function frameFromVideoTime(seconds: number): number {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return 0;
  }
  return Math.min(LAST_FRAME, Math.floor(seconds * PLAYBACK_FPS));
}

// ---------------------------------------------------------------------------
// Threat extraction + live score
// ---------------------------------------------------------------------------

function isVisuallyActive(incident: TelemetryIncident, frame: number): boolean {
  const visualEnd = Math.max(
    incident.end_frame,
    incident.start_frame + MIN_HUD_FRAMES - 1,
  );
  return frame >= incident.start_frame && frame <= visualEnd;
}

/**
 * Side-vehicle events may omit `side`. Fail visible: light both flanks rather
 * than hide a close-pass. Explicit `left` / `right` win when present.
 */
function resolveFlank(incident: TelemetryIncident): FlankSide | 'both' {
  if (incident.side === 'left' || incident.side === 'right') {
    return incident.side;
  }
  return 'both';
}

function applyFlank(
  threats: SectorThreats,
  flank: FlankSide | 'both',
  kind: 'caution' | 'close',
): void {
  const sides: FlankSide[] =
    flank === 'both' ? ['left', 'right'] : [flank];
  for (const side of sides) {
    if (kind === 'caution') {
      if (side === 'left') threats.leftCaution = true;
      else threats.rightCaution = true;
    } else if (side === 'left') {
      threats.leftClose = true;
    } else {
      threats.rightClose = true;
    }
  }
}

function extractActiveThreats(
  incidents: readonly TelemetryIncident[],
  frame: number,
): SectorThreats {
  const threats: SectorThreats = {
    leftCaution: false,
    leftClose: false,
    rightCaution: false,
    rightClose: false,
    pedestrianAhead: false,
    pedestrianCritical: false,
  };

  for (const incident of incidents) {
    if (!isVisuallyActive(incident, frame)) {
      continue;
    }

    switch (incident.incident_type) {
      case 'pedestrian_ahead':
        threats.pedestrianAhead = true;
        break;
      case 'pedestrian_critical':
        threats.pedestrianCritical = true;
        break;
      case 'sustained_side_proximity':
        applyFlank(threats, resolveFlank(incident), 'caution');
        break;
      case 'dynamic_side_closure':
        applyFlank(threats, resolveFlank(incident), 'close');
        break;
      default:
        break;
    }
  }

  return threats;
}

function extractActiveIncidents(
  incidents: readonly TelemetryIncident[],
  frame: number,
): TelemetryIncident[] {
  return incidents.filter((incident) => isVisuallyActive(incident, frame));
}

function exposureWeight(frame: number, incident: TelemetryIncident): number {
  if (frame < incident.start_frame) {
    return 0;
  }
  const span = Math.max(
    incident.end_frame - incident.start_frame + 1,
    MIN_HUD_FRAMES,
  );
  return Math.min(1, (frame - incident.start_frame + 1) / span);
}

/** Live score: 100 at frame 0, eases toward final_safety_score as hazards fire. */
function safetyScoreAt(frame: number): number {
  let drop = 0;
  for (let index = 0; index < INCIDENTS.length; index += 1) {
    drop += SCALED_DEDUCTIONS[index] * exposureWeight(frame, INCIDENTS[index]);
  }
  return Math.max(0, SUMMARY.initial_score - drop);
}

function flankLevel(caution: boolean, close: boolean): FlankLevel {
  if (close) return 'close';
  if (caution) return 'caution';
  return 'safe';
}

function corridorLevel(threats: SectorThreats): CorridorLevel {
  if (threats.pedestrianCritical) return 'critical';
  if (threats.pedestrianAhead) return 'ahead';
  return 'safe';
}

function primaryAlert(threats: SectorThreats): PrimaryAlert {
  if (threats.pedestrianCritical) {
    return { title: 'PEDESTRIAN COLLISION WARNING', tone: 'critical' };
  }
  if (threats.leftClose || threats.rightClose) {
    const side =
      threats.leftClose && threats.rightClose
        ? 'BOTH'
        : threats.leftClose
          ? 'LEFT'
          : 'RIGHT';
    return { title: `SIDE VEHICLE CLOSE · ${side}`, tone: 'critical' };
  }
  if (threats.pedestrianAhead) {
    return { title: 'PEDESTRIAN AHEAD', tone: 'caution' };
  }
  if (threats.leftCaution || threats.rightCaution) {
    const side =
      threats.leftCaution && threats.rightCaution
        ? 'BOTH'
        : threats.leftCaution
          ? 'LEFT'
          : 'RIGHT';
    return { title: `SIDE VEHICLE CAUTION · ${side}`, tone: 'caution' };
  }
  return { title: 'CORRIDOR CLEAR', tone: 'safe' };
}

function hexChannel(hex: string, shift: number): number {
  return (parseInt(hex.slice(1), 16) >> shift) & 0xff;
}

function mixHex(from: string, to: string, t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const mix = (shift: number) =>
    Math.round(
      hexChannel(from, shift) +
        (hexChannel(to, shift) - hexChannel(from, shift)) * clamped,
    );
  const r = mix(16);
  const g = mix(8);
  const b = mix(0);
  return `#${[r, g, b].map((value) => value.toString(16).padStart(2, '0')).join('')}`;
}

/** Compact meter hue: cyan (healthy) → orange → red (depleted). */
function scoreHue(score: number): string {
  if (score >= 70) {
    return mixHex(COLOR.orange, COLOR.cyan, (score - 70) / 30);
  }
  return mixHex(COLOR.red, COLOR.orange, score / 70);
}

function formatClock(frame: number): string {
  const totalSeconds = Math.floor(frame / PLAYBACK_FPS);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function padFrame(frame: number): string {
  return frame.toString().padStart(4, '0');
}

function pulse(frame: number): number {
  return 0.42 + 0.28 * Math.sin(frame * 0.21);
}

function flashOn(frame: number, period: number): boolean {
  return frame % period < period / 2;
}

function flankPaint(
  level: FlankLevel,
  frame: number,
): { color: string; opacity: number } {
  if (level === 'close') {
    return {
      color: COLOR.red,
      opacity: flashOn(frame, 4) ? 1 : 0.22,
    };
  }
  if (level === 'caution') {
    return { color: COLOR.orange, opacity: 0.92 };
  }
  return { color: COLOR.cyan, opacity: pulse(frame) };
}

function corridorPaint(
  level: CorridorLevel,
  frame: number,
): { color: string; opacity: number } {
  if (level === 'critical') {
    return {
      color: COLOR.red,
      opacity: flashOn(frame, 2) ? 0.95 : 0.18,
    };
  }
  if (level === 'ahead') {
    return { color: COLOR.orange, opacity: 0.88 };
  }
  return { color: COLOR.cyan, opacity: 0.62 + pulse(frame) * 0.28 };
}

// ---------------------------------------------------------------------------
// SVG geometry
// ---------------------------------------------------------------------------

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function lerp(from: number, to: number, t: number): number {
  return from + (to - from) * t;
}

function halfWidthAt(t: number, nearHalf: number, farHalf: number): number {
  return lerp(nearHalf, farHalf, t);
}

function pathAnchors(
  side: FlankSide,
  nearHalf = PATH_NEAR_HALF,
  farHalf = PATH_FAR_HALF,
): [Point, Point, Point, Point] {
  const sign = side === 'left' ? -1 : 1;
  const y0 = PATH_NEAR_Y;
  const y3 = HORIZON_Y;
  return [
    { x: BIKE_X + sign * nearHalf, y: y0 },
    { x: BIKE_X + sign * halfWidthAt(1 / 3, nearHalf, farHalf), y: lerp(y0, y3, 1 / 3) },
    { x: BIKE_X + sign * halfWidthAt(2 / 3, nearHalf, farHalf), y: lerp(y0, y3, 2 / 3) },
    { x: BIKE_X + sign * farHalf, y: y3 },
  ];
}

function cubicPoint(points: [Point, Point, Point, Point], t: number): Point {
  const u = 1 - t;
  const tt = t * t;
  const uu = u * u;
  return {
    x:
      uu * u * points[0].x +
      3 * uu * t * points[1].x +
      3 * u * tt * points[2].x +
      tt * t * points[3].x,
    y:
      uu * u * points[0].y +
      3 * uu * t * points[1].y +
      3 * u * tt * points[2].y +
      tt * t * points[3].y,
  };
}

function projectedLanePath(side: FlankSide): string {
  const [p0, p1, p2, p3] = pathAnchors(side);
  return `M ${p0.x} ${p0.y} C ${p1.x} ${p1.y}, ${p2.x} ${p2.y}, ${p3.x} ${p3.y}`;
}

function flankZonePath(side: FlankSide): string {
  const inner = pathAnchors(side);
  const outer = pathAnchors(
    side,
    PATH_NEAR_HALF + ZONE_PAD_NEAR,
    PATH_FAR_HALF + ZONE_PAD_FAR,
  );
  const [i0, i1, i2, i3] = inner;
  const [o0, o1, o2, o3] = outer;
  return [
    `M ${i0.x} ${i0.y}`,
    `C ${i1.x} ${i1.y}, ${i2.x} ${i2.y}, ${i3.x} ${i3.y}`,
    `L ${o3.x} ${o3.y}`,
    `C ${o2.x} ${o2.y}, ${o1.x} ${o1.y}, ${o0.x} ${o0.y}`,
    'Z',
  ].join(' ');
}

function BicycleGlyph() {
  const fill = COLOR.text;
  return (
    <G>
      <Path
        d="M-15 34 C-15 48 15 48 15 34 L12 -26 C12 -40 -12 -40 -12 -26 Z"
        fill={fill}
      />
      <Rect x={-23} y={-44} width={46} height={11} rx={5.5} fill={fill} />
      <Circle cx={0} cy={-4} r={7} fill={COLOR.bg} opacity={0.28} />
    </G>
  );
}

function Atmosphere() {
  return (
    <G>
      <Defs>
        <LinearGradient id="atmosphere" x1="0%" y1="100%" x2="0%" y2="0%">
          <Stop offset="0" stopColor={COLOR.bg} stopOpacity={0} />
          <Stop offset="0.42" stopColor={COLOR.bg} stopOpacity={0.15} />
          <Stop offset="1" stopColor={COLOR.atmosphere} stopOpacity={0.42} />
        </LinearGradient>
      </Defs>
      <Rect
        x={0}
        y={0}
        width={VIEW_W}
        height={PATH_NEAR_Y}
        fill="url(#atmosphere)"
      />
    </G>
  );
}

function PerspectiveGround() {
  const lines = [];
  for (let index = 0; index < PERSPECTIVE_LINES; index += 1) {
    const u = index / (PERSPECTIVE_LINES - 1);
    const zNear = 1.12;
    const zFar = 9;
    const z = lerp(zNear, zFar, u);
    const persp = (1 - 1 / z) / (1 - 1 / zFar);
    const left = cubicPoint(pathAnchors('left'), persp);
    const right = cubicPoint(pathAnchors('right'), persp);
    const opacity = lerp(0.26, 0.04, persp);
    lines.push(
      <Path
        key={`ground-${index}`}
        d={`M ${left.x} ${left.y} L ${right.x} ${right.y}`}
        stroke={COLOR.cyan}
        strokeWidth={1}
        fill="none"
        opacity={opacity}
      />,
    );
  }
  return <G>{lines}</G>;
}

function FlankZone({
  side,
  level,
  frame,
}: {
  side: FlankSide;
  level: FlankLevel;
  frame: number;
}) {
  const live = flankPaint(level, frame);
  const gradientId = `flank-zone-${side}`;
  const labelX = side === 'left' ? 28 : 372;
  const labelAnchor = side === 'left' ? 'start' : 'end';
  const label = side === 'left' ? 'LEFT' : 'RIGHT';
  const labelColor =
    level === 'close' ? COLOR.red : level === 'caution' ? COLOR.orange : COLOR.muted;
  const innerX = side === 'left' ? '100%' : '0%';
  const outerX = side === 'left' ? '0%' : '100%';

  return (
    <G>
      <Defs>
        <LinearGradient
          id={gradientId}
          x1={innerX}
          y1="0%"
          x2={outerX}
          y2="0%"
        >
          <Stop offset="0" stopColor={live.color} stopOpacity={Math.min(0.55, live.opacity * 0.5)} />
          <Stop offset="1" stopColor={live.color} stopOpacity={0} />
        </LinearGradient>
      </Defs>
      <Path d={flankZonePath(side)} fill={`url(#${gradientId})`} />
      <Path
        d={flankZonePath(side)}
        fill={live.color}
        opacity={Math.min(0.22, live.opacity * 0.22)}
      />
      <SvgText
        x={labelX}
        y={BIKE_Y + 4}
        fill={labelColor}
        fontSize={11}
        fontWeight="600"
        letterSpacing={2.4}
        textAnchor={labelAnchor}
      >
        {label}
      </SvgText>
    </G>
  );
}

function PathVector({
  level,
  frame,
}: {
  level: CorridorLevel;
  frame: number;
}) {
  const live = corridorPaint(level, frame);
  const left = projectedLanePath('left');
  const right = projectedLanePath('right');
  const strokeWidth = level === 'safe' ? 2.35 : 2.9;
  const rails: FlankSide[] = ['left', 'right'];

  return (
    <G>
      {rails.map((side) => {
        const d = side === 'left' ? left : right;
        return (
          <G key={`rail-${side}`}>
            <Path
              d={d}
              stroke={live.color}
              strokeWidth={14}
              strokeLinecap="round"
              fill="none"
              opacity={live.opacity * 0.14}
            />
            <Path
              d={d}
              stroke={live.color}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              fill="none"
              opacity={live.opacity}
            />
          </G>
        );
      })}
      <SvgText
        x={HORIZON_X}
        y={HORIZON_Y - 14}
        fill={
          level === 'critical'
            ? COLOR.red
            : level === 'ahead'
              ? COLOR.orange
              : COLOR.muted
        }
        fontSize={11}
        fontWeight="600"
        letterSpacing={3}
        textAnchor="middle"
      >
        AHEAD
      </SvgText>
    </G>
  );
}

function vehicleDepthT(gap: number): number {
  return clamp01(0.12 + (gap / 0.45) * 0.6);
}

function VehicleMark({
  incident,
  side,
}: {
  incident: TelemetryIncident;
  side: FlankSide;
}) {
  const gap = incident.min_normalized_gap ?? LATERAL_PROXIMITY_THRESHOLD;
  const t = vehicleDepthT(gap);
  const rail = cubicPoint(pathAnchors(side), t);
  const scale = lerp(1.12, 0.32, t);
  const width = 48 * scale;
  const height = 30 * scale;
  const sign = side === 'left' ? -1 : 1;
  const x = rail.x + sign * (width * 0.55 + 6 * scale);
  const y = rail.y;
  const accent =
    incident.incident_type === 'dynamic_side_closure' ? COLOR.red : COLOR.orange;

  return (
    <Rect
      x={x - width / 2}
      y={y - height / 2}
      width={width}
      height={height}
      rx={5 * scale}
      ry={5 * scale}
      fill={COLOR.vehicle}
      stroke={accent}
      strokeWidth={Math.max(1.2, 1.8 * scale)}
      opacity={0.94}
    />
  );
}

function PedestrianMark({
  incident,
  frame,
}: {
  incident: TelemetryIncident;
  frame: number;
}) {
  const proximity = clamp01(incident.max_proximity_index ?? 0.5);
  const t = 1 - proximity;
  const point = cubicPoint(pathAnchors('left'), t);
  const radius = 4.4 + proximity * 4.2;
  const critical = incident.incident_type === 'pedestrian_critical';
  const color = critical ? COLOR.red : COLOR.orange;
  const opacity = critical ? (flashOn(frame, 2) ? 1 : 0.2) : 0.92;

  return (
    <Circle
      cx={HORIZON_X}
      cy={point.y}
      r={radius}
      fill={color}
      opacity={opacity}
    />
  );
}

function IncidentActors({
  incidents,
  frame,
}: {
  incidents: readonly TelemetryIncident[];
  frame: number;
}) {
  const actors = [];
  for (const incident of incidents) {
    const kind = incident.incident_type;
    if (kind === 'dynamic_side_closure' || kind === 'sustained_side_proximity') {
      const flank = resolveFlank(incident);
      const sides: FlankSide[] = flank === 'both' ? ['left', 'right'] : [flank];
      for (const side of sides) {
        actors.push(
          <VehicleMark
            key={`veh-${incident.track_id}-${incident.start_frame}-${side}`}
            incident={incident}
            side={side}
          />,
        );
      }
    } else if (kind === 'pedestrian_ahead' || kind === 'pedestrian_critical') {
      actors.push(
        <PedestrianMark
          key={`ped-${incident.track_id}-${incident.start_frame}`}
          incident={incident}
          frame={frame}
        />,
      );
    }
  }
  return <G>{actors}</G>;
}

function CockpitCanvas({
  threats,
  incidents,
  frame,
  size,
}: {
  threats: SectorThreats;
  incidents: readonly TelemetryIncident[];
  frame: number;
  size: CanvasSize;
}) {
  const left = flankLevel(threats.leftCaution, threats.leftClose);
  const right = flankLevel(threats.rightCaution, threats.rightClose);
  const corridor = corridorLevel(threats);

  return (
    <Svg
      width={size.width}
      height={size.height}
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <Atmosphere />
      <PerspectiveGround />
      <FlankZone side="left" level={left} frame={frame} />
      <FlankZone side="right" level={right} frame={frame} />
      <PathVector level={corridor} frame={frame} />
      <IncidentActors incidents={incidents} frame={frame} />
      <G x={BIKE_X} y={BIKE_Y}>
        <BicycleGlyph />
      </G>
    </Svg>
  );
}

// ---------------------------------------------------------------------------
// Video clock — HTML5 on web, expo-video on native
// ---------------------------------------------------------------------------

function WebRideVideo({
  uri,
  onTimeSeconds,
  onPlayingChange,
  controllerRef,
}: {
  uri: string;
  onTimeSeconds: (seconds: number) => void;
  onPlayingChange: (playing: boolean) => void;
  controllerRef: React.MutableRefObject<PlaybackController | null>;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const onTimeRef = useRef(onTimeSeconds);
  const onPlayingRef = useRef(onPlayingChange);
  onTimeRef.current = onTimeSeconds;
  onPlayingRef.current = onPlayingChange;

  useEffect(() => {
    const node = videoRef.current;
    if (!node) {
      return undefined;
    }

    let raf = 0;
    const emit = () => {
      onTimeRef.current(node.currentTime);
    };
    const loop = () => {
      emit();
      raf = requestAnimationFrame(loop);
    };
    const stopLoop = () => {
      cancelAnimationFrame(raf);
      raf = 0;
    };

    const onTimeUpdate = () => {
      emit();
    };
    const onPlay = () => {
      onPlayingRef.current(true);
      stopLoop();
      raf = requestAnimationFrame(loop);
    };
    const onPause = () => {
      stopLoop();
      onPlayingRef.current(false);
      emit();
    };
    const onEnded = () => {
      stopLoop();
      onPlayingRef.current(false);
      emit();
    };
    const onSeeked = () => {
      emit();
    };

    controllerRef.current = {
      play: () => {
        void node.play();
      },
      pause: () => {
        node.pause();
      },
      reset: () => {
        stopLoop();
        node.pause();
        node.currentTime = 0;
        onTimeRef.current(0);
        onPlayingRef.current(false);
      },
    };

    node.addEventListener('timeupdate', onTimeUpdate);
    node.addEventListener('play', onPlay);
    node.addEventListener('pause', onPause);
    node.addEventListener('ended', onEnded);
    node.addEventListener('seeked', onSeeked);

    void node.play().catch(() => {
      onPlayingRef.current(false);
    });

    return () => {
      stopLoop();
      node.removeEventListener('timeupdate', onTimeUpdate);
      node.removeEventListener('play', onPlay);
      node.removeEventListener('pause', onPause);
      node.removeEventListener('ended', onEnded);
      node.removeEventListener('seeked', onSeeked);
      controllerRef.current = null;
    };
  }, [controllerRef, uri]);

  return (
    <View style={styles.videoSurface}>
      {createElement('video', {
        ref: videoRef,
        src: uri,
        playsInline: true,
        preload: 'auto',
        controls: false,
        disablePictureInPicture: true,
        style: {
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          backgroundColor: COLOR.bg,
          display: 'block',
        },
      })}
    </View>
  );
}

function NativeRideVideo({
  source,
  onTimeSeconds,
  onPlayingChange,
  controllerRef,
}: {
  source: string;
  onTimeSeconds: (seconds: number) => void;
  onPlayingChange: (playing: boolean) => void;
  controllerRef: React.MutableRefObject<PlaybackController | null>;
}) {
  const onTimeRef = useRef(onTimeSeconds);
  const onPlayingRef = useRef(onPlayingChange);
  onTimeRef.current = onTimeSeconds;
  onPlayingRef.current = onPlayingChange;

  const player = useVideoPlayer(source, (instance) => {
    instance.timeUpdateEventInterval = 1 / PLAYBACK_FPS;
  });

  useEventListener(player, 'timeUpdate', ({ currentTime }) => {
    onTimeRef.current(currentTime);
  });

  useEventListener(player, 'playingChange', ({ isPlaying }) => {
    onPlayingRef.current(isPlaying);
  });

  useEventListener(player, 'playToEnd', () => {
    onPlayingRef.current(false);
  });

  useEffect(() => {
    controllerRef.current = {
      play: () => {
        player.play();
      },
      pause: () => {
        player.pause();
      },
      reset: () => {
        player.pause();
        player.currentTime = 0;
        onTimeRef.current(0);
        onPlayingRef.current(false);
      },
    };

    player.play();

    return () => {
      controllerRef.current = null;
    };
  }, [controllerRef, player]);

  return (
    <VideoView
      player={player}
      style={styles.videoSurface}
      nativeControls={false}
      contentFit="contain"
    />
  );
}

const RideFeed = memo(function RideFeed({
  onTimeSeconds,
  onPlayingChange,
  controllerRef,
}: {
  onTimeSeconds: (seconds: number) => void;
  onPlayingChange: (playing: boolean) => void;
  controllerRef: React.MutableRefObject<PlaybackController | null>;
}) {
  if (Platform.OS === 'web') {
    return (
      <WebRideVideo
        uri={RIDE_VIDEO_URI}
        onTimeSeconds={onTimeSeconds}
        onPlayingChange={onPlayingChange}
        controllerRef={controllerRef}
      />
    );
  }

  return (
    <NativeRideVideo
      source={RIDE_VIDEO_URI}
      onTimeSeconds={onTimeSeconds}
      onPlayingChange={onPlayingChange}
      controllerRef={controllerRef}
    />
  );
});

// ---------------------------------------------------------------------------
// Chrome: header + transport
// ---------------------------------------------------------------------------

function ScoreMeter({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(100, score));
  const hue = scoreHue(score);

  return (
    <View
      accessibilityRole="progressbar"
      accessibilityLabel={`Safety score ${Math.round(score)}`}
      style={styles.scoreMeter}
    >
      <Text style={styles.scoreMeterLabel}>SCORE</Text>
      <View style={styles.scoreTrack}>
        <View
          style={[
            styles.scoreFill,
            { width: `${pct}%`, backgroundColor: hue },
          ]}
        />
      </View>
      <Text style={[styles.scoreMeterValue, { color: hue }]}>
        {Math.round(score)}
      </Text>
    </View>
  );
}

function HeaderPanel({
  score,
  frame,
  playing,
}: {
  score: number;
  frame: number;
  playing: boolean;
}) {
  const progress = LAST_FRAME <= 0 ? 1 : frame / LAST_FRAME;

  return (
    <View style={styles.header}>
      <View style={styles.headerTop}>
        <Text style={styles.brand}>ARAS</Text>
        <Text style={styles.brandSub}>URBAN CYCLIST</Text>
        <ScoreMeter score={score} />
        <View style={styles.livePill}>
          <View
            style={[
              styles.liveDot,
              { backgroundColor: playing ? COLOR.red : COLOR.muted },
            ]}
          />
          <Text style={styles.liveText}>{playing ? 'LIVE' : 'PAUSED'}</Text>
        </View>
      </View>

      <View style={styles.progressBlock}>
        <View style={styles.progressMeta}>
          <Text style={styles.progressTitle}>TRIP PROGRESS</Text>
          <Text style={styles.progressClock}>
            {formatClock(frame)} / {formatClock(LAST_FRAME)}
          </Text>
        </View>
        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${progress * 100}%` }]} />
        </View>
      </View>
    </View>
  );
}

function ControlIsland({
  playing,
  atEnd,
  atStart,
  frame,
  onPlay,
  onPause,
  onReset,
}: {
  playing: boolean;
  atEnd: boolean;
  atStart: boolean;
  frame: number;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
}) {
  return (
    <View style={styles.controls}>
      <View style={styles.island}>
        <TransportButton
          label="PLAY"
          glyph="▶"
          disabled={playing || atEnd}
          active={playing}
          onPress={onPlay}
        />
        <View style={styles.islandRule} />
        <TransportButton
          label="PAUSE"
          glyph="❚❚"
          disabled={!playing}
          onPress={onPause}
        />
        <View style={styles.islandRule} />
        <TransportButton
          label="RESET"
          glyph="↺"
          disabled={atStart && !playing}
          onPress={onReset}
        />
      </View>
      <Text style={styles.frameReadout}>
        FRAME {padFrame(frame)} / {padFrame(LAST_FRAME)}
      </Text>
    </View>
  );
}

function TransportButton({
  label,
  glyph,
  disabled,
  active,
  onPress,
}: {
  label: string;
  glyph: string;
  disabled: boolean;
  active?: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      disabled={disabled}
      onPress={onPress}
      hitSlop={8}
      android_ripple={{ color: 'rgba(110, 231, 249, 0.12)' }}
      style={({ pressed }) => [
        styles.transportBtn,
        pressed && !disabled ? styles.transportPressed : null,
      ]}
    >
      <Text
        style={[
          styles.transportGlyph,
          active ? styles.transportActive : null,
          disabled ? styles.transportDisabled : null,
        ]}
      >
        {glyph}
      </Text>
      <Text
        style={[
          styles.transportLabel,
          active ? styles.transportActive : null,
          disabled ? styles.transportDisabled : null,
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

// ---------------------------------------------------------------------------
// Root simulator
// ---------------------------------------------------------------------------

export default function App() {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({
    width: VIEW_W,
    height: 420,
  });
  const controllerRef = useRef<PlaybackController | null>(null);

  const onTimeSeconds = useCallback((seconds: number) => {
    const next = frameFromVideoTime(seconds);
    setCurrentFrame((prev) => (prev === next ? prev : next));
  }, []);

  const onPlayingChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
  }, []);

  const threats = useMemo(
    () => extractActiveThreats(INCIDENTS, currentFrame),
    [currentFrame],
  );
  const activeIncidents = useMemo(
    () => extractActiveIncidents(INCIDENTS, currentFrame),
    [currentFrame],
  );
  const liveScore = useMemo(() => safetyScoreAt(currentFrame), [currentFrame]);
  const alert = useMemo(() => primaryAlert(threats), [threats]);

  const onPlay = useCallback(() => {
    if (currentFrame >= LAST_FRAME) {
      return;
    }
    controllerRef.current?.play();
  }, [currentFrame]);

  const onPause = useCallback(() => {
    controllerRef.current?.pause();
  }, []);

  const onReset = useCallback(() => {
    controllerRef.current?.reset();
    setCurrentFrame(0);
    setIsPlaying(false);
  }, []);

  const onCanvasLayout = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    if (width <= 0 || height <= 0) {
      return;
    }
    setCanvasSize({ width, height });
  }, []);

  const alertColor =
    alert.tone === 'critical'
      ? COLOR.red
      : alert.tone === 'caution'
        ? COLOR.orange
        : COLOR.cyan;

  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <HeaderPanel score={liveScore} frame={currentFrame} playing={isPlaying} />

      <View style={styles.stage}>
        <View style={styles.videoColumn}>
          <View style={styles.videoFrame}>
            <RideFeed
              onTimeSeconds={onTimeSeconds}
              onPlayingChange={onPlayingChange}
              controllerRef={controllerRef}
            />
            <Text style={styles.videoBadge}>DETECTED FEED</Text>
          </View>
        </View>

        <View style={styles.columnRule} />

        <View style={styles.radarColumn}>
          <View style={styles.canvasWrap} onLayout={onCanvasLayout}>
            <CockpitCanvas
              threats={threats}
              incidents={activeIncidents}
              frame={currentFrame}
              size={canvasSize}
            />
          </View>
          <View style={styles.alertBanner}>
            <View style={[styles.alertMark, { backgroundColor: alertColor }]} />
            <Text style={[styles.alertTitle, { color: alertColor }]}>
              {alert.title}
            </Text>
          </View>
        </View>
      </View>

      <ControlIsland
        playing={isPlaying}
        atEnd={currentFrame >= LAST_FRAME}
        atStart={currentFrame === 0}
        frame={currentFrame}
        onPlay={onPlay}
        onPause={onPause}
        onReset={onReset}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLOR.bg,
    paddingTop: Platform.select({ ios: 54, android: 28, default: 20 }),
    paddingBottom: Platform.select({ ios: 18, android: 12, default: 14 }),
  },
  header: {
    paddingHorizontal: 28,
    paddingBottom: 10,
  },
  headerTop: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  brand: {
    color: COLOR.text,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 4,
  },
  brandSub: {
    color: COLOR.muted,
    fontSize: 11,
    letterSpacing: 2.2,
    flex: 1,
  },
  livePill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  liveText: {
    color: COLOR.muted,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 2,
  },
  scoreMeter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginRight: 6,
  },
  scoreMeterLabel: {
    color: COLOR.muted,
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 1.8,
  },
  scoreTrack: {
    width: 88,
    height: 5,
    borderRadius: 3,
    backgroundColor: COLOR.cyanDeep,
    overflow: 'hidden',
  },
  scoreFill: {
    height: 5,
    borderRadius: 3,
  },
  scoreMeterValue: {
    width: 22,
    color: COLOR.text,
    fontSize: 11,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.4,
    textAlign: 'right',
  },
  progressBlock: {
    marginTop: 14,
  },
  progressMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  progressTitle: {
    color: COLOR.muted,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 2.2,
  },
  progressClock: {
    color: COLOR.text,
    fontSize: 11,
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.6,
  },
  progressTrack: {
    height: 2,
    backgroundColor: COLOR.cyanDeep,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: 2,
    backgroundColor: COLOR.cyan,
    borderRadius: 2,
  },
  stage: {
    flex: 1,
    flexDirection: 'row',
    minHeight: 0,
    paddingHorizontal: 16,
    paddingTop: 8,
    gap: 4,
  },
  videoColumn: {
    flex: 1.12,
    minWidth: 0,
    paddingRight: 8,
  },
  videoFrame: {
    flex: 1,
    position: 'relative',
    backgroundColor: COLOR.bg,
    borderColor: COLOR.hairline,
    borderWidth: 1,
    borderRadius: 14,
    overflow: 'hidden',
  },
  videoSurface: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },
  videoBadge: {
    position: 'absolute',
    top: 14,
    left: 16,
    color: COLOR.muted,
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 2.4,
    pointerEvents: 'none',
  },
  columnRule: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: COLOR.hairline,
    marginVertical: 12,
  },
  radarColumn: {
    flex: 1,
    minWidth: 0,
    paddingLeft: 8,
  },
  canvasWrap: {
    flex: 1,
    minHeight: 220,
  },
  alertBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  alertMark: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  alertTitle: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 2.4,
    textAlign: 'center',
  },
  controls: {
    paddingHorizontal: 20,
    paddingTop: 4,
  },
  island: {
    flexDirection: 'row',
    backgroundColor: COLOR.island,
    borderColor: COLOR.hairline,
    borderWidth: 1,
    borderRadius: 18,
    overflow: 'hidden',
  },
  islandRule: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: COLOR.hairline,
  },
  transportBtn: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 14,
    gap: 4,
  },
  transportPressed: {
    backgroundColor: '#11151A',
  },
  transportGlyph: {
    color: COLOR.cyan,
    fontSize: 13,
  },
  transportLabel: {
    color: COLOR.text,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 2,
  },
  transportActive: {
    color: COLOR.cyan,
  },
  transportDisabled: {
    color: '#3A4046',
  },
  frameReadout: {
    color: COLOR.muted,
    fontSize: 10,
    letterSpacing: 2,
    textAlign: 'center',
    marginTop: 10,
    fontVariant: ['tabular-nums'],
  },
});
