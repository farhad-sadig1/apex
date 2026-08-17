/** Detection video on the left, HUD on the right, both driven by video time. */
import { useEventListener } from 'expo';
import { Asset } from 'expo-asset';
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
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import Svg, {
  Circle,
  Defs,
  Ellipse,
  FeComposite,
  FeFlood,
  FeGaussianBlur,
  FeMerge,
  FeMergeNode,
  Filter,
  G,
  LinearGradient,
  Path,
  RadialGradient,
  Rect,
  Stop,
  Text as SvgText,
} from 'react-native-svg';

import telemetryJson from './assets/telemetry/ride1_telemetry.json';
import { resolveAssetUri } from './assetUri';

const CYCLIST_ICON = require('./src/assets/cyclist-icon.png');

// --- types ---

type IncidentType =
  | 'dynamic_side_closure'
  | 'sustained_side_proximity'
  | 'pedestrian_ahead'
  | 'pedestrian_critical';

type FlankSide = 'left' | 'right';
type CorridorLane = 'left' | 'center' | 'right';

interface TelemetryIncident {
  track_id: number;
  incident_type: IncidentType | string;
  start_frame: number;
  end_frame: number;
  max_severity_score: number;
  min_normalized_gap?: number;
  max_proximity_index?: number;
  /** 0 left rail → 1 right rail, at peak proximity */
  lateral_position?: number;
  /** vehicle overtake side; often missing */
  side?: FlankSide | string;
}

type FrameStatus = 'clear' | 'awareness' | 'risk' | 'critical';
type IncidentTier = 'awareness' | 'risk' | 'critical';

interface RiskCriticalEvent {
  tier: IncidentTier | string;
  incident_type: IncidentType | string;
  start_frame: number;
  timestamp_seconds: number;
}

interface RideSummary {
  tier_counts: {
    awareness: number;
    risk: number;
    critical: number;
  };
  risk_critical_events: RiskCriticalEvent[];
  risk_critical_per_minute: number;
  total_incidents_detected: number;
  ride_duration_seconds: number;
  fps?: number;
}

interface TelemetryLog {
  summary: RideSummary;
  incidents: TelemetryIncident[];
  frame_status: FrameStatus[];
}

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

interface CanvasSize {
  width: number;
  height: number;
}

interface PlaybackController {
  play: () => void;
  pause: () => void;
  reset: () => void;
  seek: (seconds: number) => void;
}

// --- colors ---

const COLOR = {
  bg: '#000000',
  radar: '#0c0f12',
  radarAlt: '#111215',
  road: '#14181e',
  horizonMist: '#2C303B',
  text: '#F4F6F8',
  muted: '#8B9198',
  hairline: '#1C1F24',
  cyan: '#6EE7F9',
  rail: '#3ac1e0',
  railGlow: '#52b2cf',
  cyanDeep: '#143A42',
  greyBand: '#2A3036',
  orange: '#FF8A3D',
  orangeRed: '#FF5A33',
  red: '#FF2A2A',
  island: '#0B0D10',
  vehicle: '#3A4046',
  atmosphere: '#1A2228',
} as const;

/** from public/, not a metro require */
const RIDE_VIDEO_URI = '/ride1_detected.mp4';

/** pad short backend hits so they actually show on the HUD */
const MIN_HUD_FRAMES = 12;

const ALERT_FADE_MS = 250;

const WEB_TINT_STYLE =
  Platform.OS === 'web'
    ? {
        transition: `opacity ${ALERT_FADE_MS}ms ease, fill ${ALERT_FADE_MS}ms ease, stroke ${ALERT_FADE_MS}ms ease`,
      }
    : undefined;

/** keep in sync with incidents/lateral_safety.py */
const LATERAL_PROXIMITY_THRESHOLD = 0.25;

const VIEW_W = 400;
const VIEW_H = 700;
const BIKE_X = 200;
const BIKE_Y = 502;

const HORIZON_X = BIKE_X;
const HORIZON_Y = 64;
const PATH_NEAR_Y = BIKE_Y - 36;
const PATH_NEAR_HALF = 124;
const ZONE_PAD_NEAR = 46;
const ICON_VIEW = 168;
const GLOW_VIEW = 252;

const Z_NEAR = 1.8;
const Z_FAR = 36;
const PERSPECTIVE_LINES = 18;
const RUNG_OPACITY_NEAR = 1;
const RUNG_OPACITY_FAR = 0.15;

interface Point {
  x: number;
  y: number;
}

// --- telemetry ---

function loadTelemetry(raw: unknown): TelemetryLog {
  const data = raw as TelemetryLog;
  if (
    !data?.summary ||
    !Array.isArray(data.incidents) ||
    !Array.isArray(data.frame_status)
  ) {
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

const FRAME_COUNT = Math.max(
  1,
  TELEMETRY.frame_status.length,
  Math.round(SUMMARY.ride_duration_seconds * PLAYBACK_FPS),
  MAX_INCIDENT_FRAME + 1,
);

const LAST_FRAME = FRAME_COUNT - 1;

const STATUS_COPY: Record<FrameStatus, { label: string; color: string }> = {
  clear: { label: 'CORRIDOR CLEAR', color: COLOR.cyan },
  awareness: { label: 'AWARENESS', color: COLOR.orange },
  risk: { label: 'RISK', color: COLOR.orangeRed },
  critical: { label: 'CRITICAL', color: COLOR.red },
};

function frameFromVideoTime(seconds: number): number {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return 0;
  }
  return Math.min(LAST_FRAME, Math.floor(seconds * PLAYBACK_FPS));
}

// --- threats ---

function isVisuallyActive(incident: TelemetryIncident, frame: number): boolean {
  const visualEnd = Math.max(
    incident.end_frame,
    incident.start_frame + MIN_HUD_FRAMES - 1,
  );
  return frame >= incident.start_frame && frame <= visualEnd;
}

/** missing side → light both flanks */
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

function formatSeconds(seconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = totalSeconds % 60;
  return `${minutes}:${remainder.toString().padStart(2, '0')}`;
}

function formatClock(frame: number): string {
  return formatSeconds(frame / PLAYBACK_FPS);
}

function formatIncidentType(type: string): string {
  return type.replace(/_/g, ' ').toUpperCase();
}

function formatIncidentTypeSubtitle(type: string): string {
  return type.replace(/_/g, ' ').toLowerCase();
}

const INCIDENT_STATUS_BY_TYPE: Record<string, FrameStatus> = {
  pedestrian_ahead: 'awareness',
  sustained_side_proximity: 'risk',
  pedestrian_critical: 'critical',
  dynamic_side_closure: 'critical',
};

function incidentStatus(incident: TelemetryIncident): FrameStatus {
  return INCIDENT_STATUS_BY_TYPE[incident.incident_type] ?? 'awareness';
}

function activeIncidentTypeLabel(
  incidents: readonly TelemetryIncident[],
  frame: number,
  status: FrameStatus,
): string {
  if (status === 'clear') {
    return '';
  }

  const active = incidents.filter(
    (incident) =>
      frame >= incident.start_frame && frame <= incident.end_frame,
  );
  if (active.length === 0) {
    return '';
  }

  const ranked = active.filter(
    (incident) => incidentStatus(incident) === status,
  );
  const pool = ranked.length > 0 ? ranked : active;
  const chosen = pool.reduce((best, incident) =>
    incident.max_severity_score > best.max_severity_score ? incident : best,
  );
  return formatIncidentTypeSubtitle(String(chosen.incident_type));
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
  return { color: COLOR.rail, opacity: pulse(frame) };
}

function alertTint(status: FrameStatus): string {
  if (status === 'critical') {
    return COLOR.red;
  }
  if (status === 'risk') {
    return COLOR.orange;
  }
  return COLOR.rail;
}

function hexToRgb(hex: string): [number, number, number] {
  const raw = hex.replace('#', '');
  return [
    parseInt(raw.slice(0, 2), 16),
    parseInt(raw.slice(2, 4), 16),
    parseInt(raw.slice(4, 6), 16),
  ];
}

function lerpHex(from: string, to: string, t: number): string {
  const a = hexToRgb(from);
  const b = hexToRgb(to);
  const mix = (index: number) => Math.round(lerp(a[index], b[index], t));
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}

function easeCss(t: number): number {
  // smoothstep ≈ css ease
  return t * t * (3 - 2 * t);
}

/** 250ms fade. fromZero = start at 0 on mount. */
function useAlertFade(active: boolean, fromZero = false): number {
  const [progress, setProgress] = useState(() =>
    fromZero ? 0 : active ? 1 : 0,
  );
  const progressRef = useRef(progress);
  progressRef.current = progress;

  useEffect(() => {
    const from = progressRef.current;
    const to = active ? 1 : 0;
    if (from === to) {
      return undefined;
    }
    const started = Date.now();
    let raf = 0;
    const tick = () => {
      const linear = clamp01((Date.now() - started) / ALERT_FADE_MS);
      const next = from + (to - from) * easeCss(linear);
      progressRef.current = next;
      setProgress(next);
      if (linear < 1) {
        raf = requestAnimationFrame(tick);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  return progress;
}

// --- geometry ---

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function lerp(from: number, to: number, t: number): number {
  return from + (to - from) * t;
}

function worldXForLane(lane: CorridorLane, extra = 0): number {
  if (lane === 'left') {
    return -(PATH_NEAR_HALF + extra);
  }
  if (lane === 'right') {
    return PATH_NEAR_HALF + extra;
  }
  return 0;
}

function depthScale(z: number): number {
  return Z_NEAR / z;
}

function projectWorld(worldX: number, z: number): Point {
  const scale = depthScale(z);
  return {
    x: HORIZON_X + worldX * scale,
    y: HORIZON_Y + (PATH_NEAR_Y - HORIZON_Y) * scale,
  };
}

function zAtDepthT(t: number): number {
  const scale = lerp(1, Z_NEAR / Z_FAR, clamp01(t));
  return Z_NEAR / scale;
}

function projectLane(lane: CorridorLane, t: number, extra = 0): Point {
  return projectWorld(worldXForLane(lane, extra), zAtDepthT(t));
}

function projectedLanePath(side: FlankSide): string {
  const near = projectLane(side, 0);
  const far = projectLane(side, 1);
  return `M ${near.x} ${near.y} L ${far.x} ${far.y}`;
}

function roadSurfacePath(): string {
  const leftNear = projectLane('left', 0);
  const leftFar = projectLane('left', 1);
  const rightFar = projectLane('right', 1);
  const rightNear = projectLane('right', 0);
  return [
    `M ${leftNear.x} ${leftNear.y}`,
    `L ${leftFar.x} ${leftFar.y}`,
    `L ${rightFar.x} ${rightFar.y}`,
    `L ${rightNear.x} ${rightNear.y}`,
    'Z',
  ].join(' ');
}

function laneSheetPath(side: FlankSide): string {
  const innerNear = projectLane('center', 0);
  const innerFar = projectLane('center', 1);
  const outerNear = projectLane(side, 0);
  const outerFar = projectLane(side, 1);
  return [
    `M ${innerNear.x} ${innerNear.y}`,
    `L ${innerFar.x} ${innerFar.y}`,
    `L ${outerFar.x} ${outerFar.y}`,
    `L ${outerNear.x} ${outerNear.y}`,
    'Z',
  ].join(' ');
}

function flankZonePath(side: FlankSide): string {
  const innerNear = projectLane(side, 0);
  const innerFar = projectLane(side, 1);
  const outerNear = projectLane(side, 0, ZONE_PAD_NEAR);
  const outerFar = projectLane(side, 1, ZONE_PAD_NEAR);
  return [
    `M ${innerNear.x} ${innerNear.y}`,
    `L ${innerFar.x} ${innerFar.y}`,
    `L ${outerFar.x} ${outerFar.y}`,
    `L ${outerNear.x} ${outerNear.y}`,
    'Z',
  ].join(' ');
}

function RadarDefs() {
  return (
    <Defs>
      <RadialGradient
        id="horizonGlow"
        cx={HORIZON_X}
        cy={HORIZON_Y}
        fx={HORIZON_X}
        fy={HORIZON_Y}
        rx={220}
        ry={168}
        gradientUnits="userSpaceOnUse"
      >
        <Stop offset="0" stopColor={COLOR.horizonMist} stopOpacity={0.4} />
        <Stop offset="0.42" stopColor={COLOR.horizonMist} stopOpacity={0.16} />
        <Stop offset="1" stopColor={COLOR.horizonMist} stopOpacity={0} />
      </RadialGradient>
      <Filter
        id="laneNeon"
        x="-100%"
        y="-100%"
        width="300%"
        height="300%"
        filterUnits="objectBoundingBox"
        primitiveUnits="userSpaceOnUse"
      >
        <FeGaussianBlur in="SourceGraphic" stdDeviation="3" result="blurTight" />
        <FeFlood
          floodColor={COLOR.railGlow}
          floodOpacity={0.8}
          result="tintTight"
        />
        <FeComposite
          in="tintTight"
          in2="blurTight"
          operator="in"
          result="glowTight"
        />
        <FeGaussianBlur in="SourceGraphic" stdDeviation="9" result="blurWide" />
        <FeFlood
          floodColor={COLOR.railGlow}
          floodOpacity={0.3}
          result="tintWide"
        />
        <FeComposite
          in="tintWide"
          in2="blurWide"
          operator="in"
          result="glowWide"
        />
        <FeMerge>
          <FeMergeNode in="glowWide" />
          <FeMergeNode in="glowTight" />
          <FeMergeNode in="SourceGraphic" />
        </FeMerge>
      </Filter>
    </Defs>
  );
}

function RadarStage() {
  return (
    <G>
      <Rect
        x={0}
        y={0}
        width={VIEW_W}
        height={VIEW_H}
        fill={COLOR.radar}
      />
      <Rect
        x={0}
        y={0}
        width={VIEW_W}
        height={PATH_NEAR_Y + 48}
        fill="url(#horizonGlow)"
      />
      <Path d={roadSurfacePath()} fill={COLOR.road} opacity={0.92} />
    </G>
  );
}

function PerspectiveGround() {
  const lines = [];
  for (let index = 0; index < PERSPECTIVE_LINES; index += 1) {
    const u = index / (PERSPECTIVE_LINES - 1);
    const z = lerp(Z_NEAR, Z_FAR, u);
    const t = (1 - Z_NEAR / z) / (1 - Z_NEAR / Z_FAR);
    const left = projectWorld(-PATH_NEAR_HALF, z);
    const right = projectWorld(PATH_NEAR_HALF, z);
    const opacity = lerp(RUNG_OPACITY_NEAR, RUNG_OPACITY_FAR, t);
    lines.push(
      <Path
        key={`ground-${index}`}
        d={`M ${left.x} ${left.y} L ${right.x} ${right.y}`}
        stroke={COLOR.rail}
        strokeWidth={lerp(1.35, 0.4, t)}
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
  const alerting = level !== 'safe';
  const fade = useAlertFade(alerting);
  const live = flankPaint(level, frame);
  const gradientId = `flank-zone-${side}`;
  const labelX = side === 'left' ? 28 : 372;
  const labelAnchor = side === 'left' ? 'start' : 'end';
  const label = side === 'left' ? 'LEFT' : 'RIGHT';
  const labelColor =
    level === 'close' ? COLOR.red : level === 'caution' ? COLOR.orange : COLOR.muted;
  const innerX = side === 'left' ? '100%' : '0%';
  const outerX = side === 'left' ? '0%' : '100%';
  const glowOpacity = lerp(0.08, Math.min(0.55, live.opacity * 0.5), fade);

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
          <Stop offset="0" stopColor={live.color} stopOpacity={glowOpacity} />
          <Stop offset="1" stopColor={live.color} stopOpacity={0} />
        </LinearGradient>
      </Defs>
      <Path
        d={flankZonePath(side)}
        fill={`url(#${gradientId})`}
        style={WEB_TINT_STYLE}
      />
      <Path
        d={flankZonePath(side)}
        fill={live.color}
        opacity={lerp(0.03, Math.min(0.22, live.opacity * 0.22), fade)}
        style={WEB_TINT_STYLE}
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

function LaneSheet({
  side,
  color,
  active,
  close,
  frame,
}: {
  side: FlankSide;
  color: string;
  active: boolean;
  close: boolean;
  frame: number;
}) {
  const fade = useAlertFade(active);
  const flash = close && flashOn(frame, 2) ? 1 : 0.42;
  const opacity = fade * (close ? 0.34 * flash : 0.22);

  return (
    <Path
      d={laneSheetPath(side)}
      fill={color}
      opacity={opacity}
      style={WEB_TINT_STYLE}
    />
  );
}

function laneSheetState(
  flank: FlankLevel,
  incidents: readonly TelemetryIncident[],
  side: FlankSide,
): { color: string; active: boolean; close: boolean } {
  if (flank === 'close') {
    return { color: COLOR.red, active: true, close: true };
  }
  if (flank === 'caution') {
    return { color: COLOR.orange, active: true, close: false };
  }
  for (const incident of incidents) {
    const kind = incident.incident_type;
    if (kind !== 'pedestrian_ahead' && kind !== 'pedestrian_critical') {
      continue;
    }
    const lateral = clamp01(incident.lateral_position ?? 0.5);
    if (side === 'left' && lateral >= 0.5) {
      continue;
    }
    if (side === 'right' && lateral <= 0.5) {
      continue;
    }
    if (kind === 'pedestrian_critical') {
      return { color: COLOR.red, active: true, close: true };
    }
    return { color: COLOR.rail, active: true, close: false };
  }
  return { color: COLOR.rail, active: false, close: false };
}

function PathVector({
  level,
  leftLevel,
  rightLevel,
  frame,
}: {
  level: CorridorLevel;
  leftLevel: FlankLevel;
  rightLevel: FlankLevel;
  frame: number;
}) {
  const left = projectedLanePath('left');
  const right = projectedLanePath('right');

  return (
    <G>
      <CorridorRail d={left} level={leftLevel} frame={frame} />
      <CorridorRail d={right} level={rightLevel} frame={frame} />
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

function CorridorRail({
  d,
  level,
  frame,
}: {
  d: string;
  level: FlankLevel;
  frame: number;
}) {
  const alerting = level !== 'safe';
  const fade = useAlertFade(alerting);
  const alertColor =
    level === 'close' ? COLOR.red : level === 'caution' ? COLOR.orange : COLOR.rail;
  const color = lerpHex(COLOR.rail, alertColor, fade);
  const restOpacity = 0.88;
  const flashPeriod = level === 'close' ? 2 : 4;
  const alertOpacity = flashOn(frame, flashPeriod) ? 0.95 : 0.22;
  const opacity = lerp(restOpacity, alertOpacity, fade);
  const strokeWidth = lerp(2.4, 3.1, fade);

  return (
    <G>
      <Path
        d={d}
        stroke={color}
        strokeWidth={16}
        strokeLinecap="round"
        fill="none"
        opacity={opacity * 0.18}
        style={WEB_TINT_STYLE}
      />
      <Path
        d={d}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        fill="none"
        opacity={opacity}
        filter="url(#laneNeon)"
        style={WEB_TINT_STYLE}
      />
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
  const rail = projectLane(side, t);
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

function PedestrianGlyph({
  color,
  fillOpacity,
  strokeOpacity,
  strokeWidth,
}: {
  color: string;
  fillOpacity: number;
  strokeOpacity: number;
  strokeWidth: number;
}) {
  const paint = {
    fill: color,
    fillOpacity,
    stroke: color,
    strokeOpacity,
    strokeWidth,
  };
  return (
    <G>
      <Circle cx={0} cy={-21.5} r={4.15} {...paint} />
      <Ellipse cx={0} cy={-13.2} rx={7.1} ry={4.35} {...paint} />
      <Ellipse cx={0} cy={-5.4} rx={4.35} ry={6.4} {...paint} />
      <Ellipse cx={-2.85} cy={-0.2} rx={1.7} ry={2.25} {...paint} />
      <Ellipse cx={2.85} cy={-0.2} rx={1.7} ry={2.25} {...paint} />
    </G>
  );
}

function PedestrianMark({
  incident,
  frame,
}: {
  incident: TelemetryIncident;
  frame: number;
}) {
  const fade = useAlertFade(true, true);
  const proximity = clamp01(incident.max_proximity_index ?? 0.5);
  const t = 1 - proximity;
  const leftRail = projectLane('left', t);
  const rightRail = projectLane('right', t);
  const lateral = clamp01(incident.lateral_position ?? 0.5);
  const x = lerp(leftRail.x, rightRail.x, lateral);
  const y = lerp(leftRail.y, rightRail.y, lateral);
  const scale = lerp(1.18, 0.22, t);
  const status = incidentStatus(incident);
  const color = lerpHex(COLOR.muted, alertTint(status), fade);
  const flashPeriod = status === 'critical' ? 2 : 4;
  const flash = flashOn(frame, flashPeriod) ? 1 : 0.22;
  const strokeOpacity = lerp(0.2, flash, fade);
  const fillOpacity = lerp(0.04, 0.2 * flash, fade);

  return (
    <G x={x} y={y}>
      <G scale={scale}>
        <PedestrianGlyph
          color={color}
          fillOpacity={fillOpacity}
          strokeOpacity={strokeOpacity}
          strokeWidth={1.15}
        />
      </G>
    </G>
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
  const leftSheet = laneSheetState(left, incidents, 'left');
  const rightSheet = laneSheetState(right, incidents, 'right');

  return (
    <Svg
      width={size.width}
      height={size.height}
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ backgroundColor: COLOR.radar }}
    >
      <RadarDefs />
      <RadarStage />
      <PerspectiveGround />
      <LaneSheet
        side="left"
        color={leftSheet.color}
        active={leftSheet.active}
        close={leftSheet.close}
        frame={frame}
      />
      <LaneSheet
        side="right"
        color={rightSheet.color}
        active={rightSheet.active}
        close={rightSheet.close}
        frame={frame}
      />
      <FlankZone side="left" level={left} frame={frame} />
      <FlankZone side="right" level={right} frame={frame} />
      <PathVector
        level={corridor}
        leftLevel={left}
        rightLevel={right}
        frame={frame}
      />
      <IncidentActors incidents={incidents} frame={frame} />
    </Svg>
  );
}

function meetBox(size: CanvasSize): { scale: number; ox: number; oy: number } {
  const scale = Math.min(size.width / VIEW_W, size.height / VIEW_H);
  return {
    scale,
    ox: (size.width - VIEW_W * scale) / 2,
    oy: (size.height - VIEW_H * scale) / 2,
  };
}

function glowTint(status: FrameStatus): string {
  if (status === 'critical') {
    return COLOR.red;
  }
  if (status === 'risk') {
    return COLOR.orange;
  }
  return COLOR.rail;
}

function CyclistMarker({
  size,
  status,
}: {
  size: CanvasSize;
  status: FrameStatus;
}) {
  const fade = useAlertFade(status !== 'clear');
  const colorRef = useRef(COLOR.rail);
  if (status !== 'clear') {
    colorRef.current = glowTint(status);
  }
  const color = colorRef.current;
  const { scale, ox, oy } = meetBox(size);
  const icon = ICON_VIEW * scale;
  const glow = GLOW_VIEW * scale;
  const iconUri = resolveAssetUri(CYCLIST_ICON, Platform.OS, {
    fromModule: (moduleId) => Asset.fromModule(moduleId),
    resolveAssetSource: Image.resolveAssetSource,
  });

  return (
    <View
      pointerEvents="none"
      style={[
        styles.cyclistMarker,
        {
          left: ox + BIKE_X * scale - glow / 2,
          top: oy + BIKE_Y * scale - glow / 2,
          width: glow,
          height: glow,
        },
      ]}
    >
      <View
        style={[
          styles.cyclistGlow,
          {
            width: glow,
            height: glow,
            borderRadius: glow / 2,
            opacity: fade * 0.58,
            ...(Platform.OS === 'web'
              ? {
                  backgroundColor: 'transparent',
                  // @ts-expect-error web css
                  backgroundImage: `radial-gradient(circle, ${color} 0%, ${color} 30%, transparent 72%)`,
                  filter: 'blur(18px)',
                  transition: `opacity ${ALERT_FADE_MS}ms ease`,
                }
              : {
                  backgroundColor: color,
                  shadowColor: color,
                  shadowOffset: { width: 0, height: 0 },
                  shadowOpacity: 0.9,
                  shadowRadius: 28,
                }),
          },
        ]}
      />
      {Platform.OS === 'web'
        ? createElement('img', {
            src: iconUri,
            alt: '',
            draggable: false,
            style: {
              width: icon,
              height: icon,
              objectFit: 'contain',
              position: 'relative',
              zIndex: 1,
              pointerEvents: 'none',
              display: 'block',
            },
          })
        : (
          <Image
            source={CYCLIST_ICON}
            resizeMode="contain"
            style={{ width: icon, height: icon, zIndex: 1 }}
          />
        )}
    </View>
  );
}

// --- video ---

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
      seek: (seconds: number) => {
        node.currentTime = seconds;
        onTimeRef.current(node.currentTime);
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
      seek: (seconds: number) => {
        player.currentTime = seconds;
        onTimeRef.current(seconds);
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

// --- chrome ---

function HeaderPanel({
  frame,
  playing,
}: {
  frame: number;
  playing: boolean;
}) {
  const progress = LAST_FRAME <= 0 ? 1 : frame / LAST_FRAME;

  return (
    <View style={styles.header}>
      <View style={styles.headerTop}>
        <Text style={styles.brand}>ARAS</Text>
        <Text style={styles.brandSub}>URBAN CYCLIST</Text>
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
  reportOpen,
  onPlay,
  onPause,
  onReset,
  onToggleReport,
}: {
  playing: boolean;
  atEnd: boolean;
  atStart: boolean;
  frame: number;
  reportOpen: boolean;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onToggleReport: () => void;
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
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={reportOpen ? 'Back to ride' : 'View Ride Report'}
        onPress={onToggleReport}
        hitSlop={8}
        style={({ pressed }) => [
          styles.reportToggle,
          pressed ? styles.reportTogglePressed : null,
        ]}
      >
        <Text
          style={[
            styles.reportToggleLabel,
            reportOpen ? styles.transportActive : null,
          ]}
        >
          {reportOpen ? 'BACK TO RIDE' : 'VIEW RIDE REPORT'}
        </Text>
      </Pressable>
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

function RideReport({
  onSeekEvent,
}: {
  onSeekEvent: (seconds: number) => void;
}) {
  const counts = SUMMARY.tier_counts ?? {
    awareness: 0,
    risk: 0,
    critical: 0,
  };
  const events = SUMMARY.risk_critical_events ?? [];
  const rate = SUMMARY.risk_critical_per_minute ?? 0;

  return (
    <View style={styles.report}>
      <Text style={styles.reportKicker}>RIDE REPORT</Text>

      <View style={styles.reportTiers}>
        <TierStat
          label="AWARENESS"
          value={counts.awareness}
          color={COLOR.orange}
        />
        <TierStat
          label="RISK"
          value={counts.risk}
          color={COLOR.orangeRed}
        />
        <TierStat
          label="CRITICAL"
          value={counts.critical}
          color={COLOR.red}
        />
      </View>

      <View style={styles.reportRate}>
        <Text style={styles.progressTitle}>HAZARD EFFICIENCY RATE</Text>
        <View style={styles.reportRateScore}>
          <Text style={styles.reportRateValue}>{rate.toFixed(2)}</Text>
          <Text style={styles.reportRateNote}>
            (risk + critical events detected per minute of total ride time)
          </Text>
        </View>
      </View>

      <Text style={[styles.progressTitle, styles.reportListLabel]}>
        RISK & CRITICAL EVENTS
      </Text>
      <ScrollView
        style={styles.reportList}
        contentContainerStyle={styles.reportListContent}
      >
        {events.length === 0 ? (
          <Text style={styles.reportEmpty}>NO RISK OR CRITICAL EVENTS</Text>
        ) : (
          events.map((event, index) => {
            const tone =
              event.tier === 'critical' ? COLOR.red : COLOR.orangeRed;
            return (
              <Pressable
                key={`${event.start_frame}-${event.incident_type}-${index}`}
                accessibilityRole="button"
                accessibilityLabel={`Jump to ${event.incident_type} at ${formatSeconds(event.timestamp_seconds)}`}
                onPress={() => onSeekEvent(event.timestamp_seconds)}
                style={({ pressed }) => [
                  styles.reportRow,
                  pressed ? styles.reportRowPressed : null,
                ]}
              >
                <View style={[styles.alertMark, { backgroundColor: tone }]} />
                <Text style={[styles.reportTier, { color: tone }]}>
                  {String(event.tier).toUpperCase()}
                </Text>
                <Text style={styles.reportEventType} numberOfLines={1}>
                  {formatIncidentType(String(event.incident_type))}
                </Text>
                <Text style={styles.reportEventTime}>
                  {formatSeconds(event.timestamp_seconds)}
                </Text>
              </Pressable>
            );
          })
        )}
      </ScrollView>
    </View>
  );
}

function TierStat({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const empty = value === 0;

  return (
    <View style={[styles.tierStat, empty ? styles.tierStatEmpty : null]}>
      <Text
        style={[
          styles.tierStatValue,
          { color },
          empty ? styles.tierStatValueEmpty : null,
        ]}
      >
        {empty ? '0 (None Detected)' : value}
      </Text>
      <Text style={styles.tierStatLabel}>{label}</Text>
    </View>
  );
}

// --- app ---

export default function App() {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({
    width: VIEW_W,
    height: 420,
  });
  const controllerRef = useRef<PlaybackController | null>(null);
  const wasAtEndRef = useRef(false);

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
  const liveStatus = TELEMETRY.frame_status[currentFrame] ?? 'clear';
  const statusView = STATUS_COPY[liveStatus] ?? STATUS_COPY.clear;
  const statusSubtitle = useMemo(
    () => activeIncidentTypeLabel(INCIDENTS, currentFrame, liveStatus),
    [currentFrame, liveStatus],
  );

  const atEnd = currentFrame >= LAST_FRAME;

  useEffect(() => {
    if (atEnd && !wasAtEndRef.current) {
      setReportOpen(true);
    }
    wasAtEndRef.current = atEnd;
  }, [atEnd]);

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
    setReportOpen(false);
  }, []);

  const onToggleReport = useCallback(() => {
    if (!reportOpen) {
      controllerRef.current?.pause();
      setIsPlaying(false);
    }
    setReportOpen((open) => !open);
  }, [reportOpen]);

  const onSeekEvent = useCallback((seconds: number) => {
    controllerRef.current?.seek(seconds);
    controllerRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const onCanvasLayout = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    if (width <= 0 || height <= 0) {
      return;
    }
    setCanvasSize({ width, height });
  }, []);

  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <HeaderPanel frame={currentFrame} playing={isPlaying} />

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
          {reportOpen ? (
            <RideReport onSeekEvent={onSeekEvent} />
          ) : (
            <>
              <View style={styles.canvasWrap} onLayout={onCanvasLayout}>
                <CockpitCanvas
                  threats={threats}
                  incidents={activeIncidents}
                  frame={currentFrame}
                  size={canvasSize}
                />
                <CyclistMarker size={canvasSize} status={liveStatus} />
              </View>
              <View style={styles.alertBanner}>
                <View style={styles.alertHeadline}>
                  <View
                    style={[
                      styles.alertMark,
                      { backgroundColor: statusView.color },
                    ]}
                  />
                  <Text style={[styles.alertTitle, { color: statusView.color }]}>
                    {statusView.label}
                  </Text>
                </View>
                <View style={styles.alertSubtitleSlot}>
                  {statusSubtitle ? (
                    <Text style={styles.alertSubtitle}>{statusSubtitle}</Text>
                  ) : null}
                </View>
              </View>
            </>
          )}
        </View>
      </View>

      <ControlIsland
        playing={isPlaying}
        atEnd={atEnd}
        atStart={currentFrame === 0}
        frame={currentFrame}
        reportOpen={reportOpen}
        onPlay={onPlay}
        onPause={onPause}
        onReset={onReset}
        onToggleReport={onToggleReport}
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
    backgroundColor: COLOR.radar,
    borderRadius: 14,
    overflow: 'hidden',
  },
  cyclistMarker: {
    position: 'absolute',
    alignItems: 'center',
    justifyContent: 'center',
  },
  cyclistGlow: {
    position: 'absolute',
  },
  alertBanner: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  alertHeadline: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
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
  alertSubtitleSlot: {
    minHeight: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alertSubtitle: {
    color: COLOR.muted,
    fontSize: 10,
    fontWeight: '500',
    letterSpacing: 0.6,
    textAlign: 'center',
    textTransform: 'lowercase',
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
  reportToggle: {
    alignItems: 'center',
    paddingVertical: 10,
    marginTop: 8,
  },
  reportTogglePressed: {
    opacity: 0.7,
  },
  reportToggleLabel: {
    color: COLOR.muted,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 2.2,
  },
  report: {
    flex: 1,
    minHeight: 220,
    paddingHorizontal: 8,
    paddingTop: 8,
    paddingBottom: 4,
  },
  reportKicker: {
    color: COLOR.text,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 4,
    marginBottom: 16,
  },
  reportTiers: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  tierStat: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 6,
    borderWidth: 1,
    borderColor: COLOR.hairline,
    borderRadius: 14,
    backgroundColor: COLOR.island,
    gap: 4,
  },
  tierStatEmpty: {
    opacity: 0.42,
  },
  tierStatValue: {
    fontSize: 22,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.4,
    textAlign: 'center',
  },
  tierStatValueEmpty: {
    fontSize: 11,
    letterSpacing: 0.2,
  },
  tierStatLabel: {
    color: COLOR.muted,
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 1.8,
  },
  reportRate: {
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: COLOR.hairline,
    gap: 6,
  },
  reportRateScore: {
    gap: 4,
  },
  reportRateValue: {
    color: COLOR.text,
    fontSize: 18,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.6,
  },
  reportRateNote: {
    color: COLOR.muted,
    fontSize: 9,
    fontWeight: '500',
    letterSpacing: 0.3,
    lineHeight: 13,
  },
  reportListLabel: {
    marginBottom: 8,
  },
  reportList: {
    flex: 1,
    minHeight: 0,
  },
  reportListContent: {
    paddingBottom: 8,
    gap: 4,
  },
  reportEmpty: {
    color: COLOR.muted,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 2,
    paddingVertical: 16,
  },
  reportRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLOR.hairline,
    backgroundColor: COLOR.island,
  },
  reportRowPressed: {
    backgroundColor: '#11151A',
  },
  reportTier: {
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 1.6,
    width: 72,
  },
  reportEventType: {
    flex: 1,
    color: COLOR.text,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.4,
  },
  reportEventTime: {
    color: COLOR.muted,
    fontSize: 11,
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.6,
  },
  frameReadout: {
    color: COLOR.muted,
    fontSize: 10,
    letterSpacing: 2,
    textAlign: 'center',
    marginTop: 4,
    fontVariant: ['tabular-nums'],
  },
});
