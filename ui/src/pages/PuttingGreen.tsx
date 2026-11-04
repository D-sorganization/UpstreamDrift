/** Interactive 3D putting scene for epic #8345, phase P1. */
import { useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";

import type {
  GreenReadingResponse,
  PuttSimulationResponse,
  ScatterAnalysisResponse,
} from "@/api/generated/types";
import { simulatePutt3D } from "@/api/puttingClient";
import { WorkspaceShell } from "@/components/layout/WorkspaceShell";
import { DispersionPanel } from "@/components/putting/DispersionPanel";
import { PutterHeadControl } from "@/components/putting/PutterHeadControl";
import { PuttingScene3D } from "@/components/visualization/PuttingScene3D";
import {
  CAMERA_PRESETS,
  VIEW_LABELS,
  phaseJumps,
  type CameraPreset,
} from "@/utils/cameraPresets";
import { usePuttingStore } from "@/stores/usePuttingStore";
import {
  UNIT_SYSTEMS,
  formatQuantity,
  fromDisplay,
  toDisplay,
  type Quantity,
  type UnitSystem,
} from "@/utils/units";
import { IMPACT_LEAD_IN_S, sampleAtPlaybackTime } from "./puttingPlayback";

export type PuttResult = PuttSimulationResponse;
export type GreenReading = GreenReadingResponse;
export type ScatterResult = ScatterAnalysisResponse;

interface RangeControlProps {
  label: string;
  value: number;
  displayValue: string;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}

function RangeControl({
  label,
  value,
  displayValue,
  min,
  max,
  step,
  disabled = false,
  onChange,
}: RangeControlProps) {
  return (
    <label
      className="block text-xs"
      style={{ color: "var(--sidekick-color-text-muted)" }}
    >
      <span className="mb-1 flex justify-between gap-3">
        <span>{label}</span>
        <span
          className="font-mono"
          style={{ color: "var(--sidekick-color-text)" }}
        >
          {displayValue}
        </span>
      </span>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
        style={{ accentColor: "var(--sidekick-color-accent)" }}
      />
    </label>
  );
}

/** [min, max, step] of a slider, in display units for each system. */
type DisplayRange = Record<UnitSystem, readonly [number, number, number]>;

interface UnitRangeControlProps {
  label: string;
  valueSi: number;
  quantity: Quantity;
  range: DisplayRange;
  decimals: Record<UnitSystem, number>;
  disabled?: boolean;
  onChangeSi: (value: number) => void;
}

function UnitRangeControl({
  label,
  valueSi,
  quantity,
  range,
  decimals,
  disabled,
  onChangeSi,
}: UnitRangeControlProps) {
  const system = usePuttingStore((state) => state.unitSystem);
  const [min, max, step] = range[system];
  return (
    <RangeControl
      label={label}
      value={toDisplay(valueSi, quantity, system)}
      displayValue={formatQuantity(valueSi, quantity, system, decimals[system])}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(value) => onChangeSi(fromDisplay(value, quantity, system))}
    />
  );
}

function UnitSystemSelect() {
  const system = usePuttingStore((state) => state.unitSystem);
  const setUnitSystem = usePuttingStore((state) => state.setUnitSystem);
  return (
    <label className="mt-3 block text-xs">
      Units
      <select
        aria-label="Units"
        value={system}
        onChange={(event) => setUnitSystem(event.target.value as UnitSystem)}
        className="ml-2 rounded border px-2 py-1"
        style={{
          backgroundColor: "var(--sidekick-color-input)",
          borderColor: "var(--sidekick-color-border)",
        }}
      >
        {UNIT_SYSTEMS.map((option) => (
          <option key={option} value={option}>
            {option === "metric" ? "Metric (m, m/s)" : "Imperial (ft, mph)"}
          </option>
        ))}
      </select>
    </label>
  );
}

const SMALL_LENGTH_DECIMALS = { metric: 0, imperial: 2 } as const;

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className="space-y-3 border-b p-4"
      style={{ borderColor: "var(--sidekick-color-border)" }}
    >
      <h2
        className="text-xs font-semibold tracking-wide"
        style={{ color: "var(--sidekick-color-text-muted)" }}
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

function HoselPositionControl() {
  const parameters = usePuttingStore((state) => state.parameters);
  const updateParameters = usePuttingStore((state) => state.updateParameters);
  const toePercent = ((parameters.hosel_toe_m + 0.08) / 0.16) * 100;
  const forwardPercent = ((0.05 - parameters.hosel_forward_m) / 0.1) * 100;
  // A balance preset's hosel offsets come from the shared core (#14462).
  const presetHead = Boolean(parameters.balance_type);

  return (
    <div className="space-y-3">
      <div
        className="relative h-24 rounded border"
        style={{
          backgroundColor: "var(--sidekick-color-surface-muted)",
          borderColor: "var(--sidekick-color-border)",
        }}
        aria-label="Top view of putter head, center of gravity, and hosel position"
        role="img"
      >
        <span
          className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{ backgroundColor: "var(--sidekick-color-text-subtle)" }}
          title="Center of gravity"
        />
        <span
          className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2"
          style={{
            left: `${toePercent}%`,
            top: `${forwardPercent}%`,
            backgroundColor: "var(--sidekick-color-warning)",
            borderColor: "var(--sidekick-color-selection-text)",
          }}
          title="Shaft attachment"
        />
        <span className="absolute bottom-1 left-2 text-xs">Heel</span>
        <span className="absolute bottom-1 right-2 text-xs">Toe</span>
        <span className="absolute left-1 top-1 text-xs">Forward</span>
      </div>
      <UnitRangeControl
        label="Hosel Heel to Toe"
        valueSi={parameters.hosel_toe_m}
        quantity="smallLength"
        range={{ metric: [-80, 80, 1], imperial: [-3.15, 3.15, 0.05] }}
        decimals={SMALL_LENGTH_DECIMALS}
        disabled={presetHead}
        onChangeSi={(hosel_toe_m) => updateParameters({ hosel_toe_m })}
      />
      <UnitRangeControl
        label="Hosel Back to Forward"
        valueSi={parameters.hosel_forward_m}
        quantity="smallLength"
        range={{ metric: [-50, 50, 1], imperial: [-2, 2, 0.05] }}
        decimals={SMALL_LENGTH_DECIMALS}
        disabled={presetHead}
        onChangeSi={(hosel_forward_m) => updateParameters({ hosel_forward_m })}
      />
    </div>
  );
}

function PlaybackControls() {
  const result = usePuttingStore((state) => state.result);
  const playbackTimeS = usePuttingStore((state) => state.playbackTimeS);
  const playbackRate = usePuttingStore((state) => state.playbackRate);
  const playing = usePuttingStore((state) => state.playing);
  const setPlaybackTime = usePuttingStore((state) => state.setPlaybackTime);
  const setPlaybackRate = usePuttingStore((state) => state.setPlaybackRate);
  const setPlaying = usePuttingStore((state) => state.setPlaying);
  const cameraView = usePuttingStore((state) => state.cameraView);
  const setCameraView = usePuttingStore((state) => state.setCameraView);
  if (!result) return null;

  const durationS = result.duration_s + IMPACT_LEAD_IN_S;
  return (
    <div
      className="flex flex-wrap items-center gap-3 border-t px-4 py-3"
      style={{
        backgroundColor: "var(--sidekick-color-surface)",
        borderColor: "var(--sidekick-color-border)",
      }}
    >
      <button
        type="button"
        onClick={() => setPlaying(!playing)}
        className="rounded px-3 py-1.5 text-sm font-medium"
        style={{
          backgroundColor: "var(--sidekick-color-accent)",
          color: "var(--sidekick-color-selection-text)",
        }}
        aria-label={playing ? "Pause putt playback" : "Play putt playback"}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        onClick={() => {
          setPlaybackTime(0);
          setPlaying(false);
        }}
        className="rounded border px-3 py-1.5 text-sm"
        style={{ borderColor: "var(--sidekick-color-border)" }}
      >
        Reset
      </button>
      <input
        aria-label="Putt playback position"
        type="range"
        min={0}
        max={durationS}
        step={0.002}
        value={playbackTimeS}
        onChange={(event) => {
          setPlaybackTime(Number(event.target.value));
          setPlaying(false);
        }}
        className="min-w-40 flex-1"
        style={{ accentColor: "var(--sidekick-color-accent)" }}
      />
      <span className="w-24 text-right font-mono text-xs">
        {playbackTimeS.toFixed(2)} / {durationS.toFixed(2)} s
      </span>
      <label className="text-xs">
        Playback Rate
        <select
          aria-label="Playback rate"
          value={playbackRate}
          onChange={(event) => setPlaybackRate(Number(event.target.value))}
          className="ml-2 rounded border px-2 py-1"
          style={{
            backgroundColor: "var(--sidekick-color-input)",
            borderColor: "var(--sidekick-color-border)",
          }}
        >
          {[0.1, 0.25, 0.5, 1, 2].map((rate) => (
            <option key={rate} value={rate}>
              {rate}x
            </option>
          ))}
        </select>
      </label>
      <label className="text-xs">
        Camera View
        <select
          aria-label="Camera view"
          value={cameraView}
          onChange={(event) =>
            setCameraView(event.target.value as CameraPreset)
          }
          className="ml-2 rounded border px-2 py-1"
          style={{
            backgroundColor: "var(--sidekick-color-input)",
            borderColor: "var(--sidekick-color-border)",
          }}
        >
          {CAMERA_PRESETS.map((preset) => (
            <option key={preset} value={preset}>
              {VIEW_LABELS[preset]}
            </option>
          ))}
        </select>
      </label>
      <div className="flex flex-wrap gap-2" aria-label="Jump to phase">
        {phaseJumps(result.phases ?? []).map((jump) => (
          <button
            key={jump.label}
            type="button"
            onClick={() => {
              setPlaybackTime(jump.playbackTimeS);
              setPlaying(false);
            }}
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: "var(--sidekick-color-border)" }}
          >
            {jump.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function PuttingGreenPage() {
  const parameters = usePuttingStore((state) => state.parameters);
  const result = usePuttingStore((state) => state.result);
  const playbackTimeS = usePuttingStore((state) => state.playbackTimeS);
  const cameraView = usePuttingStore((state) => state.cameraView);
  const playbackRate = usePuttingStore((state) => state.playbackRate);
  const playing = usePuttingStore((state) => state.playing);
  const updateParameters = usePuttingStore((state) => state.updateParameters);
  const setResult = usePuttingStore((state) => state.setResult);
  const setPlaying = usePuttingStore((state) => state.setPlaying);
  const frameRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number | null>(null);

  const simulation = useMutation({
    mutationFn: simulatePutt3D,
    onSuccess: (response) => {
      setResult(response);
      setPlaying(true);
    },
  });

  useEffect(() => {
    if (!playing || !result) return;
    const durationS = result.duration_s + IMPACT_LEAD_IN_S;
    const animate = (timestamp: number) => {
      const previous = lastFrameRef.current ?? timestamp;
      lastFrameRef.current = timestamp;
      const elapsedS = ((timestamp - previous) / 1000) * playbackRate;
      const store = usePuttingStore.getState();
      const nextTime = Math.min(durationS, store.playbackTimeS + elapsedS);
      store.setPlaybackTime(nextTime);
      if (nextTime >= durationS) {
        store.setPlaying(false);
        lastFrameRef.current = null;
        return;
      }
      frameRef.current = requestAnimationFrame(animate);
    };
    frameRef.current = requestAnimationFrame(animate);
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
      lastFrameRef.current = null;
    };
  }, [playing, playbackRate, result]);

  const currentSample = result
    ? sampleAtPlaybackTime(result.samples, playbackTimeS)
    : null;
  const unitSystem = usePuttingStore((state) => state.unitSystem);
  const show = (si: number, quantity: Quantity, decimals: number) =>
    formatQuantity(si, quantity, unitSystem, decimals);

  const leftPanel = (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <div
        className="border-b p-4"
        style={{ borderColor: "var(--sidekick-color-border)" }}
      >
        <h1 className="heading-page mb-1">3D Putting Simulator</h1>
        <p
          className="text-xs"
          style={{ color: "var(--sidekick-color-text-muted)" }}
        >
          Surface-aware collision, skid, roll, and capture
        </p>
        <UnitSystemSelect />
      </div>
      <Section title="Stroke and Target">
        <UnitRangeControl
          label="Putter Speed"
          valueSi={parameters.putter_speed_mps}
          quantity="speed"
          range={{ metric: [0.3, 4, 0.05], imperial: [0.7, 9, 0.1] }}
          decimals={{ metric: 2, imperial: 2 }}
          onChangeSi={(putter_speed_mps) =>
            updateParameters({ putter_speed_mps })
          }
        />
        <RangeControl
          label="Dynamic Loft"
          value={parameters.loft_deg}
          displayValue={`${parameters.loft_deg.toFixed(1)}°`}
          min={-4}
          max={8}
          step={0.5}
          onChange={(loft_deg) => updateParameters({ loft_deg })}
        />
        <UnitRangeControl
          label="Hole Distance"
          valueSi={parameters.hole_x_m}
          quantity="distance"
          range={{ metric: [1, 8.7, 0.05], imperial: [3, 28.5, 0.5] }}
          decimals={{ metric: 2, imperial: 1 }}
          onChangeSi={(hole_x_m) => updateParameters({ hole_x_m })}
        />
        <UnitRangeControl
          label="Impact Toe Offset"
          valueSi={parameters.impact_toe_m}
          quantity="smallLength"
          range={{ metric: [-60, 60, 1], imperial: [-2.4, 2.4, 0.05] }}
          decimals={SMALL_LENGTH_DECIMALS}
          onChangeSi={(impact_toe_m) => updateParameters({ impact_toe_m })}
        />
      </Section>
      <Section title="Putter Head">
        <PutterHeadControl />
      </Section>
      <Section title="Shaft Attachment">
        <HoselPositionControl />
      </Section>
      <Section title="Green Surface">
        <RangeControl
          label="Stimp Rating"
          value={parameters.stimp_rating}
          displayValue={`${parameters.stimp_rating.toFixed(1)} ft`}
          min={6}
          max={15}
          step={0.5}
          onChange={(stimp_rating) => updateParameters({ stimp_rating })}
        />
        <RangeControl
          label="Slope Grade"
          value={parameters.grade_percent}
          displayValue={`${parameters.grade_percent.toFixed(1)}%`}
          min={0}
          max={6}
          step={0.1}
          onChange={(grade_percent) => updateParameters({ grade_percent })}
        />
        <RangeControl
          label="Downhill Direction"
          value={parameters.downhill_aspect_deg}
          displayValue={`${parameters.downhill_aspect_deg.toFixed(0)}°`}
          min={-180}
          max={180}
          step={5}
          onChange={(downhill_aspect_deg) =>
            updateParameters({ downhill_aspect_deg })
          }
        />
        <RangeControl
          label="Grain Strength"
          value={parameters.grain_strength}
          displayValue={parameters.grain_strength.toFixed(2)}
          min={0}
          max={0.5}
          step={0.01}
          onChange={(grain_strength) => updateParameters({ grain_strength })}
        />
        <UnitRangeControl
          label="Bump Height"
          valueSi={parameters.bump_height_m}
          quantity="smallLength"
          range={{ metric: [0, 5, 0.25], imperial: [0, 0.2, 0.01] }}
          decimals={{ metric: 1, imperial: 2 }}
          onChangeSi={(bump_height_m) => updateParameters({ bump_height_m })}
        />
        <RangeControl
          label="Friction Variation"
          value={parameters.friction_variation}
          displayValue={`${(parameters.friction_variation * 100).toFixed(0)}%`}
          min={0}
          max={0.4}
          step={0.01}
          onChange={(friction_variation) =>
            updateParameters({ friction_variation })
          }
        />
      </Section>
      <div className="p-4">
        <button
          type="button"
          onClick={() => simulation.mutate(parameters)}
          disabled={simulation.isPending}
          className="w-full rounded px-4 py-2 text-sm font-semibold disabled:opacity-60"
          style={{
            backgroundColor: "var(--sidekick-color-accent)",
            color: "var(--sidekick-color-selection-text)",
          }}
        >
          {simulation.isPending ? "Simulating…" : "Simulate Putt"}
        </button>
        {simulation.error && (
          <p
            className="mt-3 rounded border p-2 text-xs"
            role="alert"
            style={{
              borderColor: "var(--sidekick-color-error)",
              color: "var(--sidekick-color-error)",
            }}
          >
            {simulation.error.message}
          </p>
        )}
      </div>
    </div>
  );

  const mainContent = (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1">
        {result ? (
          <PuttingScene3D
            result={result}
            playbackTimeS={playbackTimeS}
            hoselToeM={parameters.hosel_toe_m}
            hoselForwardM={parameters.hosel_forward_m}
            view={cameraView}
          />
        ) : (
          <div
            className="flex h-full items-center justify-center p-8 text-center text-sm"
            style={{
              backgroundColor: "var(--sidekick-color-canvas)",
              color: "var(--sidekick-color-text-muted)",
            }}
          >
            Choose the stroke, shaft attachment, and green, then simulate the
            putt.
          </div>
        )}
      </div>
      <PlaybackControls />
    </div>
  );

  const rightPanel = (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-4">
      <h2 className="mb-4 text-sm font-semibold">Impact and Roll Readout</h2>
      {result && currentSample ? (
        <dl className="space-y-3 text-xs">
          {[
            ["Motion Mode", currentSample.mode],
            ["Ball Speed", show(currentSample.speed_mps, "speed", 3)],
            [
              "Ball Launch Speed",
              show(result.collision.ball_speed_mps, "speed", 3),
            ],
            [
              "Launch Angle",
              `${result.collision.launch_angle_deg.toFixed(2)}°`,
            ],
            [
              "Putter Before Impact",
              show(result.collision.putter_speed_before_mps, "speed", 3),
            ],
            [
              "Putter After Impact",
              show(result.collision.putter_speed_after_mps, "speed", 3),
            ],
            [
              "Contact Proxy",
              `${(result.collision.contact_time_proxy_s * 1000).toFixed(2)} ms`,
            ],
            [
              "Face Twist Proxy",
              `${result.collision.face_twist_rad_s.toFixed(3)} rad/s`,
            ],
            [
              "Impact Energy Loss",
              `${result.collision.kinetic_energy_loss_j.toFixed(4)} J`,
            ],
            ["Skid Distance", show(result.skid_distance_m, "distance", 2)],
            ["Total Distance", show(result.total_distance_m, "distance", 2)],
            ["Outcome", result.holed ? "Holed" : "Stopped"],
            // ADR-0045 F1 (#9343): results from different roll models
            // must never be compared without the model name shown.
            ["Roll Model", result.roll_model],
          ].map(([label, value]) => (
            <div
              key={label}
              className="flex items-start justify-between gap-3 border-b pb-2"
              style={{ borderColor: "var(--sidekick-color-border)" }}
            >
              <dt style={{ color: "var(--sidekick-color-text-muted)" }}>
                {label}
              </dt>
              <dd className="text-right font-mono">{value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p
          className="text-xs"
          style={{ color: "var(--sidekick-color-text-muted)" }}
        >
          Collision and trajectory quantities appear after simulation.
        </p>
      )}
      <h2 className="mb-3 mt-6 text-sm font-semibold">Batch Dispersion</h2>
      <DispersionPanel />
    </div>
  );

  return (
    <WorkspaceShell
      leftPanel={leftPanel}
      rightPanel={rightPanel}
      leftPanelLabel="Putting Controls"
      rightPanelLabel="Impact and Roll Readout"
    >
      {mainContent}
    </WorkspaceShell>
  );
}
