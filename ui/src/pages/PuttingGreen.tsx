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
import { PuttingScene3D } from "@/components/visualization/PuttingScene3D";
import { usePuttingStore } from "@/stores/usePuttingStore";
import { IMPACT_LEAD_IN_S, sampleAtPlaybackTime } from "./puttingPlayback";

export type PuttResult = PuttSimulationResponse;
export type GreenReading = GreenReadingResponse;
export type ScatterResult = ScatterAnalysisResponse;

const METERS_PER_YARD = 0.9144;

interface RangeControlProps {
  label: string;
  value: number;
  displayValue: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function RangeControl({
  label,
  value,
  displayValue,
  min,
  max,
  step,
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
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full cursor-pointer"
        style={{ accentColor: "var(--sidekick-color-accent)" }}
      />
    </label>
  );
}

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
        <span className="absolute bottom-1 left-2 text-[10px]">Heel</span>
        <span className="absolute bottom-1 right-2 text-[10px]">Toe</span>
        <span className="absolute left-1 top-1 text-[10px]">Forward</span>
      </div>
      <RangeControl
        label="Hosel Heel to Toe"
        value={parameters.hosel_toe_m}
        displayValue={`${(parameters.hosel_toe_m * 1000).toFixed(0)} mm`}
        min={-0.08}
        max={0.08}
        step={0.001}
        onChange={(hosel_toe_m) => updateParameters({ hosel_toe_m })}
      />
      <RangeControl
        label="Hosel Back to Forward"
        value={parameters.hosel_forward_m}
        displayValue={`${(parameters.hosel_forward_m * 1000).toFixed(0)} mm`}
        min={-0.05}
        max={0.05}
        step={0.001}
        onChange={(hosel_forward_m) => updateParameters({ hosel_forward_m })}
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
    </div>
  );
}

export function PuttingGreenPage() {
  const parameters = usePuttingStore((state) => state.parameters);
  const result = usePuttingStore((state) => state.result);
  const playbackTimeS = usePuttingStore((state) => state.playbackTimeS);
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
  const distanceYards = parameters.hole_x_m / METERS_PER_YARD;

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
      </div>
      <Section title="Stroke and Target">
        <RangeControl
          label="Putter Speed"
          value={parameters.putter_speed_mps}
          displayValue={`${parameters.putter_speed_mps.toFixed(2)} m/s`}
          min={0.3}
          max={4}
          step={0.05}
          onChange={(putter_speed_mps) =>
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
        <RangeControl
          label="Hole Distance"
          value={distanceYards}
          displayValue={`${distanceYards.toFixed(1)} yd`}
          min={1}
          max={9.5}
          step={0.25}
          onChange={(yards) =>
            updateParameters({ hole_x_m: yards * METERS_PER_YARD })
          }
        />
        <RangeControl
          label="Impact Toe Offset"
          value={parameters.impact_toe_m}
          displayValue={`${(parameters.impact_toe_m * 1000).toFixed(0)} mm`}
          min={-0.06}
          max={0.06}
          step={0.001}
          onChange={(impact_toe_m) => updateParameters({ impact_toe_m })}
        />
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
        <RangeControl
          label="Bump Height"
          value={parameters.bump_height_m}
          displayValue={`${(parameters.bump_height_m * 1000).toFixed(1)} mm`}
          min={0}
          max={0.005}
          step={0.00025}
          onChange={(bump_height_m) => updateParameters({ bump_height_m })}
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
            ["Ball Speed", `${currentSample.speed_mps.toFixed(3)} m/s`],
            [
              "Ball Launch Speed",
              `${result.collision.ball_speed_mps.toFixed(3)} m/s`,
            ],
            [
              "Launch Angle",
              `${result.collision.launch_angle_deg.toFixed(2)}°`,
            ],
            [
              "Putter Before Impact",
              `${result.collision.putter_speed_before_mps.toFixed(3)} m/s`,
            ],
            [
              "Putter After Impact",
              `${result.collision.putter_speed_after_mps.toFixed(3)} m/s`,
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
            [
              "Skid Distance",
              `${(result.skid_distance_m / METERS_PER_YARD).toFixed(2)} yd`,
            ],
            [
              "Total Distance",
              `${(result.total_distance_m / METERS_PER_YARD).toFixed(2)} yd`,
            ],
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
