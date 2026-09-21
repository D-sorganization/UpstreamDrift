/**
 * GolfSimulator — capability-aware controls for simulator integration (GS-07, #10196).
 *
 * Provides web counterpart of desktop Golf Simulator controls:
 * - Destination selection (Local, GSPro)
 * - Session lifecycle (IDLE, PREPARED, ARMED, SUBMITTING, UNCERTAIN)
 * - Shot preparation, arming, disarming, cancelling, and one-impact submission
 * - Clear delivery status badges: DISCONNECTED, CONNECTED, ARMED, SENT_UNCONFIRMED, ACCEPTED, REJECTED, VISUALLY_VERIFIED
 * - Monotonic replay transport controls
 * - Safe operator recovery for uncertain delivery
 */

import { useState, useEffect, useMemo } from "react";
import { apiFetch } from "@/api/fetch";
import { WorkspaceShell } from "@/components/layout/WorkspaceShell";

export type DeliveryStatus =
  | "DISCONNECTED"
  | "CONNECTED"
  | "PREPARED"
  | "ARMED"
  | "SENT_UNCONFIRMED"
  | "ACCEPTED"
  | "REJECTED"
  | "VISUALLY_VERIFIED";

export interface CapabilityItem {
  state: string;
  evidence: string;
}

export interface Capabilities {
  shot_input?: CapabilityItem;
  club_data?: CapabilityItem;
  local_trajectory_return?: CapabilityItem;
}

export interface DestinationItem {
  destination_id: string;
  name: string;
  description: string;
  is_connected: boolean;
  capabilities?: Capabilities;
}

function getErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return fallback;
}

export function GolfSimulatorPage() {
  const [destinations, setDestinations] = useState<DestinationItem[]>([]);
  const [selectedDest, setSelectedDest] = useState<string>("local");
  const [sessionState, setSessionState] = useState<DeliveryStatus>("DISCONNECTED");
  const [preparedShotId, setPreparedShotId] = useState<string | null>(null);
  const [armToken, setArmToken] = useState<string | null>(null);
  const [replayState, setReplayState] = useState<string>("stopped");
  const [replayTimeS, setReplayTimeS] = useState<number>(0.0);
  const [operatorEvidence, setOperatorEvidence] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch<{ destinations: DestinationItem[] }>("/tools/golf-simulator/destinations")
      .then((res) => {
        if (!cancelled && res?.destinations) {
          setDestinations(res.destinations);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setErrorMsg(getErrorMessage(err, "Failed to load destinations"));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const capabilities = useMemo(() => {
    const dest = destinations.find((d) => d.destination_id === selectedDest);
    return dest?.capabilities ?? null;
  }, [destinations, selectedDest]);

  const handleConnect = async () => {
    try {
      setErrorMsg(null);
      const res = await apiFetch<{ state: string; session_id: string }>("/tools/golf-simulator/session", {
        method: "POST",
        body: JSON.stringify({ destination_id: selectedDest, session_id: "web-session" }),
      });
      setSessionState(res.state === "idle" ? "CONNECTED" : (res.state.toUpperCase() as DeliveryStatus));
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to connect to simulator"));
    }
  };

  const handlePrepare = async () => {
    try {
      setErrorMsg(null);
      const res = await apiFetch<{ prepared_shot_id: string }>("/tools/golf-simulator/shot/prepare", {
        method: "POST",
        body: JSON.stringify({
          shot: {
            schema_version: 1,
            shot_id: `shot-${Date.now()}`,
            session_id: "web-session",
            source_kind: "manual",
            ball_velocity_m_s: [65.0, 0.0, 14.0],
            ball_angular_velocity_rad_s: [0.0, -260.0, 0.0],
            aim_context: {
              source_to_target_rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
              ],
              revision: 1,
            },
            qualification: {
              contact: "qualified",
              numerical: "converged",
              scientific: "benchmarked",
              evidence_refs: ["web-ui"],
            },
            created_at_utc: new Date().toISOString(),
          },
          context_revision: 1,
        }),
      });
      setPreparedShotId(res.prepared_shot_id);
      setSessionState("PREPARED");
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to prepare shot"));
    }
  };

  const handleArm = async () => {
    if (!preparedShotId) return;
    try {
      setErrorMsg(null);
      const res = await apiFetch<{ arm_token: string; state: string }>("/tools/golf-simulator/shot/arm", {
        method: "POST",
        body: JSON.stringify({ prepared_shot_id: preparedShotId, context_revision: 1 }),
      });
      setArmToken(res.arm_token);
      setSessionState("ARMED");
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to arm shot"));
    }
  };

  const handleDisarm = async () => {
    if (!preparedShotId) return;
    try {
      setErrorMsg(null);
      await apiFetch("/tools/golf-simulator/shot/disarm", {
        method: "POST",
        body: JSON.stringify({ prepared_shot_id: preparedShotId }),
      });
      setArmToken(null);
      setSessionState("PREPARED");
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to disarm shot"));
    }
  };

  const handleCancel = async () => {
    if (!preparedShotId) return;
    try {
      setErrorMsg(null);
      await apiFetch("/tools/golf-simulator/shot/cancel", {
        method: "POST",
        body: JSON.stringify({ prepared_shot_id: preparedShotId }),
      });
      setPreparedShotId(null);
      setArmToken(null);
      setSessionState("CONNECTED");
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to cancel shot"));
    }
  };

  const handleSubmit = async () => {
    if (!preparedShotId || !armToken) return;
    try {
      setErrorMsg(null);
      const res = await apiFetch<{ state: string; shot_id: string }>("/tools/golf-simulator/shot/submit", {
        method: "POST",
        body: JSON.stringify({ prepared_shot_id: preparedShotId, arm_token: armToken }),
      });
      setPreparedShotId(null);
      setArmToken(null);
      if (res.state === "confirmed_accepted") {
        setSessionState("ACCEPTED");
      } else if (res.state === "unknown_ambiguous") {
        setSessionState("SENT_UNCONFIRMED");
      } else {
        setSessionState("REJECTED");
      }
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to submit shot"));
    }
  };

  const handleReconcile = async () => {
    if (!operatorEvidence) return;
    try {
      setErrorMsg(null);
      await apiFetch("/tools/golf-simulator/shot/pending/resolve", {
        method: "POST",
        body: JSON.stringify({ operator_evidence: operatorEvidence, confirmed: true }),
      });
      setSessionState("VISUALLY_VERIFIED");
      setOperatorEvidence("");
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Failed to reconcile delivery"));
    }
  };

  const handleReplayAction = async (action: "play" | "pause" | "stop") => {
    try {
      setErrorMsg(null);
      const res = await apiFetch<{ playback_state: string; current_time_s: number }>(
        "/tools/golf-simulator/replay/action",
        {
          method: "POST",
          body: JSON.stringify({ action }),
        },
      );
      setReplayState(res.playback_state);
      setReplayTimeS(res.current_time_s);
    } catch (err: unknown) {
      setErrorMsg(getErrorMessage(err, "Replay action failed"));
    }
  };

  const isArmed = sessionState === "ARMED";
  const isPrepared = sessionState === "PREPARED";
  const isUncertain = sessionState === "SENT_UNCONFIRMED";

  const getStatusBadgeStyle = () => {
    switch (sessionState) {
      case "DISCONNECTED":
        return "bg-gray-700 text-gray-200";
      case "CONNECTED":
        return "bg-green-700 text-white";
      case "PREPARED":
        return "bg-blue-600 text-white";
      case "ARMED":
        return "bg-red-600 text-white font-bold animate-pulse";
      case "SENT_UNCONFIRMED":
        return "bg-amber-600 text-white font-bold";
      case "ACCEPTED":
      case "VISUALLY_VERIFIED":
        return "bg-emerald-600 text-white";
      case "REJECTED":
        return "bg-rose-700 text-white";
      default:
        return "bg-gray-600 text-white";
    }
  };

  return (
    <WorkspaceShell>
      <div className="space-y-6 max-w-4xl mx-auto p-4 text-gray-200">
        <h1 className="text-xl font-bold">Golf Simulator Console</h1>
        {errorMsg && (
          <div role="alert" className="p-3 bg-red-900/50 border border-red-500 rounded text-red-200 text-sm">
            {errorMsg}
          </div>
        )}

        {/* Header & Connection */}
        <section className="bg-gray-800 p-4 rounded-lg border border-gray-700 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <label htmlFor="destination-select" className="text-sm font-medium">
              Destination:
            </label>
            <select
              id="destination-select"
              aria-label="Simulator Destination"
              disabled={isArmed}
              value={selectedDest}
              onChange={(e) => setSelectedDest(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500"
            >
              {destinations.length === 0 ? (
                <>
                  <option value="local">Local Reference Simulator</option>
                  <option value="gspro">GSPro Open Connect v1</option>
                </>
              ) : (
                destinations.map((d) => (
                  <option key={d.destination_id} value={d.destination_id}>
                    {d.name}
                  </option>
                ))
              )}
            </select>
            <button
              onClick={handleConnect}
              disabled={isArmed}
              className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-sm font-medium"
            >
              Connect
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Status:</span>
            <span data-testid="status-badge" className={`px-3 py-1 rounded text-xs tracking-wider uppercase ${getStatusBadgeStyle()}`}>
              {sessionState}
            </span>
          </div>
        </section>

        {/* Capability Indicators */}
        {capabilities && (
          <section className="bg-gray-800/80 p-3 rounded-lg border border-gray-700 flex flex-wrap items-center gap-4 text-xs">
            <span className="text-gray-400 font-medium">Capabilities:</span>
            {capabilities.shot_input && (
              <span className="px-2 py-0.5 rounded bg-gray-700 text-gray-200">
                Shot Input: <span className="font-semibold text-emerald-400">{capabilities.shot_input.state}</span>
              </span>
            )}
            {capabilities.club_data && (
              <span className="px-2 py-0.5 rounded bg-gray-700 text-gray-200">
                Club Data: <span className="font-semibold text-sky-400">{capabilities.club_data.state}</span>
              </span>
            )}
            {capabilities.local_trajectory_return && (
              <span className="px-2 py-0.5 rounded bg-gray-700 text-gray-200">
                Local Trajectory: <span className="font-semibold text-blue-300">{capabilities.local_trajectory_return.state}</span>
              </span>
            )}
          </section>
        )}

        {/* Shot Lifecycle Controls */}
        <section className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Shot Lifecycle & Impact</h2>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={handlePrepare}
              disabled={sessionState === "DISCONNECTED" || isArmed}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white rounded text-sm font-medium"
            >
              Prepare Shot
            </button>

            <button
              onClick={handleArm}
              disabled={!isPrepared}
              className="px-4 py-2 bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white rounded text-sm font-medium"
            >
              Arm for Impact
            </button>

            <button
              onClick={handleDisarm}
              disabled={!isArmed}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white rounded text-sm font-medium"
            >
              Disarm
            </button>

            <button
              onClick={handleCancel}
              disabled={!isPrepared && !isArmed}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white rounded text-sm font-medium"
            >
              Cancel
            </button>

            <button
              onClick={handleSubmit}
              disabled={!isArmed}
              className="px-5 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-40 text-white rounded text-sm font-bold ml-auto"
            >
              Trigger Impact Submit
            </button>
          </div>
        </section>

        {/* Monotonic Replay Transport */}
        <section className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Replay Transport</h2>
          <div className="flex items-center gap-3">
            <button
              onClick={() => handleReplayAction("play")}
              disabled={sessionState === "DISCONNECTED"}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white rounded text-sm"
            >
              Play
            </button>
            <button
              onClick={() => handleReplayAction("pause")}
              disabled={sessionState === "DISCONNECTED"}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white rounded text-sm"
            >
              Pause
            </button>
            <button
              onClick={() => handleReplayAction("stop")}
              disabled={sessionState === "DISCONNECTED"}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-white rounded text-sm"
            >
              Stop
            </button>
            <span className="text-sm font-mono text-gray-400 ml-4">
              State: <span className="text-gray-200">{replayState}</span> | Time:{" "}
              <span className="text-gray-200">{replayTimeS.toFixed(2)}s</span>
            </span>
          </div>
        </section>

        {/* Delivery Reconciliation / Uncertainty Recovery */}
        <section className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Delivery Uncertainty Recovery</h2>
          <div className="flex items-center gap-3">
            <input
              type="text"
              aria-label="Operator Evidence"
              placeholder="Operator evidence (e.g. Visually confirmed on simulator)"
              value={operatorEvidence}
              onChange={(e) => setOperatorEvidence(e.target.value)}
              disabled={!isUncertain}
              className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm disabled:opacity-40 focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={handleReconcile}
              disabled={!isUncertain || !operatorEvidence}
              className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded text-sm font-medium"
            >
              Confirm Delivery
            </button>
          </div>
        </section>
      </div>
    </WorkspaceShell>
  );
}

export default GolfSimulatorPage;
