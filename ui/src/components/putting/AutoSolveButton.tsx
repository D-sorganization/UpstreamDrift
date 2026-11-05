import { useMutation } from "@tanstack/react-query";

import type { Putt3DSimulationRequest } from "@/api/generated/types";
import { solveAim3D } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import { formatRead, solvedControls } from "@/utils/autoSolve";

interface AutoSolveButtonProps {
  /** Runs the solved putt, as the desktop reruns its simulation. */
  onSolved: (parameters: Putt3DSimulationRequest) => void;
}

/** Desktop "Auto-Solve Read" (#14466): the core's pace and line for this green. */
export function AutoSolveButton({ onSolved }: AutoSolveButtonProps) {
  const parameters = usePuttingStore((state) => state.parameters);
  const unitSystem = usePuttingStore((state) => state.unitSystem);
  const updateParameters = usePuttingStore((state) => state.updateParameters);
  const solve = useMutation({
    mutationFn: solveAim3D,
    onSuccess: (response, request) => {
      const controls = solvedControls(response);
      updateParameters(controls);
      onSolved({ ...request, ...controls });
    },
  });

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => solve.mutate(parameters)}
        disabled={solve.isPending}
        title="Solve the pace and line that hole this putt"
        className="w-full rounded border px-4 py-2 text-sm font-semibold disabled:opacity-60"
        style={{
          borderColor: "var(--sidekick-color-accent)",
          color: "var(--sidekick-color-text)",
        }}
      >
        {solve.isPending ? "Solving…" : "🎯 Auto-Solve Read"}
      </button>
      {solve.data && (
        <p
          className="mt-3 whitespace-pre-wrap text-xs"
          role="status"
          style={{ color: "var(--sidekick-color-text)" }}
        >
          {formatRead(solve.data, unitSystem)}
        </p>
      )}
      {solve.error && (
        <p
          className="mt-3 rounded border p-2 text-xs"
          role="alert"
          style={{
            borderColor: "var(--sidekick-color-error)",
            color: "var(--sidekick-color-error)",
          }}
        >
          Auto-Solve rejected: {solve.error.message}
        </p>
      )}
    </div>
  );
}
