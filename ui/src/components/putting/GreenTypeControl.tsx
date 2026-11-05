/** Green Type selector on the shared core's green presets (#14464). */
import { useQuery } from "@tanstack/react-query";

import { fetchGreenPresets } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import { greenFromType, greenLabel } from "@/utils/greenTypes";

export function GreenTypeControl() {
  const green = usePuttingStore((state) => state.parameters.green);
  const updateParameters = usePuttingStore((state) => state.updateParameters);
  const presets = useQuery({
    queryKey: ["putting-green", "green-presets"],
    queryFn: fetchGreenPresets,
    staleTime: Infinity,
  });
  const choices = presets.data ?? {};

  return (
    <label className="block text-xs">
      <span style={{ color: "var(--sidekick-color-text-muted)" }}>
        Green Type
      </span>
      <select
        aria-label="Green Type"
        value={green?.preset ?? ""}
        onChange={(event) =>
          updateParameters({
            green: greenFromType(event.target.value || null, choices),
          })
        }
        className="mt-1 w-full rounded border px-2 py-1"
        style={{
          backgroundColor: "var(--sidekick-color-input)",
          borderColor: "var(--sidekick-color-border)",
        }}
      >
        <option value="">{greenLabel(null)}</option>
        {Object.keys(choices).map((key) => (
          <option key={key} value={key}>
            {greenLabel(key)}
          </option>
        ))}
      </select>
    </label>
  );
}
