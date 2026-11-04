/** Balance type and head mass/MOI from the shared putting core (#14462). */
import { useQuery } from "@tanstack/react-query";

import { fetchPutterPresets } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import {
  HEAD_MASS_G,
  HEAD_MOI_GCM2,
  gCm2ToKgM2,
  headFromPreset,
  kgM2ToGCm2,
} from "@/utils/putterHead";

const inputStyle = {
  backgroundColor: "var(--sidekick-color-input)",
  borderColor: "var(--sidekick-color-border)",
};

function HeadInput({
  label,
  value,
  bounds,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  bounds: { min: number; max: number; step: number };
  disabled: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span style={{ color: "var(--sidekick-color-text-muted)" }}>{label}</span>
      <input
        aria-label={label}
        type="number"
        min={bounds.min}
        max={bounds.max}
        step={bounds.step}
        value={Number(value.toFixed(1))}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1 w-full rounded border px-2 py-1 disabled:opacity-60"
        style={inputStyle}
      />
    </label>
  );
}

export function PutterHeadControl() {
  const parameters = usePuttingStore((state) => state.parameters);
  const updateParameters = usePuttingStore((state) => state.updateParameters);
  const presets = useQuery({
    queryKey: ["putting-green", "putter-presets"],
    queryFn: fetchPutterPresets,
    staleTime: Infinity,
  });
  const choices = presets.data?.presets ?? [];
  const locked = Boolean(parameters.balance_type);

  return (
    <div className="space-y-3 text-xs">
      <label className="block">
        <span style={{ color: "var(--sidekick-color-text-muted)" }}>
          Balance Type
        </span>
        <select
          aria-label="Balance Type"
          value={parameters.balance_type ?? ""}
          onChange={(event) =>
            updateParameters(
              headFromPreset(
                choices.find((p) => p.key === event.target.value) ?? null,
              ),
            )
          }
          className="mt-1 w-full rounded border px-2 py-1"
          style={inputStyle}
        >
          <option value="">Typical</option>
          {choices.map((preset) => (
            <option key={preset.key} value={preset.key}>
              {preset.name}
            </option>
          ))}
        </select>
      </label>
      {presets.isError && (
        <p role="alert" style={{ color: "var(--sidekick-color-error)" }}>
          Balance presets unavailable from the shared core; the typical head is
          still editable.
        </p>
      )}
      <div className="grid grid-cols-2 gap-2">
        <HeadInput
          label="Head Mass (g)"
          value={parameters.head_mass_kg * 1000}
          bounds={HEAD_MASS_G}
          disabled={locked}
          onChange={(grams) => updateParameters({ head_mass_kg: grams / 1000 })}
        />
        <HeadInput
          label="Head MOI (g·cm²)"
          value={kgM2ToGCm2(parameters.head_moi_kg_m2)}
          bounds={HEAD_MOI_GCM2}
          disabled={locked}
          onChange={(moi) =>
            updateParameters({ head_moi_kg_m2: gCm2ToKgM2(moi) })
          }
        />
      </div>
      {locked && (
        <p style={{ color: "var(--sidekick-color-text-muted)" }}>
          The shared core sets this head&apos;s mass, MOI and hosel offsets;
          choose Typical to edit them.
        </p>
      )}
    </div>
  );
}
