import { binRows, displayStatus, type SpineHeatmapPayload } from '@/api/spineViewer';

const MAX_ROWS = 200;
const CELL_W = 40;
const CELL_H = 3;
const LABEL_H = 16;

function cellFill(value: number, scale: number): string {
  const alpha = scale > 0 ? Math.min(1, Math.abs(value) / scale) : 0;
  const rgb = value >= 0 ? '217,119,6' : '37,99,235';
  return `rgba(${rgb},${alpha.toFixed(3)})`;
}

/**
 * Time-by-level load heatmap (rows: time, columns: levels caudal to cranial).
 * Orange is positive, blue negative, in the payload's sign convention.
 */
export function SpineHeatmap({
  heatmap,
  levels,
}: {
  heatmap: SpineHeatmapPayload;
  levels: string[];
}) {
  const rows = binRows(heatmap.values, MAX_ROWS);
  const flat = rows.flat();
  const scale = flat.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const width = levels.length * CELL_W;
  const height = rows.length * CELL_H;
  return (
    <figure
      data-testid="spine-heatmap"
      data-quantity={heatmap.quantity}
      data-status={displayStatus(heatmap.scientific_status)}
    >
      <figcaption className="mb-2 text-xs text-gray-300">
        {heatmap.status_label} - {heatmap.quantity} ({heatmap.units},{' '}
        {heatmap.sign_convention}) - source: {heatmap.distribution_source}
        <br />
        Peak magnitude {scale.toFixed(1)} {heatmap.units}; time runs top to
        bottom ({heatmap.time_s[0] ?? 0} s to{' '}
        {heatmap.time_s[heatmap.time_s.length - 1] ?? 0} s).
      </figcaption>
      <svg
        viewBox={`0 0 ${width} ${height + LABEL_H}`}
        className="w-full max-w-xl"
        role="img"
        aria-label={`${heatmap.quantity} heatmap, ${heatmap.status_label}`}
      >
        {rows.map((row, r) =>
          row.map((value, c) => (
            <rect
              key={`${r}-${c}`}
              data-testid="spine-heatmap-cell"
              data-value={value}
              x={c * CELL_W}
              y={r * CELL_H}
              width={CELL_W}
              height={CELL_H}
              fill={cellFill(value, scale)}
            />
          )),
        )}
        {levels.map((name, c) => (
          <text
            key={name}
            x={c * CELL_W + CELL_W / 2}
            y={height + 12}
            textAnchor="middle"
            fontSize={7}
            fill="currentColor"
          >
            {name}
          </text>
        ))}
      </svg>
    </figure>
  );
}
