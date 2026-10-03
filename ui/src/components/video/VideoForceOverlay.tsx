/**
 * VideoForceOverlay - SVG projection layer for force/torque glyphs (FTO-29, #11314).
 *
 * Renders server-projected 2D force/torque glyphs aligned with video frames.
 * Arrows are rendered as polyline shafts with dark halos underneath and polygon heads.
 * Moment arcs are rendered as pixel polylines.
 * Uses vector-effect="non-scaling-stroke" to maintain crisp line rendering across container resizes.
 */


export interface ProjectedArrowGlyphPayload {
  start_px: [number, number];
  end_px: [number, number];
  polyline_px: [number, number][];
  head_poly_px: [number, number][];
  rgba: [number, number, number, number];
  color_hex: string;
  kind: string;
  label: string;
  magnitude: number;
  units: string;
  shaft_width_px: number;
  halo_width_px: number;
}

export interface ProjectedTorqueArcGlyphPayload {
  polyline_px: [number, number][];
  head_poly_px: [number, number][] | null;
  rgba: [number, number, number, number];
  color_hex: string;
  kind: string;
  label: string;
  magnitude: number;
  units: string;
  shaft_width_px: number;
  halo_width_px: number;
}

export interface ProjectedGlyphLegendPayload {
  engine?: string;
  force_reference_n?: number | null;
  torque_reference_nm?: number | null;
  kinds_present: string[];
  unavailable_labels: string[];
  source_labels: string[];
}

export interface VideoGlyphReceiptPayload {
  drawn: number;
  skipped_behind_camera: number;
  skipped_out_of_frame: number;
  unavailable_labels: string[];
}

export interface ProjectedGlyphSetPayload {
  time_s: number;
  image_size_px: [number, number];
  arrows: ProjectedArrowGlyphPayload[];
  torque_arcs: ProjectedTorqueArcGlyphPayload[];
  legend: ProjectedGlyphLegendPayload;
  receipt: VideoGlyphReceiptPayload;
}

export interface VideoForceOverlayProps {
  glyphs: ProjectedGlyphSetPayload | null;
  width: number;
  height: number;
  visible?: boolean;
  showLegend?: boolean;
  scale?: number;
}

function pointsToString(pts: [number, number][]): string {
  return pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
}

export function VideoForceOverlay({
  glyphs,
  width,
  height,
  visible = true,
  showLegend = true,
}: VideoForceOverlayProps) {


  if (!visible || !glyphs) {
    return null;
  }

  const { arrows = [], torque_arcs = [], legend } = glyphs;
  const heightRatio = Math.max(0.1, height / 1080.0);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="absolute inset-0 w-full h-full pointer-events-none"
      data-testid="video-force-overlay"
      aria-label="Force and Torque Vector Overlay"
    >
      {/* 1. Arrows: Halo precedes stroke in DOM order (painters-model) */}
      {arrows.map((arrow, idx) => {
        const shaftW = Math.max(1, (arrow.shaft_width_px || 2.5) * heightRatio);
        const haloW = shaftW + Math.max(2, 2 * heightRatio);
        const polyPoints = pointsToString(arrow.polyline_px);
        const headPoints = pointsToString(arrow.head_poly_px);
        const color = arrow.color_hex || '#FFFFFF';

        return (
          <g key={`arrow-${idx}-${arrow.label}`}>
            {/* Halo polyline shaft (underneath) */}
            <polyline
              points={polyPoints}
              stroke="rgba(16, 16, 16, 0.85)"
              strokeWidth={haloW}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              data-role="halo"
            />
            {/* Shaft polyline (on top) */}
            <polyline
              points={polyPoints}
              stroke={color}
              strokeWidth={shaftW}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              data-role="shaft"
            />
            {/* Halo polygon head (underneath) */}
            <polygon
              points={headPoints}
              fill="rgba(16, 16, 16, 0.85)"
              stroke="rgba(16, 16, 16, 0.85)"
              strokeWidth={Math.max(1, haloW - shaftW)}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              data-role="halo"
            />
            {/* Head polygon (on top) */}
            <polygon
              points={headPoints}
              fill={color}
              stroke={color}
              strokeWidth={1}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              data-role="head"
            >
              <title>{`${arrow.label}: ${arrow.magnitude.toFixed(1)} ${arrow.units}`}</title>
            </polygon>
          </g>
        );
      })}

      {/* 2. Torque Arcs: Halo precedes stroke */}
      {torque_arcs.map((arc, idx) => {
        const shaftW = Math.max(1, (arc.shaft_width_px || 2.5) * heightRatio);
        const haloW = shaftW + Math.max(2, 2 * heightRatio);
        const polyPoints = pointsToString(arc.polyline_px);
        const color = arc.color_hex || '#FFFFFF';

        return (
          <g key={`arc-${idx}-${arc.label}`}>
            <polyline
              points={polyPoints}
              stroke="rgba(16, 16, 16, 0.85)"
              strokeWidth={haloW}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              data-role="halo"
            />
            <polyline
              points={polyPoints}
              stroke={color}
              strokeWidth={shaftW}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              data-role="shaft"
            />
            {arc.head_poly_px && (
              <>
                <polygon
                  points={pointsToString(arc.head_poly_px)}
                  fill="rgba(16, 16, 16, 0.85)"
                  stroke="rgba(16, 16, 16, 0.85)"
                  strokeWidth={Math.max(1, haloW - shaftW)}
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                  data-role="halo"
                />
                <polygon
                  points={pointsToString(arc.head_poly_px)}
                  fill={color}
                  stroke={color}
                  strokeWidth={1}
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                  data-role="head"
                >
                  <title>{`${arc.label}: ${arc.magnitude.toFixed(1)} ${arc.units}`}</title>
                </polygon>
              </>
            )}
          </g>
        );
      })}

      {/* 3. Legend Box */}
      {showLegend && legend && (
        <g
          data-testid="force-overlay-legend"
          transform={`translate(${Math.max(12, 16 * heightRatio)}, ${Math.max(
            height - 120 * heightRatio,
            40,
          )})`}
          className="select-none pointer-events-auto"
        >
          <rect
            width={240 * heightRatio}
            height={90 * heightRatio}
            rx={4}
            fill="rgba(17, 24, 39, 0.85)"
            stroke="rgba(75, 85, 99, 0.6)"
            strokeWidth={1}
          />
          <text
            x={10 * heightRatio}
            y={20 * heightRatio}
            fill="#F3F4F6"
            fontSize={12 * heightRatio}
            fontFamily="monospace"
            fontWeight="bold"
          >
            {legend.engine ? `Engine: ${legend.engine}` : 'Forces & Torques'}
          </text>
          {legend.force_reference_n != null && (
            <text
              x={10 * heightRatio}
              y={38 * heightRatio}
              fill="#D1D5DB"
              fontSize={11 * heightRatio}
              fontFamily="monospace"
            >
              Ref: {legend.force_reference_n.toFixed(1)} N
            </text>
          )}
          {legend.torque_reference_nm != null && (
            <text
              x={10 * heightRatio}
              y={54 * heightRatio}
              fill="#D1D5DB"
              fontSize={11 * heightRatio}
              fontFamily="monospace"
            >
              Torque: {legend.torque_reference_nm.toFixed(1)} N*m
            </text>
          )}
          {legend.kinds_present?.length > 0 && (
            <text
              x={10 * heightRatio}
              y={72 * heightRatio}
              fill="#9CA3AF"
              fontSize={10 * heightRatio}
              fontFamily="monospace"
            >
              Kinds: {legend.kinds_present.join(', ')}
            </text>
          )}
        </g>
      )}
    </svg>
  );
}
