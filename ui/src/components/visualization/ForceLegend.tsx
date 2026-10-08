/**
 * ForceLegend - Legend display for force and torque overlays (ADR-0052, #11308).
 *
 * Renders the reference arrow and arc values with units, the kind swatches,
 * the unavailable list, and the engine/source information.
 */

import type { GlyphSetV1, WrenchKind } from '@/types/glyphs';

export interface ForceLegendProps {
  glyphs?: GlyphSetV1 | null;
  className?: string;
}

const KIND_INFO: Record<WrenchKind, { label: string; color: string }> = {
  joint_actuator: { label: 'Actuator Torque', color: '#E69F00' },
  joint_reaction: { label: 'Joint Reaction', color: '#56B4E9' },
  contact: { label: 'Contact Force', color: '#009E73' },
  grip: { label: 'Grip Force (Net)', color: '#56B4E9' },
  external: { label: 'External Force', color: '#0072B2' },
  gravity: { label: 'Gravity', color: '#D55E00' },
  muscle: { label: 'Muscle Force', color: '#CC79A7' },
};

export function ForceLegend({ glyphs, className = '' }: ForceLegendProps) {
  if (!glyphs?.legend) {
    return null;
  }

  const { legend } = glyphs;
  const hasForceRef =
    legend.force_reference_n !== null &&
    legend.force_reference_length_m !== null;
  const hasTorqueRef =
    legend.torque_reference_nm !== null &&
    legend.torque_reference_radius_m !== null;

  return (
    <div
      className={`bg-black/80 backdrop-blur-md p-3 rounded-lg border border-white/10 text-xs text-gray-200 shadow-xl space-y-2 select-none ${className}`}
      data-testid="force-legend"
    >
      <div className="flex items-center justify-between border-b border-white/10 pb-1 font-semibold text-gray-300">
        <span>Force & Torque Legend</span>
        <span className="text-[10px] text-gray-400 uppercase font-mono">
          {legend.engine}
        </span>
      </div>

      {/* Reference Scales */}
      {(hasForceRef || hasTorqueRef) && (
        <div className="space-y-1 font-mono text-[11px] text-gray-300">
          {hasForceRef && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-gray-400">Force Ref:</span>
              <span>
                {legend.force_reference_n} N (
                {legend.force_reference_length_m} m)
              </span>
            </div>
          )}
          {hasTorqueRef && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-gray-400">Torque Ref:</span>
              <span>
                {legend.torque_reference_nm} N*m (r ={' '}
                {legend.torque_reference_radius_m} m)
              </span>
            </div>
          )}
        </div>
      )}

      {/* Kind Swatches */}
      {legend.kinds_present?.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-white/5">
          <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">
            Kinds
          </div>
          <div className="grid grid-cols-1 gap-1">
            {legend.kinds_present.map((kind) => {
              const info = KIND_INFO[kind] || {
                label: kind,
                color: '#FFFFFF',
              };
              return (
                <div key={kind} className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-sm flex-shrink-0 border border-black/30"
                    style={{ backgroundColor: info.color }}
                  />
                  <span className="text-gray-300">{info.label}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Grip split (GCV-10): per-hand shades and the method behind the split */}
      {legend.grip_split_method && (
        <div className="pt-1 border-t border-white/5 text-[11px] space-y-1" data-testid="grip-legend">
          <div>
            <span className="text-gray-400">Grip split: </span>
            <span className="font-mono text-sky-300" data-testid="grip-legend-split">
              {legend.grip_split_method}
            </span>
          </div>
          <div className="flex gap-3 text-gray-300">
            <span><span style={{ color: '#9AD0F2' }}>&#9632;</span> Left hand</span>
            <span><span style={{ color: '#1B6C99' }}>&#9632;</span> Right hand</span>
            <span><span style={{ color: '#56B4E9' }}>&#9632;</span> Net / couple</span>
          </div>
        </div>
      )}

      {/* Unavailable Channels */}
      {legend.unavailable_labels?.length > 0 && (
        <div className="pt-1 border-t border-white/5 text-[11px]">
          <span className="text-amber-400/80 font-medium">Unavailable: </span>
          <span className="text-gray-400 font-mono">
            {legend.unavailable_labels.join(', ')}
          </span>
        </div>
      )}

      {/* Source labels */}
      {legend.source_labels?.length > 0 && (
        <div className="text-[10px] text-gray-400 font-mono pt-1">
          Sources: {legend.source_labels.join(', ')}
        </div>
      )}
    </div>
  );
}
