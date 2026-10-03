/**
 * Types for renderer-neutral geometric force/torque glyphs (ADR-0052, #11288, #11308).
 * Wire schema authority: schemas/glyph-set-v1.json
 */

export type WrenchKind =
  | 'joint_actuator'
  | 'joint_reaction'
  | 'contact'
  | 'grip'
  | 'external'
  | 'gravity'
  | 'muscle';

export type Vec3 = [number, number, number];
export type RGBA = [number, number, number, number];

export interface ArrowGlyph {
  label: string;
  kind: WrenchKind;
  tail_m: Vec3;
  tip_m: Vec3;
  head_base_m: Vec3;
  shaft_radius_m: number;
  head_radius_m: number;
  rgba: RGBA;
  magnitude: number;
  units: string;
  clamped: boolean;
}

export interface TorqueArcGlyph {
  label: string;
  kind: WrenchKind;
  center_m: Vec3;
  axis_unit: Vec3;
  radius_m: number;
  polyline_m: Vec3[];
  head_tip_m: Vec3;
  head_base_m: Vec3;
  rgba: RGBA;
  magnitude: number;
  units: string;
  clamped: boolean;
}

export interface LegendSpec {
  force_reference_n: number | null;
  force_reference_length_m: number | null;
  torque_reference_nm: number | null;
  torque_reference_radius_m: number | null;
  kinds_present: WrenchKind[];
  unavailable_labels: string[];
  engine: string;
  source_labels: string[];
}

export interface GlyphSetV1 {
  schema_version: 'glyph-set-v1';
  time_s: number;
  arrows: ArrowGlyph[];
  torque_arcs: TorqueArcGlyph[];
  legend: LegendSpec;
}
