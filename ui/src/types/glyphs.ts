/**
 * Types for renderer-neutral geometric force/torque glyphs (ADR-0052, #11288, #11308).
 * Wire schema authority: schemas/glyph-set-v1.json
 */

export type WrenchKind =
  | "joint_actuator"
  | "joint_reaction"
  | "contact"
  | "grip"
  | "external"
  | "gravity"
  | "muscle";

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
  /** Absent on payloads from older servers. */
  scale_mode?: ScaleMode;
  /** Labels whose arrow was shortened to the maximum length (double tip). */
  clamped_labels?: string[];
}

/** Arrow scaling (GCV-4, #11710); mirrors `ForceGlyphStyle.scale_mode`. */
export type ScaleMode = "fixed" | "body_weight" | "peak";

/** Overlay groups selected by label prefix; mirrors `force_overlay.glyphs.ALL_GROUPS`. */
export type GlyphGroup =
  | "per_foot"
  | "net"
  | "free_moment"
  | "moment_about_com"
  | "contact_points"
  | "grip_per_hand"
  | "grip_net"
  | "grip_couple"
  | "grip_mof";

export const GLYPH_GROUP_LABELS: Record<GlyphGroup, string> = {
  per_foot: "Per-Foot GRF",
  net: "Net GRF",
  free_moment: "Free Moment",
  moment_about_com: "Moment About CoM",
  contact_points: "Contact Points",
  grip_per_hand: "Grip Per Hand",
  grip_net: "Grip Net",
  grip_couple: "Grip Couple",
  grip_mof: "Grip MOF",
};

export const ALL_GLYPH_GROUPS = Object.keys(GLYPH_GROUP_LABELS) as GlyphGroup[];

/** Raw per-sphere contacts and moment-about-CoM arcs are opt-in. */
export const DEFAULT_GLYPH_GROUPS: GlyphGroup[] = ALL_GLYPH_GROUPS.filter(
  (g) => g !== "contact_points" && g !== "moment_about_com",
);

export const SCALE_MODE_LABELS: Record<ScaleMode, string> = {
  fixed: "Fixed (Slider)",
  body_weight: "Body Weight",
  peak: "Series Peak",
};

/** Standard gravity used to turn body mass into body weight (N). */
export const STANDARD_GRAVITY_M_S2 = 9.80665;

/** Style keys accepted by the force-overlay API (`ForceOverlayRequest`). */
export interface ForceStyleOptions {
  scale_mode: ScaleMode;
  reference_force_n: number | null;
  reference_length_m: number;
  groups: GlyphGroup[];
}

export interface GlyphSetV1 {
  schema_version: "glyph-set-v1";
  time_s: number;
  arrows: ArrowGlyph[];
  torque_arcs: TorqueArcGlyph[];
  legend: LegendSpec;
}
