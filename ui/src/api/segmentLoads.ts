/** Producer-owned axial loads at declared proximal segment sections, SI units. */
export interface SegmentForceFrame {
  time_s: number;
  source: string;
  units: 'N';
  sign_convention: 'tension-positive';
  values_n: Record<string, number | null>;
}
