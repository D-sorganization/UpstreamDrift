/**
 * ForceOverlay - Force and torque vector overlay visualization using Three.js (ADR-0052, #11308).
 *
 * Renders server-serialized GlyphSet (glyph-set-v1) specifications directly into the Three.js scene graph.
 * Pure geometric mapping with no physics and no client-side scaling/clamping maths.
 */

import { useMemo } from "react";
import type { GlyphGroup, GlyphSetV1, ScaleMode } from "@/types/glyphs";
import { buildForceOverlayScene } from "./forceOverlayScene";

/** Legacy overlay configuration interface */
export interface ForceOverlayConfig {
  enabled: boolean;
  forceTypes: string[];
  scaleFactor: number;
  colorByMagnitude: boolean;
  showLabels: boolean;
  bodyFilter: string[] | null;
  /** Arrow scaling mode (GCV-4, #11710); `fixed` uses `scaleFactor`. */
  scaleMode: ScaleMode;
  /** Body mass in kg; 1 body weight is the `body_weight` reference. */
  bodyMassKg: number;
  /** Series peak force in N; the `peak` reference. */
  peakForceN: number;
  /** Arrow length in metres for one reference force. */
  referenceLengthM: number;
  groups: GlyphGroup[];
}

export interface ForceOverlayProps {
  /** Serialized GlyphSet from WebSocket frame or REST endpoint */
  glyphs?: GlyphSetV1 | null;
}

/**
 * ForceOverlay component rendering the serialized GlyphSet into React Three Fiber.
 */
export function ForceOverlay({ glyphs }: ForceOverlayProps) {
  const scene = useMemo(() => {
    return buildForceOverlayScene(glyphs);
  }, [glyphs]);

  if (!scene) {
    return null;
  }

  return <primitive object={scene} />;
}
