/**
 * Skeleton-frame helpers shared by the Motion Capture views (issue #8406).
 *
 * Kept outside the page module so component files export only components
 * (react-refresh) and so both the 2D SVG and 3D R3F views can share them.
 */

/**
 * Joint data for skeleton rendering. Structurally identical to `JointData`
 * in `ui/src/pages/MotionCapture.tsx` and `MocapJoint` in
 * `ui/src/components/visualization/MocapSkeleton3D.tsx`.
 */
export interface SkeletonJointData {
  name: string;
  /** [x, y, z] position; z may be omitted for 2D sources. */
  position: number[];
  /** Detection confidence in [0, 1]. */
  confidence: number;
  parent: string | null;
}

/**
 * Extract a skeleton joint list from a live `pose/canonical` payload.
 *
 * The realtime channel is shared by several publishers; only payloads that
 * actually carry joint positions (a `joints` array of `{name, position}`
 * objects, or a bare array of them) can drive the skeleton views. Anything
 * else (e.g. Pose Studio's joint-angle dict) returns null so the UI falls
 * back to the polled frame data.
 *
 * @param payload - Parsed realtime message of unknown shape.
 * @returns Normalized joints, or null when the payload is not renderable.
 */
export function extractSkeletonJoints(
  payload: unknown,
): SkeletonJointData[] | null {
  const candidates = Array.isArray(payload)
    ? payload
    : payload !== null &&
        typeof payload === 'object' &&
        Array.isArray((payload as { joints?: unknown }).joints)
      ? (payload as { joints: unknown[] }).joints
      : null;
  if (!candidates || candidates.length === 0) return null;

  const joints: SkeletonJointData[] = [];
  for (const item of candidates) {
    if (item === null || typeof item !== 'object') return null;
    const j = item as Record<string, unknown>;
    if (typeof j.name !== 'string' || !Array.isArray(j.position)) return null;
    if (!j.position.every((v) => typeof v === 'number')) return null;
    joints.push({
      name: j.name,
      position: j.position as number[],
      confidence: typeof j.confidence === 'number' ? j.confidence : 1.0,
      parent: typeof j.parent === 'string' ? j.parent : null,
    });
  }
  return joints;
}

/** True when any joint carries a meaningful depth (z) component. */
export function frameHasDepth(joints: SkeletonJointData[]): boolean {
  return joints.some((j) => Math.abs(j.position[2] ?? 0) > 1e-6);
}
