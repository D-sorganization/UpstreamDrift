/**
 * Golf ball placement at address for the web golfer scene (GCV-13, #11719).
 *
 * Mirrors `ball_position_at_address` in
 * `src/shared/python/model_appearance/ball.py` (Z up there, Y up here): the
 * centre is the face-centre contact point offset by one ball radius along the
 * unit face normal, lowered or raised so the ball rests on the ground. Pure
 * data only so the maths is testable without three.js.
 */

export type Vec3 = [number, number, number];

/** Regulation ball diameter 42.67 mm, radius 21.335 mm. */
export const BALL_RADIUS_M = 0.021335;

/**
 * World position of the ball centre at address.
 *
 * Preconditions: finite 3-vectors, nonzero normal, positive radius.
 * Postcondition: the returned y equals `groundY + radiusM`.
 */
export function ballPositionAtAddress(
  faceCentre: Vec3,
  faceNormal: Vec3,
  groundY: number,
  radiusM: number = BALL_RADIUS_M,
): Vec3 {
  if (![...faceCentre, ...faceNormal, groundY].every(Number.isFinite)) {
    throw new Error('face centre, face normal and ground height must be finite');
  }
  if (!Number.isFinite(radiusM) || radiusM <= 0) throw new Error('radius must be positive');
  const len = Math.hypot(...faceNormal);
  if (len < 1e-12) throw new Error('face normal must be nonzero');
  return [
    faceCentre[0] + (radiusM * faceNormal[0]) / len,
    groundY + radiusM,
    faceCentre[2] + (radiusM * faceNormal[2]) / len,
  ];
}
