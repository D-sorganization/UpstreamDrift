import { BALL_RADIUS_M, type Vec3 } from './ballGeometry';

interface BallModelProps {
  /** Ball centre in the parent frame, metres. */
  position: Vec3;
  name?: string;
}

/**
 * Decorative golf ball (GCV-13, #11719). Visual only: it takes no part in the
 * simulation. White and slightly rough so it reads as a ball without a
 * dimple texture.
 */
export function BallModel({ position, name = 'golf_ball' }: BallModelProps) {
  return (
    <mesh name={name} position={position}>
      <sphereGeometry args={[BALL_RADIUS_M, 24, 16]} />
      <meshStandardMaterial color="#ffffff" roughness={0.6} />
    </mesh>
  );
}
