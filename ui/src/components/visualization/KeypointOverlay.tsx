export interface ImageLandmark { x: number; y: number; visibility: number | null }

/** Shared image-space overlay. A missing detector visibility stays unknown. */
export function KeypointOverlay({points, coordinates, width, height, showVisibility = false}: {
  points: Record<string, ImageLandmark>; coordinates: 'normalized_image_xy' | 'image_pixels';
  width: number; height: number; showVisibility?: boolean;
}) {
  if (Object.keys(points).length === 0) return null;
  const normalized = coordinates === 'normalized_image_xy';
  return <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 w-full h-full pointer-events-none" data-testid="pose-overlay" aria-label="Image Landmark Observations">
    {Object.entries(points).map(([name, point]) => <circle key={name}
      cx={point.x * (normalized ? width : 1)} cy={point.y * (normalized ? height : 1)} r="4"
      className={showVisibility && point.visibility === null ? 'fill-amber-300 stroke-white' : 'fill-emerald-400 stroke-white'}>
      <title>{name}: {point.visibility === null ? 'Visibility Unknown' : `Visibility ${point.visibility.toFixed(2)}`}</title>
    </circle>)}
  </svg>;
}
