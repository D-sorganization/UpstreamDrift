/**
 * PuttingGreen - Interactive putting green simulator page.
 *
 * Provides 2D green visualization with putt trajectory rendering,
 * slope contours, and speed/slope/distance controls. Connects to
 * the FastAPI putting green simulation engine.
 *
 * See issue #1206
 */

import { useState, useCallback } from 'react';

/** Putt simulation result from the API. See issue #1206 */
export interface PuttResult {
  positions: number[][];
  velocities: number[][];
  times: number[];
  holed: boolean;
  final_position: number[];
  total_distance: number;
  duration: number;
}

/** Green reading from the API. See issue #1206 */
export interface GreenReading {
  distance: number;
  total_break: number;
  recommended_speed: number;
  aim_point: number[];
  elevations: number[];
  slopes: number[][];
}

/** Scatter analysis result. See issue #1206 */
export interface ScatterResult {
  final_positions: number[][];
  holed_count: number;
  total_simulations: number;
  average_distance_from_hole: number;
  make_percentage: number;
}

/** Green contour data. See issue #1206 */
export interface GreenContour {
  width: number;
  height: number;
  grid_x: number[][];
  grid_y: number[][];
  elevations: number[][];
  hole_position: number[];
}

/**
 * GreenCanvas - 2D putting green visualization with SVG.
 */
function GreenCanvas({
  width,
  height,
  holeX,
  holeY,
  ballX,
  ballY,
  trajectory,
  scatterPoints,
  aimPoint,
}: {
  width: number;
  height: number;
  holeX: number;
  holeY: number;
  ballX: number;
  ballY: number;
  trajectory: number[][] | null;
  scatterPoints: number[][] | null;
  aimPoint: number[] | null;
}) {
  // SVG viewbox maps green coords to pixels
  const svgWidth = 500;
  const svgHeight = 500;
  const scaleX = svgWidth / width;
  const scaleY = svgHeight / height;

  const toSvgX = (x: number) => x * scaleX;
  const toSvgY = (y: number) => svgHeight - y * scaleY; // Flip Y for SVG

  // Compute trajectory path inline (no useMemo needed; parent re-renders on prop changes)
  let trajectoryPath: string | null = null;
  if (trajectory && trajectory.length >= 2) {
    const points = trajectory.map(
      ([x, y]) => `${toSvgX(x).toFixed(1)},${toSvgY(y).toFixed(1)}`,
    );
    trajectoryPath = `M ${points.join(' L ')}`;
  }

  return (
    <svg
      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
      className="w-full h-full"
      data-testid="green-canvas"
    >
      {/* Green surface */}
      <rect
        x={0}
        y={0}
        width={svgWidth}
        height={svgHeight}
        fill="#2d5a27"
        rx={8}
      />

      {/* Grid lines */}
      {Array.from({ length: Math.floor(width) + 1 }, (_, i) => (
        <line
          key={`vl-${i}`}
          x1={toSvgX(i)}
          y1={0}
          x2={toSvgX(i)}
          y2={svgHeight}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={0.5}
        />
      ))}
      {Array.from({ length: Math.floor(height) + 1 }, (_, i) => (
        <line
          key={`hl-${i}`}
          x1={0}
          y1={toSvgY(i)}
          x2={svgWidth}
          y2={toSvgY(i)}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={0.5}
        />
      ))}

      {/* Aim point */}
      {aimPoint && (
        <>
          <line
            x1={toSvgX(ballX)}
            y1={toSvgY(ballY)}
            x2={toSvgX(aimPoint[0])}
            y2={toSvgY(aimPoint[1])}
            stroke="rgba(255,200,50,0.5)"
            strokeWidth={1}
            strokeDasharray="4,4"
          />
          <circle
            cx={toSvgX(aimPoint[0])}
            cy={toSvgY(aimPoint[1])}
            r={4}
            fill="rgba(255,200,50,0.7)"
          />
        </>
      )}

      {/* Scatter points */}
      {scatterPoints?.map((pt, i) => (
        <circle
          key={`scatter-${i}`}
          cx={toSvgX(pt[0])}
          cy={toSvgY(pt[1])}
          r={3}
          fill="rgba(200,100,100,0.6)"
          stroke="rgba(255,150,150,0.4)"
          strokeWidth={0.5}
        />
      ))}

      {/* Trajectory */}
      {trajectoryPath && (
        <path
          d={trajectoryPath}
          fill="none"
          stroke="white"
          strokeWidth={2}
          strokeLinecap="round"
          opacity={0.9}
        />
      )}

      {/* Hole */}
      <circle
        cx={toSvgX(holeX)}
        cy={toSvgY(holeY)}
        r={8}
        fill="#1a1a1a"
        stroke="white"
        strokeWidth={1}
      />
      <circle
        cx={toSvgX(holeX)}
        cy={toSvgY(holeY)}
        r={3}
        fill="#333"
      />

      {/* Ball */}
      <circle
        cx={toSvgX(ballX)}
        cy={toSvgY(ballY)}
        r={6}
        fill="white"
        stroke="rgba(0,0,0,0.3)"
        strokeWidth={1}
        data-testid="ball"
      />

      {/* Final ball position if trajectory exists */}
      {trajectory && trajectory.length > 1 && (
        <circle
          cx={toSvgX(trajectory[trajectory.length - 1][0])}
          cy={toSvgY(trajectory[trajectory.length - 1][1])}
          r={5}
          fill="rgba(255,255,255,0.5)"
          stroke="rgba(255,255,255,0.8)"
          strokeWidth={1}
          strokeDasharray="2,2"
        />
      )}
    </svg>
  );
}

/**
 * PuttingGreenPage - Full putting green simulator tool page.
 *
 * See issue #1206
 */
export function PuttingGreenPage() {
  // Putt parameters
  const [ballX, setBallX] = useState(10.0);
  const [ballY, setBallY] = useState(5.0);
  const [speed, setSpeed] = useState(2.0);
  const [dirX, setDirX] = useState(0.0);
  const [dirY, setDirY] = useState(1.0);
  const [stimpRating, setStimpRating] = useState(10.0);
  const [windSpeed, setWindSpeed] = useState(0.0);

  // Green params
  const [greenWidth] = useState(20.0);
  const [greenHeight] = useState(20.0);
  const [holeX, setHoleX] = useState(10.0);
  const [holeY, setHoleY] = useState(15.0);

  // Results
  const [result, setResult] = useState<PuttResult | null>(null);
  const [reading, setReading] = useState<GreenReading | null>(null);
  const [scatterResult, setScatterResult] = useState<ScatterResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Simulate a putt
  const handleSimulate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setScatterResult(null);

    try {
      const response = await fetch('/api/tools/putting-green/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ball_x: ballX,
          ball_y: ballY,
          speed,
          direction_x: dirX,
          direction_y: dirY,
          stimp_rating: stimpRating,
          green_width: greenWidth,
          green_height: greenHeight,
          hole_x: holeX,
          hole_y: holeY,
          wind_speed: windSpeed,
          wind_direction_x: 1.0,
          wind_direction_y: 0.0,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: PuttResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }, [ballX, ballY, speed, dirX, dirY, stimpRating, greenWidth, greenHeight, holeX, holeY, windSpeed]);

  // Read the green
  const handleReadGreen = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/tools/putting-green/read-green', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ball_x: ballX,
          ball_y: ballY,
          target_x: holeX,
          target_y: holeY,
          green_width: greenWidth,
          green_height: greenHeight,
          stimp_rating: stimpRating,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: GreenReading = await response.json();
      setReading(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Green reading failed');
    } finally {
      setLoading(false);
    }
  }, [ballX, ballY, holeX, holeY, greenWidth, greenHeight, stimpRating]);

  // Scatter analysis
  const handleScatter = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/tools/putting-green/scatter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ball_x: ballX,
          ball_y: ballY,
          speed,
          direction_x: dirX,
          direction_y: dirY,
          n_simulations: 20,
          speed_variance: 0.15,
          direction_variance_deg: 3.0,
          green_width: greenWidth,
          green_height: greenHeight,
          stimp_rating: stimpRating,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: ScatterResult = await response.json();
      setScatterResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scatter analysis failed');
    } finally {
      setLoading(false);
    }
  }, [ballX, ballY, speed, dirX, dirY, greenWidth, greenHeight, stimpRating]);

  const trajectory = result?.positions ?? null;
  const aimPoint = reading?.aim_point ?? null;
  const scatterPoints = scatterResult?.final_positions ?? null;

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Panel: Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white mb-1">Putting Green Simulator</h2>
          <p className="text-xs text-gray-500">Physics-based putting simulation</p>
        </div>

        {/* Stroke Controls */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Stroke Parameters
          </h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Speed: {speed.toFixed(1)} m/s
            </label>
            <input
              type="range"
              min={0.5}
              max={8.0}
              step={0.1}
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              aria-label="Stroke speed"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Dir X: {dirX.toFixed(2)}
              </label>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.01}
                value={dirX}
                onChange={(e) => setDirX(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                aria-label="Direction X"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Dir Y: {dirY.toFixed(2)}
              </label>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.01}
                value={dirY}
                onChange={(e) => setDirY(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                aria-label="Direction Y"
              />
            </div>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Stimp Rating: {stimpRating.toFixed(1)} ft
            </label>
            <input
              type="range"
              min={6.0}
              max={15.0}
              step={0.5}
              value={stimpRating}
              onChange={(e) => setStimpRating(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              aria-label="Stimp rating"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Wind: {windSpeed.toFixed(1)} m/s
            </label>
            <input
              type="range"
              min={0}
              max={5}
              step={0.1}
              value={windSpeed}
              onChange={(e) => setWindSpeed(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              aria-label="Wind speed"
            />
          </div>
        </div>

        {/* Position Controls */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Positions
          </h3>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Ball X: {ballX.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.5}
                max={greenWidth - 0.5}
                step={0.5}
                value={ballX}
                onChange={(e) => setBallX(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                aria-label="Ball X position"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Ball Y: {ballY.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.5}
                max={greenHeight - 0.5}
                step={0.5}
                value={ballY}
                onChange={(e) => setBallY(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                aria-label="Ball Y position"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Hole X: {holeX.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.5}
                max={greenWidth - 0.5}
                step={0.5}
                value={holeX}
                onChange={(e) => setHoleX(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                aria-label="Hole X position"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Hole Y: {holeY.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.5}
                max={greenHeight - 0.5}
                step={0.5}
                value={holeY}
                onChange={(e) => setHoleY(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                aria-label="Hole Y position"
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="p-4 space-y-2">
          <button
            onClick={handleSimulate}
            disabled={loading}
            className="w-full py-2 px-4 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
            data-testid="simulate-btn"
          >
            {loading ? 'Simulating...' : 'Simulate Putt'}
          </button>
          <button
            onClick={handleReadGreen}
            disabled={loading}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
          >
            Read Green
          </button>
          <button
            onClick={handleScatter}
            disabled={loading}
            className="w-full py-2 px-4 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 text-white text-sm font-medium rounded transition-colors"
          >
            Scatter Analysis
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-4 mb-4 text-xs text-red-400 bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}
      </aside>

      {/* Center: Green Visualization */}
      <main className="flex-1 flex items-center justify-center bg-gray-950 relative min-w-0">
        <div className="w-full max-w-[600px] aspect-square p-4">
          <GreenCanvas
            width={greenWidth}
            height={greenHeight}
            holeX={holeX}
            holeY={holeY}
            ballX={ballX}
            ballY={ballY}
            trajectory={trajectory}
            scatterPoints={scatterPoints}
            aimPoint={aimPoint}
          />
        </div>

        {/* Result overlay */}
        {result && (
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg border border-white/10">
            <div className="text-sm text-gray-200 space-y-1">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${result.holed ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="font-medium">{result.holed ? 'Holed!' : 'Missed'}</span>
              </div>
              <div className="text-xs text-gray-400">
                Distance: {result.total_distance.toFixed(2)} m
              </div>
              <div className="text-xs text-gray-400">
                Duration: {result.duration.toFixed(2)} s
              </div>
            </div>
          </div>
        )}

        {/* Scatter overlay */}
        {scatterResult && (
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg border border-white/10">
            <div className="text-sm text-gray-200 space-y-1">
              <div className="font-medium">Scatter Analysis</div>
              <div className="text-xs text-gray-400">
                Make rate: {scatterResult.make_percentage.toFixed(1)}%
                ({scatterResult.holed_count}/{scatterResult.total_simulations})
              </div>
              <div className="text-xs text-gray-400">
                Avg miss: {scatterResult.average_distance_from_hole.toFixed(2)} m
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Right Panel: Green Reading */}
      <aside className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col flex-shrink-0 overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Green Reading
          </h3>

          {reading ? (
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Distance</span>
                <span className="text-gray-200 font-mono">
                  {reading.distance.toFixed(2)} m
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Total Break</span>
                <span className="text-gray-200 font-mono">
                  {reading.total_break.toFixed(3)} m
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Recommended Speed</span>
                <span className="text-blue-400 font-mono">
                  {reading.recommended_speed.toFixed(2)} m/s
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Aim Point</span>
                <span className="text-yellow-400 font-mono">
                  ({reading.aim_point[0].toFixed(1)}, {reading.aim_point[1].toFixed(1)})
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500 italic text-center py-4">
              Click &quot;Read Green&quot; to analyze the putt line
            </div>
          )}
        </div>

        {/* Putt Result Details */}
        <div className="p-4 flex-1">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Last Putt
          </h3>

          {result ? (
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Result</span>
                <span className={result.holed ? 'text-green-400' : 'text-red-400'}>
                  {result.holed ? 'HOLED' : 'MISSED'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Final Position</span>
                <span className="text-gray-200 font-mono">
                  ({result.final_position[0].toFixed(1)}, {result.final_position[1].toFixed(1)})
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Total Distance</span>
                <span className="text-gray-200 font-mono">
                  {result.total_distance.toFixed(2)} m
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Duration</span>
                <span className="text-gray-200 font-mono">
                  {result.duration.toFixed(2)} s
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Trajectory Points</span>
                <span className="text-gray-200 font-mono">
                  {result.positions.length}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500 italic text-center py-4">
              Simulate a putt to see results
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
