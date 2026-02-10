/**
 * API utilities for simulation tools: body positioning and measurement.
 *
 * These are standalone functions that can be called from any component
 * that needs to interact with the simulation tools endpoints.
 *
 * See issue #1179
 */

import type { MeasurementResult } from '@/components/simulation/SimulationToolbar';

/**
 * Set the position of a body in the active simulation.
 *
 * @param bodyName - Name of the body to position
 * @param position - Target position [x, y, z]
 * @returns Applied position response or throws on error
 */
export async function positionBody(
  bodyName: string,
  position: [number, number, number],
): Promise<{ body_name: string; position: number[]; rotation: number[]; status: string }> {
  const response = await fetch('/api/simulation/position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      body_name: bodyName,
      position,
    }),
  });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Measure distance between two bodies in the active simulation.
 *
 * @param bodyA - First body name
 * @param bodyB - Second body name
 * @returns Measurement result with distance and positions
 */
export async function measureDistance(
  bodyA: string,
  bodyB: string,
): Promise<MeasurementResult> {
  const response = await fetch('/api/simulation/measure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ body_a: bodyA, body_b: bodyB }),
  });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `HTTP ${response.status}`);
  }
  return response.json();
}
