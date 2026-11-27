/**
 * Simulation hook & lifecycle transitions (#14950).
 *
 * Provides the canonical React hook for running, streaming, and controlling
 * physics simulations with a reducer-backed lifecycle.
 */

export { useSimulation } from '../api/client';
export type {
  SimulationPhase,
  SimulationLifecycleState,
  SimulationRunLifecycle,
  SimulationLifecycleAction,
} from '../api/simulationLifecycle';
export {
  simulationLifecycleReducer,
  reduceSimLifecycle,
  initialSimulationLifecycle,
  deriveIsRunning,
  deriveIsPaused,
} from '../api/simulationLifecycle';
