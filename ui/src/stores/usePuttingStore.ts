import { create } from "zustand";

import type {
  Putt3DSimulationRequest,
  Putt3DSimulationResponse,
} from "@/api/generated/types";

export const DEFAULT_PUTT_3D_PARAMETERS: Putt3DSimulationRequest = {
  putter_speed_mps: 1.8,
  loft_deg: 3,
  head_mass_kg: 0.35,
  head_moi_kg_m2: 4.5e-4,
  coefficient_of_restitution: 0.78,
  hosel_toe_m: 0,
  hosel_forward_m: 0,
  impact_toe_m: 0,
  stimp_rating: 10,
  grade_percent: 0,
  downhill_aspect_deg: 0,
  grain_strength: 0,
  grain_direction_deg: 0,
  rolling_velocity_coefficient: 0,
  bump_height_m: 0,
  friction_variation: 0,
  random_seed: 8345,
  hole_x_m: 3,
  hole_y_m: 0,
};

interface PuttingState {
  parameters: Putt3DSimulationRequest;
  result: Putt3DSimulationResponse | null;
  playbackTimeS: number;
  playbackRate: number;
  playing: boolean;
  updateParameters: (changes: Partial<Putt3DSimulationRequest>) => void;
  setResult: (result: Putt3DSimulationResponse) => void;
  setPlaybackTime: (timeS: number) => void;
  setPlaybackRate: (rate: number) => void;
  setPlaying: (playing: boolean) => void;
  reset: () => void;
}

export const usePuttingStore = create<PuttingState>((set) => ({
  parameters: { ...DEFAULT_PUTT_3D_PARAMETERS },
  result: null,
  playbackTimeS: 0,
  playbackRate: 1,
  playing: false,
  updateParameters: (changes) =>
    set((state) => ({
      parameters: { ...state.parameters, ...changes },
      playbackTimeS: 0,
      playing: false,
    })),
  setResult: (result) => set({ result, playbackTimeS: 0, playing: false }),
  setPlaybackTime: (playbackTimeS) => set({ playbackTimeS }),
  setPlaybackRate: (playbackRate) => set({ playbackRate }),
  setPlaying: (playing) => set({ playing }),
  reset: () =>
    set({
      parameters: { ...DEFAULT_PUTT_3D_PARAMETERS },
      result: null,
      playbackTimeS: 0,
      playbackRate: 1,
      playing: false,
    }),
}));
