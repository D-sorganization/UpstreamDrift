import {
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react';
import {
  DEFAULT_PARAMS,
  specRequestBody,
  type CharacterSpecRequestBody,
  type SliderParams,
} from './characterSpecTypes';

/**
 * Preset/slider state for the spec-native Character Builder (CMB-7a,
 * #11658), hoisted out of `CharacterSpecPanel` so the appearance export's
 * spec binding (CMB-7d) can read the same `requestBody` without reaching
 * into the panel.
 */
export interface CharacterSpecState {
  selectedPresetId: string;
  setSelectedPresetId: Dispatch<SetStateAction<string>>;
  params: SliderParams;
  setParams: Dispatch<SetStateAction<SliderParams>>;
  requestBody: CharacterSpecRequestBody;
}

export function useCharacterSpec(): CharacterSpecState {
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [params, setParams] = useState<SliderParams>(DEFAULT_PARAMS);

  const requestBody = useMemo(
    () => specRequestBody(selectedPresetId, params),
    [selectedPresetId, params],
  );

  return {
    selectedPresetId,
    setSelectedPresetId,
    params,
    setParams,
    requestBody,
  };
}
