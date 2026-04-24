/**
 * ExpertiseLevelDropdown — small header dropdown for user expertise level.
 *
 * Values drive glossary language selection and are forwarded with chat
 * payloads so the backend can tailor explanations to the user.
 *
 * See issue #3165.
 */

import { useUIStore, type ExpertiseLevel } from '@/stores/useUIStore';

const LEVELS: ExpertiseLevel[] = ['beginner', 'intermediate', 'advanced', 'expert'];

export function ExpertiseLevelDropdown() {
  const level = useUIStore((s) => s.expertiseLevel);
  const setLevel = useUIStore((s) => s.setExpertiseLevel);

  return (
    <label className="inline-flex items-center gap-2 text-xs text-gray-300 bg-gray-800/80 border border-gray-700 rounded-md px-2 py-1 shadow-sm">
      <span className="sr-only">Expertise level</span>
      <span aria-hidden="true" className="text-gray-400">Level:</span>
      <select
        aria-label="Expertise level"
        value={level}
        onChange={(e) => setLevel(e.target.value as ExpertiseLevel)}
        className="bg-transparent text-gray-200 focus:outline-none"
      >
        {LEVELS.map((l) => (
          <option key={l} value={l} className="bg-gray-800">
            {l.charAt(0).toUpperCase() + l.slice(1)}
          </option>
        ))}
      </select>
    </label>
  );
}
