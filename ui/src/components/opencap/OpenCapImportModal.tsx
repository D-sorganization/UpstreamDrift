/**
 * OpenCapImportModal - Modal for selecting and importing an OpenCap session trial (#11409).
 *
 * Lists the session's trials, displays subject and model metadata,
 * allows user selection, and dispatches the import event to the OpenSim target.
 */

import React, { useState, useMemo } from 'react';

export interface OpenCapSessionSubject {
  mass_kg?: number | null;
  height_m?: number | null;
  sex?: string | null;
  opensim_model?: string | null;
  subject_id?: string | null;
}

export interface OpenCapSessionMetadata {
  session_dir?: string;
  trials: string[];
  subject?: OpenCapSessionSubject | null;
  model_file?: string | null;
  kinematics_trials?: string[];
  notes?: string[];
}

export interface OpenCapImportEvent {
  sessionDir: string;
  selectedTrial: string;
  metadata?: OpenCapSessionMetadata | null;
}

export interface OpenCapImportModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionDir?: string;
  initialTrials?: string[];
  metadata?: OpenCapSessionMetadata | null;
  error?: string | null;
  onImport: (event: OpenCapImportEvent) => void | Promise<void>;
}

export const OpenCapImportModal: React.FC<OpenCapImportModalProps> = ({
  isOpen,
  onClose,
  sessionDir = '',
  initialTrials = [],
  metadata = null,
  error = null,
  onImport,
}) => {
  const trials = useMemo(() => {
    if (metadata?.trials && metadata.trials.length > 0) {
      return metadata.trials;
    }
    return initialTrials;
  }, [metadata, initialTrials]);

  const [userSelectedTrial, setUserSelectedTrial] = useState<string | null>(null);

  const currentDir = sessionDir || metadata?.session_dir || '';
  const defaultTrial = trials.find((t) => t.toLowerCase() !== 'neutral') || trials[0] || '';
  const selectedTrial = userSelectedTrial ?? defaultTrial;

  if (!isOpen) {
    return null;
  }

  const handleImportClick = () => {
    if (!selectedTrial) return;
    onImport({
      sessionDir: currentDir,
      selectedTrial,
      metadata,
    });
  };

  const subject = metadata?.subject;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="opencap-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    >
      <div className="w-full max-w-lg rounded-lg border border-gray-700 bg-gray-900 p-6 shadow-xl text-gray-100 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-800 pb-3">
          <h2 id="opencap-modal-title" className="text-lg font-bold text-gray-100">
            Import OpenCap Session
          </h2>
          <button
            type="button"
            aria-label="Close modal"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Error message */}
        {error && (
          <div
            role="alert"
            className="rounded border border-red-500/50 bg-red-950/40 p-3 text-xs text-red-200"
          >
            {error}
          </div>
        )}

        {/* Session details */}
        {currentDir && (
          <div className="text-xs text-gray-400 truncate">
            <span className="font-semibold text-gray-300">Session Directory: </span>
            {currentDir}
          </div>
        )}

        {/* Subject & Model Metadata */}
        {subject && (
          <div className="rounded bg-gray-800/60 p-2.5 text-xs text-gray-300 space-y-1">
            <div className="font-semibold text-gray-200">Subject Anthropometry</div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-gray-400">
              {subject.mass_kg != null && <span>Mass: {subject.mass_kg} kg</span>}
              {subject.height_m != null && <span>Height: {subject.height_m} m</span>}
              {subject.opensim_model && <span>Model: {subject.opensim_model}</span>}
              {subject.sex && <span>Sex: {subject.sex}</span>}
            </div>
            {metadata?.model_file && (
              <div className="text-gray-400 truncate mt-1">
                <span className="font-medium text-gray-300">Scaled Model: </span>
                {metadata.model_file}
              </div>
            )}
          </div>
        )}

        {/* Trial selection list */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400">
            Select Trial to Import
          </label>
          {trials.length === 0 ? (
            <div className="text-xs italic text-gray-500 py-3 text-center">
              No trials available in session.
            </div>
          ) : (
            <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
              {trials.map((trial) => {
                const isSelected = selectedTrial === trial;
                const hasKinematics = metadata?.kinematics_trials?.includes(trial);
                return (
                  <button
                    key={trial}
                    type="button"
                    role="button"
                    aria-pressed={isSelected}
                    onClick={() => setUserSelectedTrial(trial)}
                    className={`w-full flex items-center justify-between rounded px-3 py-2 text-left text-xs transition-colors ${
                      isSelected
                        ? 'bg-blue-600/30 border border-blue-500 text-blue-100 font-medium'
                        : 'bg-gray-800/40 hover:bg-gray-800 text-gray-300 border border-transparent'
                    }`}
                  >
                    <span>{trial}</span>
                    <div className="flex items-center gap-2 text-[10px]">
                      {trial.toLowerCase() === 'neutral' && (
                        <span className="rounded bg-gray-700 px-1.5 py-0.5 text-gray-300">
                          static
                        </span>
                      )}
                      {hasKinematics && (
                        <span className="rounded bg-emerald-900/60 text-emerald-300 border border-emerald-700/50 px-1.5 py-0.5">
                          IK ready
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-3 border-t border-gray-800 pt-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!selectedTrial}
            onClick={handleImportClick}
            className="rounded bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Import Session
          </button>
        </div>
      </div>
    </div>
  );
};
