import { TreeDiff } from '@/utils/frankensteinTree';

interface TreeDiffModalProps {
  isOpen: boolean;
  onClose: () => void;
  sourceModelName: string;
  targetModelName: string;
  diff: TreeDiff | null;
  onApplyMergeAll?: () => void;
}

export function TreeDiffModal({
  isOpen,
  onClose,
  sourceModelName,
  targetModelName,
  diff,
  onApplyMergeAll,
}: TreeDiffModalProps) {
  if (!isOpen) return null;

  const noDiff =
    !diff ||
    (diff.added.length === 0 &&
      diff.removed.length === 0 &&
      diff.modified.length === 0);

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
      <div className="bg-gray-900 border border-gray-800 rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="px-6 py-4 bg-gray-950 border-b border-gray-800 flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="text-blue-500 font-mono">📊</span> Model Comparison
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-gray-850"
            aria-label="Close modal"
          >
            ✕
          </button>
        </div>

        {/* Info bar */}
        <div className="px-6 py-3 bg-gray-950/40 border-b border-gray-800/50 flex justify-between text-xs text-gray-400">
          <div>
            Source: <span className="font-mono text-purple-400">{sourceModelName}</span>
          </div>
          <div>
            Target: <span className="font-mono text-blue-400">{targetModelName}</span>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {noDiff ? (
            <div className="text-center py-12 space-y-2">
              <div className="text-4xl">🎉</div>
              <p className="text-sm text-gray-300 font-medium">Models are completely identical!</p>
              <p className="text-xs text-gray-500">No differences detected in structure or properties.</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Added Nodes */}
              {diff && diff.added.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                    <span>🟢</span> Added in Target ({diff.added.length})
                  </h3>
                  <div className="bg-emerald-950/20 border border-emerald-900/30 rounded-lg p-3 space-y-1">
                    {diff.added.map((id) => (
                      <div key={id} className="text-xs text-emerald-300 font-mono">
                        + {id}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Removed Nodes */}
              {diff && diff.removed.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-xs font-bold text-rose-400 uppercase tracking-wider flex items-center gap-2">
                    <span>🔴</span> Removed / Missing in Target ({diff.removed.length})
                  </h3>
                  <div className="bg-rose-950/20 border border-rose-900/30 rounded-lg p-3 space-y-1">
                    {diff.removed.map((id) => (
                      <div key={id} className="text-xs text-rose-300 font-mono">
                        - {id}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Modified Nodes */}
              {diff && diff.modified.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-2">
                    <span>🟡</span> Modified Components ({diff.modified.length})
                  </h3>
                  <div className="space-y-2">
                    {diff.modified.map((mod) => (
                      <div
                        key={mod.id}
                        className="bg-gray-850/50 border border-gray-800 rounded-lg p-3 space-y-2"
                      >
                        <div className="text-xs font-bold text-gray-200 font-mono">
                          ~ {mod.id}
                        </div>
                        <div className="pl-3 space-y-1 border-l border-gray-800">
                          {mod.changes.map((change, i) => (
                            <div
                              key={i}
                              className="text-xs flex flex-wrap items-center gap-1.5 text-gray-400"
                            >
                              <span className="font-medium text-gray-300">{change.field}:</span>
                              <span className="font-mono bg-gray-900/50 px-1 py-0.5 rounded text-rose-400">
                                {change.sourceVal !== undefined ? String(change.sourceVal) : 'none'}
                              </span>
                              <span>→</span>
                              <span className="font-mono bg-gray-900/50 px-1 py-0.5 rounded text-emerald-400">
                                {change.targetVal !== undefined ? String(change.targetVal) : 'none'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-950 border-t border-gray-800 flex items-center justify-between">
          <div className="text-xs text-gray-500">
            {!noDiff && 'Review the tree differences before finalizing your URDF composition.'}
          </div>
          <div className="flex gap-2">
            {onApplyMergeAll && !noDiff && (
              <button
                onClick={() => {
                  onApplyMergeAll();
                  onClose();
                }}
                className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-medium transition-colors"
              >
                Merge All
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-xs font-medium transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
