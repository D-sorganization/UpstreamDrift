import { useState, useCallback, useEffect, useMemo } from 'react';
import { useURDFModel } from '@/api/useURDFModel';
import { apiFetch } from '@/api/fetch';
import { ModelPreviewViewport } from '@/components/model-explorer/ModelPreviewViewport';
import { ModelTree } from '@/components/model-explorer/ModelTree';
import { PropertyInspector, JointManipulator } from '@/components/model-explorer/InspectorPanel';
import { TreeDiffModal } from '@/components/model-explorer/TreeDiffModal';
import {
  URDFTreeNode,
  copyComponent,
  copyLinkChain,
  swapSubtrees,
  mergeTrees,
  computeTreeDiff,
} from '@/utils/frankensteinTree';

interface ModelEntry {
  name: string;
  format: string;
  path: string;
}

export function ModelExplorerPage() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [frankensteinMode, setFrankensteinMode] = useState(false);

  // Single mode states
  const [selectedModelName, setSelectedModelName] = useState<string | null>(null);
  const [singleTree, setSingleTree] = useState<URDFTreeNode[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Frankenstein mode states
  const [sourceModelName, setSourceModelName] = useState<string | null>(null);
  const [targetModelName, setTargetModelName] = useState<string | null>(null);
  const [sourceTree, setSourceTree] = useState<URDFTreeNode[]>([]);
  const [targetTree, setTargetTree] = useState<URDFTreeNode[]>([]);
  const [selectedSourceNodeId, setSelectedSourceNodeId] = useState<string | null>(null);
  const [selectedTargetNodeId, setSelectedTargetNodeId] = useState<string | null>(null);

  const [jointValues, setJointValues] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDiffOpen, setIsDiffOpen] = useState(false);

  // Fetch model for 3D preview (renders target model in Frankenstein mode if loaded)
  const activePreviewModel = frankensteinMode
    ? (targetModelName || sourceModelName)
    : selectedModelName;
  const { model: urdfModel } = useURDFModel(activePreviewModel);

  // Fetch available models
  useEffect(() => {
    async function fetchModels() {
      try {
        const data = await apiFetch<{ models?: string[] }>('/api/models');
        setModels(data.models || []);
      } catch {
        // Fallback or offline
      }
    }
    fetchModels();
  }, []);

  // Fetch explorer data
  const loadModelData = useCallback(async (modelName: string, type: 'single' | 'source' | 'target') => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch<{ tree?: URDFTreeNode[] }>(
        `/api/tools/model-explorer/${encodeURIComponent(modelName)}`,
      );
      if (type === 'single') {
        setSingleTree(data.tree || []);
        setSelectedNodeId(null);
      } else if (type === 'source') {
        setSourceTree(data.tree || []);
        setSelectedSourceNodeId(null);
      } else if (type === 'target') {
        setTargetTree(data.tree || []);
        setSelectedTargetNodeId(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSingleModel = useCallback((name: string) => {
    setSelectedModelName(name);
    loadModelData(name, 'single');
  }, [loadModelData]);

  const loadSourceModel = useCallback((name: string) => {
    setSourceModelName(name);
    loadModelData(name, 'source');
  }, [loadModelData]);

  const loadTargetModel = useCallback((name: string) => {
    setTargetModelName(name);
    loadModelData(name, 'target');
  }, [loadModelData]);

  // Tree action handlers
  const handleCopyComponent = useCallback((nodeId: string) => {
    const updated = copyComponent(sourceTree, targetTree, nodeId, selectedTargetNodeId);
    setTargetTree(updated);
  }, [sourceTree, targetTree, selectedTargetNodeId]);

  const handleCopyChain = useCallback((nodeId: string) => {
    const updated = copyLinkChain(sourceTree, targetTree, nodeId, selectedTargetNodeId);
    setTargetTree(updated);
  }, [sourceTree, targetTree, selectedTargetNodeId]);

  const handleSwapSubtrees = useCallback(() => {
    if (!selectedSourceNodeId || !selectedTargetNodeId) return;
    const { sourceTree: newSrc, targetTree: newTgt } = swapSubtrees(
      sourceTree,
      targetTree,
      selectedSourceNodeId,
      selectedTargetNodeId
    );
    setSourceTree(newSrc);
    setTargetTree(newTgt);
  }, [sourceTree, targetTree, selectedSourceNodeId, selectedTargetNodeId]);

  const handleMergeAll = useCallback(() => {
    const updated = mergeTrees(sourceTree, targetTree, selectedTargetNodeId);
    setTargetTree(updated);
  }, [sourceTree, targetTree, selectedTargetNodeId]);

  // Compute diff
  const treeDiff = useMemo(() => {
    if (!frankensteinMode || sourceTree.length === 0 || targetTree.length === 0) return null;
    return computeTreeDiff(sourceTree, targetTree);
  }, [frankensteinMode, sourceTree, targetTree]);

  // Active node inspector selection
  const selectedNode = useMemo(() => {
    if (frankensteinMode) {
      if (selectedTargetNodeId) {
        return targetTree.find((n) => n.id === selectedTargetNodeId) ?? null;
      }
      if (selectedSourceNodeId) {
        return sourceTree.find((n) => n.id === selectedSourceNodeId) ?? null;
      }
      return null;
    }
    return singleTree.find((n) => n.id === selectedNodeId) ?? null;
  }, [frankensteinMode, singleTree, sourceTree, targetTree, selectedNodeId, selectedSourceNodeId, selectedTargetNodeId]);

  // Joints for JointManipulator
  const activeTree = frankensteinMode
    ? (targetTree.length > 0 ? targetTree : sourceTree)
    : singleTree;

  const movableJoints = useMemo(() => {
    return activeTree.filter(
      (n) => n.node_type === 'joint' && n.properties.joint_type !== 'fixed'
    );
  }, [activeTree]);

  const handleJointChange = useCallback((name: string, value: number) => {
    setJointValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  const handleResetAll = useCallback(() => {
    const next: Record<string, number> = {};
    for (const joint of movableJoints) {
      next[joint.name] = 0;
    }
    setJointValues(next);
  }, [movableJoints]);

  const handleRandomPose = useCallback(() => {
    const next: Record<string, number> = {};
    for (const joint of movableJoints) {
      const lower = typeof joint.properties.lower === 'number' ? joint.properties.lower : -3.14;
      const upper = typeof joint.properties.upper === 'number' ? joint.properties.upper : 3.14;
      const val = lower + Math.random() * (upper - lower);
      next[joint.name] = val;
    }
    setJointValues(next);
  }, [movableJoints]);

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden text-gray-100">
      {/* Sidebar: dual tree vs single tree */}
      <aside className={`${frankensteinMode ? 'w-[42rem]' : 'w-80'} bg-gray-950 border-r border-gray-800 flex flex-col flex-shrink-0 transition-all duration-300`}>
        {/* Top controls */}
        <div className="p-4 border-b border-gray-850 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-white tracking-tight">Model Explorer</h2>
            <label className="flex items-center gap-2 text-xs font-semibold text-gray-400 hover:text-white cursor-pointer select-none">
              <input
                type="checkbox"
                id="frankenstein-toggle"
                aria-label="Frankenstein Mode"
                checked={frankensteinMode}
                onChange={(e) => setFrankensteinMode(e.target.checked)}
                className="rounded bg-gray-800 border-none text-blue-500 focus:ring-blue-500/50"
              />
              Frankenstein Mode
            </label>
          </div>

          {frankensteinMode && (
            <div className="flex gap-2">
              <button
                onClick={() => setIsDiffOpen(true)}
                disabled={sourceTree.length === 0 || targetTree.length === 0}
                className="flex-1 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded text-xs font-medium transition-colors shadow-lg"
              >
                Compare
              </button>
              {selectedSourceNodeId && selectedTargetNodeId && (
                <button
                  onClick={handleSwapSubtrees}
                  className="flex-1 py-1.5 bg-yellow-600 hover:bg-yellow-500 text-white rounded text-xs font-medium transition-colors shadow-lg"
                  aria-label="Swap Subtrees"
                >
                  Swap Selected
                </button>
              )}
            </div>
          )}
        </div>

        {/* Tree Container */}
        <div className="flex-1 overflow-hidden flex p-4 gap-4 bg-gray-950/50">
          {error && (
            <div className="text-xs text-red-400 bg-red-950/20 border border-red-900/30 p-3 rounded-lg w-full self-start">
              {error}
            </div>
          )}

          {!error && !frankensteinMode && (
            <div className="flex-1 flex flex-col gap-3 min-w-0">
              <div className="space-y-1">
                <label htmlFor="single-select" className="text-[10px] uppercase font-bold text-gray-500 tracking-wider">
                  Select Model
                </label>
                <select
                  id="single-select"
                  value={selectedModelName || ''}
                  onChange={(e) => {
                    if (e.target.value) loadSingleModel(e.target.value);
                  }}
                  className="w-full bg-gray-800 text-gray-200 rounded-lg px-3 py-2 text-xs border border-gray-700 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                >
                  <option value="">Select a model...</option>
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name} ({m.format})
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex-1 min-h-0">
                {loading ? (
                  <div className="text-xs text-gray-500 italic text-center py-12">Loading model structure...</div>
                ) : (
                  <ModelTree
                    modelName={selectedModelName || ''}
                    treeNodes={singleTree}
                    selectedNodeId={selectedNodeId}
                    onNodeSelect={(node) => setSelectedNodeId(node.id)}
                    side="single"
                  />
                )}
              </div>
            </div>
          )}

          {!error && frankensteinMode && (
            <div className="flex-1 flex gap-4 min-w-0 h-full">
              {/* Source Tree Area */}
              <div className="flex-1 flex flex-col gap-3 min-w-0 h-full">
                <div className="space-y-1">
                  <label htmlFor="source-select" className="text-[10px] uppercase font-bold text-gray-500 tracking-wider">
                    Select Source
                  </label>
                  <select
                    id="source-select"
                    aria-label="Select Source"
                    value={sourceModelName || ''}
                    onChange={(e) => {
                      if (e.target.value) loadSourceModel(e.target.value);
                    }}
                    className="w-full bg-gray-805 text-gray-200 rounded-lg px-3 py-2 text-xs border border-gray-700/80 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  >
                    <option value="">Select source model...</option>
                    {models.map((m) => (
                      <option key={m.name} value={m.name}>
                        {m.name} ({m.format})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex-1 min-h-0">
                  <ModelTree
                    modelName={sourceModelName || ''}
                    treeNodes={sourceTree}
                    selectedNodeId={selectedSourceNodeId}
                    onNodeSelect={(node) => setSelectedSourceNodeId(node.id)}
                    side="source"
                    onCopyComponent={handleCopyComponent}
                    onCopyChain={handleCopyChain}
                  />
                </div>
              </div>

              {/* Target Tree Area */}
              <div className="flex-1 flex flex-col gap-3 min-w-0 h-full">
                <div className="space-y-1">
                  <label htmlFor="target-select" className="text-[10px] uppercase font-bold text-gray-500 tracking-wider">
                    Select Target
                  </label>
                  <select
                    id="target-select"
                    aria-label="Select Target"
                    value={targetModelName || ''}
                    onChange={(e) => {
                      if (e.target.value) loadTargetModel(e.target.value);
                    }}
                    className="w-full bg-gray-805 text-gray-200 rounded-lg px-3 py-2 text-xs border border-gray-700/80 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  >
                    <option value="">Select target model...</option>
                    {models.map((m) => (
                      <option key={m.name} value={m.name}>
                        {m.name} ({m.format})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex-1 min-h-0">
                  <ModelTree
                    modelName={targetModelName || ''}
                    treeNodes={targetTree}
                    selectedNodeId={selectedTargetNodeId}
                    onNodeSelect={(node) => setSelectedTargetNodeId(node.id)}
                    side="target"
                    onSelectForSwap={(nodeId) => setSelectedTargetNodeId(nodeId)}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main 3D View */}
      <ModelPreviewViewport
        urdfModel={urdfModel}
        jointValues={jointValues}
        activePreviewModel={activePreviewModel}
      />
      {/* Right Sidebar: Inspector */}
      <aside className="w-72 bg-gray-950 border-l border-gray-800 flex flex-col flex-shrink-0">
        <div className="p-4 border-b border-gray-850">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-3">
            Properties
          </h3>
          <PropertyInspector node={selectedNode} />
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-3">
            Joint Manipulator
          </h3>
          <JointManipulator
            joints={movableJoints}
            jointValues={jointValues}
            onJointChange={handleJointChange}
            onResetAll={handleResetAll}
            onRandomPose={handleRandomPose}
          />
        </div>
      </aside>

      {/* Comparison Diff Modal */}
      <TreeDiffModal
        isOpen={isDiffOpen}
        onClose={() => setIsDiffOpen(false)}
        sourceModelName={sourceModelName || ''}
        targetModelName={targetModelName || ''}
        diff={treeDiff}
        onApplyMergeAll={handleMergeAll}
      />
    </div>
  );
}
