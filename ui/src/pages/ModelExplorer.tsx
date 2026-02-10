/**
 * ModelExplorer - URDF model browser and inspector page.
 *
 * Provides a tree view of URDF model structure, property inspector
 * for selected nodes, joint manipulator sliders, and Frankenstein
 * mode for side-by-side model comparison.
 *
 * See issue #1200
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { URDFViewer } from '@/components/visualization/URDFViewer';
import { useURDFModel } from '@/api/useURDFModel';

/** Tree node from the model explorer API. See issue #1200 */
interface URDFTreeNode {
  id: string;
  name: string;
  node_type: 'link' | 'joint' | 'root';
  parent_id: string | null;
  children: string[];
  properties: Record<string, unknown>;
}

/** Model explorer data from the API. See issue #1200 */
interface ModelExplorerData {
  model_name: string;
  tree: URDFTreeNode[];
  joint_count: number;
  link_count: number;
  model_format: string;
  file_path: string;
}

/** Available model entry. */
interface ModelEntry {
  name: string;
  format: string;
  path: string;
}

/**
 * TreeNodeComponent - Recursive tree node renderer.
 */
function TreeNodeComponent({
  node,
  allNodes,
  selectedId,
  onSelect,
  depth = 0,
}: {
  node: URDFTreeNode;
  allNodes: Map<string, URDFTreeNode>;
  selectedId: string | null;
  onSelect: (node: URDFTreeNode) => void;
  depth?: number;
}) {
  const [expanded, setExpanded] = useState(depth < 2);
  const isSelected = selectedId === node.id;

  const childNodes = useMemo(
    () =>
      node.children
        .map((childId) => allNodes.get(childId))
        .filter((n): n is URDFTreeNode => n != null),
    [node.children, allNodes],
  );

  const iconColor: string =
    node.node_type === 'root'
      ? 'text-purple-400'
      : node.node_type === 'joint'
        ? 'text-yellow-400'
        : 'text-blue-400';

  const icon: string =
    node.node_type === 'root'
      ? 'R'
      : node.node_type === 'joint'
        ? 'J'
        : 'L';

  return (
    <div>
      <div
        className={`flex items-center gap-1 py-0.5 px-1 rounded cursor-pointer hover:bg-gray-600/50 ${
          isSelected ? 'bg-blue-900/40 ring-1 ring-blue-500/50' : ''
        }`}
        style={{ paddingLeft: `${depth * 16 + 4}px` }}
        onClick={() => onSelect(node)}
        role="treeitem"
        aria-selected={isSelected}
        aria-expanded={childNodes.length > 0 ? expanded : undefined}
      >
        {/* Expand/collapse toggle */}
        {childNodes.length > 0 ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
            className="text-xs text-gray-500 w-3 flex-shrink-0"
            aria-label={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? '-' : '+'}
          </button>
        ) : (
          <span className="w-3 flex-shrink-0" />
        )}

        {/* Node icon */}
        <span className={`text-xs font-bold ${iconColor} w-4 flex-shrink-0`}>
          {icon}
        </span>

        {/* Node name */}
        <span className="text-xs text-gray-300 truncate">{node.name}</span>

        {/* Joint type badge */}
        {node.node_type === 'joint' && node.properties.joint_type != null && (
          <span className="text-xs text-gray-500 ml-auto flex-shrink-0">
            {String(node.properties.joint_type)}
          </span>
        )}
      </div>

      {/* Children */}
      {expanded &&
        childNodes.map((child) => (
          <TreeNodeComponent
            key={child.id}
            node={child}
            allNodes={allNodes}
            selectedId={selectedId}
            onSelect={onSelect}
            depth={depth + 1}
          />
        ))}
    </div>
  );
}

/**
 * PropertyInspector - Shows properties of the selected tree node.
 */
function PropertyInspector({ node }: { node: URDFTreeNode | null }) {
  if (!node) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-4">
        Select a node to inspect properties
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-gray-200">{node.name}</span>
        <span className="text-xs bg-gray-600 px-1.5 py-0.5 rounded text-gray-300">
          {node.node_type}
        </span>
      </div>

      <div className="space-y-1">
        {Object.entries(node.properties).map(([key, value]) => (
          <div key={key} className="flex justify-between text-xs">
            <span className="text-gray-400">{key}</span>
            <span className="text-gray-300 font-mono ml-2 truncate max-w-[150px]">
              {typeof value === 'number'
                ? value.toFixed(4)
                : String(value)}
            </span>
          </div>
        ))}
      </div>

      {node.parent_id && (
        <div className="text-xs text-gray-500 border-t border-gray-600 pt-1">
          Parent: {node.parent_id}
        </div>
      )}
      {node.children.length > 0 && (
        <div className="text-xs text-gray-500">
          Children: {node.children.join(', ')}
        </div>
      )}
    </div>
  );
}

/**
 * JointManipulator - Sliders for adjusting joint angles in the preview.
 */
function JointManipulator({
  joints,
  jointValues,
  onJointChange,
}: {
  joints: URDFTreeNode[];
  jointValues: Record<string, number>;
  onJointChange: (name: string, value: number) => void;
}) {
  if (joints.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-2">
        No movable joints
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {joints.map((joint) => {
        const lower = typeof joint.properties.lower === 'number' ? joint.properties.lower : -3.14;
        const upper = typeof joint.properties.upper === 'number' ? joint.properties.upper : 3.14;
        const value = jointValues[joint.name] ?? 0;

        return (
          <div key={joint.id} className="bg-gray-700/30 p-1.5 rounded">
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-xs text-gray-300 truncate max-w-[120px]">
                {joint.name}
              </span>
              <span className="text-xs font-mono text-blue-400">
                {value.toFixed(2)} rad
              </span>
            </div>
            <input
              type="range"
              min={lower}
              max={upper}
              step={0.01}
              value={value}
              onChange={(e) =>
                onJointChange(joint.name, parseFloat(e.target.value))
              }
              className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              aria-label={`${joint.name} angle`}
            />
          </div>
        );
      })}
    </div>
  );
}

/**
 * ModelExplorerPage - Full model explorer with tree view, 3D preview,
 * property inspector, and joint manipulator.
 *
 * See issue #1200
 */
export function ModelExplorerPage() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [selectedModelName, setSelectedModelName] = useState<string | null>(null);
  const [explorerData, setExplorerData] = useState<ModelExplorerData | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [jointValues, setJointValues] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch URDF model for 3D preview
  const { model: urdfModel } = useURDFModel(selectedModelName);

  // Fetch available models
  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch('/api/models');
        if (!response.ok) return;
        const data = await response.json();
        setModels(data.models || []);
      } catch {
        // Models endpoint may not be available
      }
    }
    fetchModels();
  }, []);

  // Fetch explorer data for selected model
  const loadModel = useCallback(async (modelName: string) => {
    setLoading(true);
    setError(null);
    setSelectedModelName(modelName);
    setSelectedNodeId(null);
    setJointValues({});

    try {
      const response = await fetch(
        `/api/tools/model-explorer/${encodeURIComponent(modelName)}`,
      );
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data: ModelExplorerData = await response.json();
      setExplorerData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
      setExplorerData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Build node map for tree navigation
  const nodeMap = useMemo(() => {
    const map = new Map<string, URDFTreeNode>();
    if (explorerData) {
      for (const node of explorerData.tree) {
        map.set(node.id, node);
      }
    }
    return map;
  }, [explorerData]);

  // Find root nodes (no parent)
  const rootNodes = useMemo(() => {
    if (!explorerData) return [];
    return explorerData.tree.filter(
      (n) => n.parent_id === null || n.node_type === 'root',
    );
  }, [explorerData]);

  // Get movable joints for the manipulator
  const movableJoints = useMemo(() => {
    if (!explorerData) return [];
    return explorerData.tree.filter(
      (n) =>
        n.node_type === 'joint' &&
        n.properties.joint_type !== 'fixed',
    );
  }, [explorerData]);

  // Selected node
  const selectedNode = selectedNodeId ? nodeMap.get(selectedNodeId) ?? null : null;

  const handleNodeSelect = useCallback((node: URDFTreeNode) => {
    setSelectedNodeId(node.id);
  }, []);

  const handleJointChange = useCallback((name: string, value: number) => {
    setJointValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Panel: Model selector + Tree */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0">
        {/* Model Selector */}
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white mb-2">Model Explorer</h2>
          <select
            value={selectedModelName || ''}
            onChange={(e) => {
              if (e.target.value) loadModel(e.target.value);
            }}
            className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1.5 text-sm border-none focus:ring-1 focus:ring-blue-400"
          >
            <option value="">Select a model...</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name} ({m.format})
              </option>
            ))}
          </select>

          {explorerData && (
            <div className="mt-2 text-xs text-gray-400 space-y-0.5">
              <div>Model: {explorerData.model_name}</div>
              <div>
                {explorerData.link_count} links, {explorerData.joint_count} joints
              </div>
              <div className="text-gray-500 truncate">{explorerData.file_path}</div>
            </div>
          )}
        </div>

        {/* Tree View */}
        <div className="flex-1 overflow-y-auto p-2" role="tree">
          {loading && (
            <div className="text-xs text-gray-400 text-center py-4">
              Loading model...
            </div>
          )}
          {error && (
            <div className="text-xs text-red-400 bg-red-900/20 p-2 rounded">
              {error}
            </div>
          )}
          {!loading && !error && rootNodes.length > 0 && (
            rootNodes.map((node) => (
              <TreeNodeComponent
                key={node.id}
                node={node}
                allNodes={nodeMap}
                selectedId={selectedNodeId}
                onSelect={handleNodeSelect}
              />
            ))
          )}
          {!loading && !error && !explorerData && (
            <div className="text-xs text-gray-500 italic text-center py-4">
              Select a model to view its structure
            </div>
          )}
        </div>
      </aside>

      {/* Center: 3D Preview */}
      <main className="flex-1 relative min-w-0">
        <Canvas
          camera={{ position: [3, 2, 3], fov: 50 }}
          className="bg-gray-950 w-full h-full"
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          <Grid
            infiniteGrid
            cellSize={0.5}
            cellThickness={0.5}
            sectionSize={2}
            sectionThickness={1}
            fadeDistance={30}
          />

          {urdfModel && (
            <URDFViewer
              model={urdfModel}
              jointAngles={jointValues}
              showAxes={true}
            />
          )}

          <axesHelper args={[1]} />
          <Environment preset="studio" />
        </Canvas>

        {/* Model name overlay */}
        {explorerData && (
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-white/10">
            <span className="text-sm text-gray-200 font-mono">
              {explorerData.model_name}
            </span>
          </div>
        )}
      </main>

      {/* Right Panel: Inspector + Joint Manipulator */}
      <aside className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col flex-shrink-0">
        {/* Property Inspector */}
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Properties
          </h3>
          <PropertyInspector node={selectedNode} />
        </div>

        {/* Joint Manipulator */}
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Joint Manipulator
          </h3>
          <JointManipulator
            joints={movableJoints}
            jointValues={jointValues}
            onJointChange={handleJointChange}
          />
        </div>
      </aside>
    </div>
  );
}
