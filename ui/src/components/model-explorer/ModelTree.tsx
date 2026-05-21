import { useState, useMemo } from 'react';
import { URDFTreeNode } from '@/utils/frankensteinTree';

interface ModelTreeProps {
  modelName: string;
  treeNodes: URDFTreeNode[];
  selectedNodeId: string | null;
  onNodeSelect: (node: URDFTreeNode) => void;
  side: 'source' | 'target' | 'single';
  onCopyComponent?: (nodeId: string) => void;
  onCopyChain?: (nodeId: string) => void;
  onSelectForSwap?: (nodeId: string) => void;
  activeSwapSelection?: { side: 'source' | 'target'; nodeId: string } | null;
}

function TreeNodeComponent({
  node,
  allNodes,
  selectedId,
  onSelect,
  depth = 0,
  side,
  onCopyComponent,
  onCopyChain,
  onSelectForSwap,
  activeSwapSelection,
}: {
  node: URDFTreeNode;
  allNodes: Map<string, URDFTreeNode>;
  selectedId: string | null;
  onSelect: (node: URDFTreeNode) => void;
  depth?: number;
  side: 'source' | 'target' | 'single';
  onCopyComponent?: (nodeId: string) => void;
  onCopyChain?: (nodeId: string) => void;
  onSelectForSwap?: (nodeId: string) => void;
  activeSwapSelection?: { side: 'source' | 'target'; nodeId: string } | null;
}) {
  const [expanded, setExpanded] = useState(depth < 2);
  const isSelected = selectedId === node.id;
  const isSelectedForSwap = activeSwapSelection?.nodeId === node.id && activeSwapSelection?.side === side;

  const childNodes = useMemo(
    () =>
      node.children
        .map((childId) => allNodes.get(childId))
        .filter((n): n is URDFTreeNode => n != null),
    [node.children, allNodes],
  );

  const iconColor =
    node.node_type === 'root'
      ? 'text-purple-400'
      : node.node_type === 'joint'
        ? 'text-yellow-400'
        : 'text-blue-400';

  const icon =
    node.node_type === 'root'
      ? 'R'
      : node.node_type === 'joint'
        ? 'J'
        : 'L';

  return (
    <div>
      <div
        className={`group flex items-center gap-1 py-1 px-2 rounded cursor-pointer hover:bg-gray-700/50 transition-colors ${
          isSelected ? 'bg-blue-900/30 ring-1 ring-blue-500/50' : ''
        } ${isSelectedForSwap ? 'bg-yellow-900/30 ring-1 ring-yellow-500/50' : ''}`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
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
            className="text-xs text-gray-500 hover:text-gray-300 w-3 flex-shrink-0"
            aria-label={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? '▼' : '▶'}
          </button>
        ) : (
          <span className="w-3 flex-shrink-0" />
        )}

        {/* Node icon */}
        <span className={`text-[10px] font-bold ${iconColor} w-4 h-4 rounded bg-gray-900/50 flex items-center justify-center flex-shrink-0`}>
          {icon}
        </span>

        {/* Node name */}
        <span className="text-xs text-gray-300 truncate font-mono">{node.name}</span>

        {/* Joint type badge */}
        {node.node_type === 'joint' && node.properties.joint_type != null && (
          <span className="text-[10px] text-gray-500 ml-1">
            ({String(node.properties.joint_type)})
          </span>
        )}

        {/* Frankenstein Actions for Selected Node */}
        {isSelected && side === 'source' && (
          <div className="ml-auto flex items-center gap-1 opacity-90 group-hover:opacity-100">
            {onCopyComponent && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onCopyComponent(node.id);
                }}
                className="text-[10px] bg-blue-600 hover:bg-blue-500 text-white px-1.5 py-0.5 rounded transition-colors"
                aria-label="Copy Component"
                title="Copy single component to target"
              >
                Copy
              </button>
            )}
            {onCopyChain && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onCopyChain(node.id);
                }}
                className="text-[10px] bg-purple-600 hover:bg-purple-500 text-white px-1.5 py-0.5 rounded transition-colors"
                aria-label="Copy Chain"
                title="Copy recursively to target"
              >
                Chain
              </button>
            )}
            {onSelectForSwap && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectForSwap(node.id);
                }}
                className={`text-[10px] ${
                  isSelectedForSwap
                    ? 'bg-yellow-600 hover:bg-yellow-500'
                    : 'bg-gray-600 hover:bg-gray-500'
                } text-white px-1.5 py-0.5 rounded transition-colors`}
                aria-label="Swap Subtree Selection"
                title="Select subtree to swap"
              >
                Swap
              </button>
            )}
          </div>
        )}

        {/* Target swap selection trigger */}
        {isSelected && side === 'target' && onSelectForSwap && (
          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSelectForSwap(node.id);
              }}
              className={`text-[10px] ${
                isSelectedForSwap
                  ? 'bg-yellow-600 hover:bg-yellow-500'
                  : 'bg-gray-600 hover:bg-gray-500'
              } text-white px-1.5 py-0.5 rounded transition-colors`}
              aria-label="Swap Subtree Selection"
              title="Select subtree to swap"
            >
              Swap Target
            </button>
          </div>
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
            side={side}
            onCopyComponent={onCopyComponent}
            onCopyChain={onCopyChain}
            onSelectForSwap={onSelectForSwap}
            activeSwapSelection={activeSwapSelection}
          />
        ))}
    </div>
  );
}

export function ModelTree({
  modelName,
  treeNodes,
  selectedNodeId,
  onNodeSelect,
  side,
  onCopyComponent,
  onCopyChain,
  onSelectForSwap,
  activeSwapSelection,
}: ModelTreeProps) {
  // Build node map for fast lookup
  const nodeMap = useMemo(() => {
    const map = new Map<string, URDFTreeNode>();
    for (const node of treeNodes) {
      map.set(node.id, node);
    }
    return map;
  }, [treeNodes]);

  // Find root nodes
  const rootNodes = useMemo(() => {
    return treeNodes.filter((n) => n.parent_id === null || n.node_type === 'root');
  }, [treeNodes]);

  return (
    <div className="flex flex-col h-full bg-gray-800/40 backdrop-blur-md border border-gray-700/50 rounded-xl overflow-hidden shadow-2xl">
      <div className="px-4 py-3 bg-gray-800 border-b border-gray-700/80 flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold text-gray-500 tracking-wider">
            {side === 'source' ? 'Source Model' : side === 'target' ? 'Target Model' : 'Model Structure'}
          </span>
          <span className="text-xs font-semibold text-gray-200 font-mono truncate max-w-[200px]" title={modelName}>
            {modelName || 'No model loaded'}
          </span>
        </div>
        <span className="text-[10px] bg-gray-700 text-gray-400 px-2 py-0.5 rounded-full font-mono">
          {treeNodes.length} nodes
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-1" role="tree">
        {rootNodes.length === 0 ? (
          <div className="text-xs text-gray-500 italic text-center py-8">
            No nodes found
          </div>
        ) : (
          rootNodes.map((node) => (
            <TreeNodeComponent
              key={node.id}
              node={node}
              allNodes={nodeMap}
              selectedId={selectedNodeId}
              onSelect={onNodeSelect}
              side={side}
              onCopyComponent={onCopyComponent}
              onCopyChain={onCopyChain}
              onSelectForSwap={onSelectForSwap}
              activeSwapSelection={activeSwapSelection}
            />
          ))
        )}
      </div>
    </div>
  );
}
