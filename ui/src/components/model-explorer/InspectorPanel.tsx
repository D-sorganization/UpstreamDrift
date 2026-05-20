import { URDFTreeNode } from '@/utils/frankensteinTree';

interface PropertyInspectorProps {
  node: URDFTreeNode | null;
}

export function PropertyInspector({ node }: PropertyInspectorProps) {
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
        <span className="text-xs bg-gray-705 border border-gray-700 px-1.5 py-0.5 rounded text-gray-400">
          {node.node_type}
        </span>
      </div>

      <div className="space-y-1">
        {Object.entries(node.properties).map(([key, value]) => (
          <div key={key} className="flex justify-between text-xs">
            <span className="text-gray-400 font-mono text-[11px]">{key}</span>
            <span className="text-gray-300 font-mono ml-2 truncate max-w-[150px]" title={String(value)}>
              {typeof value === 'number' ? value.toFixed(4) : String(value)}
            </span>
          </div>
        ))}
      </div>

      {node.parent_id && (
        <div className="text-[11px] text-gray-500 border-t border-gray-700/50 pt-2 font-mono">
          Parent: {node.parent_id}
        </div>
      )}
      {node.children.length > 0 && (
        <div className="text-[11px] text-gray-500 font-mono truncate" title={node.children.join(', ')}>
          Children: {node.children.join(', ')}
        </div>
      )}
    </div>
  );
}

export { JointManipulator } from './JointManipulator';
