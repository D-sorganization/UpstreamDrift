export interface URDFTreeNode {
  id: string;
  name: string;
  node_type: 'link' | 'joint' | 'root';
  parent_id: string | null;
  children: string[];
  properties: Record<string, unknown>;
}

interface JointManipulatorProps {
  joints: URDFTreeNode[];
  jointValues: Record<string, number>;
  onJointChange: (name: string, value: number) => void;
  onResetAll?: () => void;
  onRandomPose?: () => void;
}

export function JointManipulator({
  joints,
  jointValues,
  onJointChange,
  onResetAll,
  onRandomPose,
}: JointManipulatorProps) {
  if (joints.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-2">
        No movable joints
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Control Buttons */}
      <div className="flex gap-2 mb-3">
        <button
          type="button"
          onClick={onResetAll}
          className="flex-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 py-1.5 px-2.5 rounded transition-colors font-semibold cursor-pointer"
        >
          Reset All
        </button>
        <button
          type="button"
          onClick={onRandomPose}
          className="flex-1 text-xs bg-blue-600 hover:bg-blue-500 text-white py-1.5 px-2.5 rounded transition-colors font-semibold cursor-pointer"
        >
          Random Pose
        </button>
      </div>

      {/* Sliders List */}
      <div className="space-y-2">
        {joints.map((joint) => {
          const lower =
            typeof joint.properties.lower === 'number'
              ? joint.properties.lower
              : -3.14;
          const upper =
            typeof joint.properties.upper === 'number'
              ? joint.properties.upper
              : 3.14;
          const value = jointValues[joint.name] ?? 0;

          return (
            <div key={joint.id} className="bg-gray-700/30 p-1.5 rounded">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs text-gray-300 truncate max-w-[120px]">
                  {joint.name}
                </span>
                <span className="text-xs font-mono text-blue-400 font-semibold">
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
    </div>
  );
}
