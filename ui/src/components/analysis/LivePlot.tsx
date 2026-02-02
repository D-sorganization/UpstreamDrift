import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { SimulationFrame } from '@/api/client';

interface Props {
  frames: SimulationFrame[];
  maxPoints?: number;
}

// Color palette for different data series
const COLORS = {
  jointAngles: ['#60a5fa', '#34d399', '#fbbf24', '#f87171'],
  velocities: ['#a78bfa', '#2dd4bf', '#fb923c', '#f472b6'],
};

export function LivePlot({ frames, maxPoints = 100 }: Props) {
  // Process frame data for plotting
  const chartData = useMemo(() => {
    const recentFrames = frames.slice(-maxPoints);

    return recentFrames.map((frame) => {
      const data: Record<string, number | string> = {
        time: frame.time.toFixed(3),
        frame: frame.frame,
      };

      // Add joint angles if available
      if (frame.analysis?.joint_angles) {
        frame.analysis.joint_angles.forEach((angle, idx) => {
          data[`angle_${idx}`] = angle;
        });
      }

      // Add velocities if available
      if (frame.analysis?.velocities) {
        frame.analysis.velocities.forEach((vel, idx) => {
          data[`vel_${idx}`] = vel;
        });
      }

      return data;
    });
  }, [frames, maxPoints]);

  // Determine which data series are available
  const hasJointAngles = frames.some(
    (f) => f.analysis?.joint_angles && f.analysis.joint_angles.length > 0
  );
  const hasVelocities = frames.some(
    (f) => f.analysis?.velocities && f.analysis.velocities.length > 0
  );

  // Get number of joints from first frame with data
  const numJoints = useMemo(() => {
    for (const frame of frames) {
      if (frame.analysis?.joint_angles) {
        return frame.analysis.joint_angles.length;
      }
    }
    return 0;
  }, [frames]);

  if (frames.length < 2) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <p className="text-sm italic">Waiting for simulation data...</p>
      </div>
    );
  }

  if (!hasJointAngles && !hasVelocities) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <p className="text-sm italic">No analysis data available. Enable Live Analysis in parameters.</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Joint Angles Chart */}
      {hasJointAngles && (
        <div className="flex-1 min-h-0">
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
            Joint Angles (rad)
          </h4>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                stroke="#9ca3af"
                tick={{ fontSize: 10 }}
                tickFormatter={(value) => `${value}s`}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fontSize: 10 }}
                tickFormatter={(value: number) => value.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '6px',
                }}
                labelStyle={{ color: '#9ca3af' }}
                itemStyle={{ color: '#e5e7eb' }}
                formatter={(value) => [(value as number)?.toFixed(4) ?? '', '']}
              />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {Array.from({ length: Math.min(numJoints, 4) }, (_, i) => (
                <Line
                  key={`angle_${i}`}
                  type="monotone"
                  dataKey={`angle_${i}`}
                  name={`Joint ${i + 1}`}
                  stroke={COLORS.jointAngles[i]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Velocities Chart */}
      {hasVelocities && (
        <div className="flex-1 min-h-0">
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
            Joint Velocities (rad/s)
          </h4>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                stroke="#9ca3af"
                tick={{ fontSize: 10 }}
                tickFormatter={(value) => `${value}s`}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fontSize: 10 }}
                tickFormatter={(value: number) => value.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '6px',
                }}
                labelStyle={{ color: '#9ca3af' }}
                itemStyle={{ color: '#e5e7eb' }}
                formatter={(value) => [(value as number)?.toFixed(4) ?? '', '']}
              />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {Array.from({ length: Math.min(numJoints, 4) }, (_, i) => (
                <Line
                  key={`vel_${i}`}
                  type="monotone"
                  dataKey={`vel_${i}`}
                  name={`Joint ${i + 1}`}
                  stroke={COLORS.velocities[i]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
