/**
 * PlotDataChart — generic, data-driven renderer for backend PlotData
 * (issue #7449).
 *
 * Renders any multi-series PlotData payload from
 * `GET /api/analysis/plot-data/{plot_type}` with Recharts. All labels,
 * units, and series names come from the payload; the only per-kind
 * branching is the chart-kind hint the payload itself carries
 * (`metadata.chart === 'radar'`).
 */

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
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import type { PlotData, PlotSeries } from '@/api/useAnalysisPlots';

// Shared qualitative palette (cycled for any number of series).
const PALETTE = [
  '#60a5fa', '#34d399', '#fbbf24', '#f87171',
  '#a78bfa', '#2dd4bf', '#fb923c', '#f472b6',
  '#93c5fd', '#86efac', '#fde047', '#fca5a5',
];

interface Props {
  data: PlotData;
}

function seriesLegendName(s: PlotSeries): string {
  return s.units ? `${s.name} (${s.units})` : s.name;
}

function isEmpty(data: PlotData): boolean {
  return data.series.length === 0 || data.series.every((s) => s.x.length === 0);
}

function RadarPlot({ data }: Props) {
  const series = data.series[0];
  const categories = (series.metadata?.categories as string[] | undefined) ?? [];
  const chartData = categories.map((category, i) => ({
    category,
    value: series.y[i],
  }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={chartData}>
        <PolarGrid stroke="#374151" />
        <PolarAngleAxis dataKey="category" tick={{ fontSize: 11, fill: '#9ca3af' }} />
        <PolarRadiusAxis stroke="#9ca3af" tick={{ fontSize: 10 }} />
        <Radar
          name={seriesLegendName(series)}
          dataKey="value"
          stroke={PALETTE[0]}
          fill={PALETTE[0]}
          fillOpacity={0.4}
          isAnimationActive={false}
        />
        <Legend wrapperStyle={{ fontSize: '11px' }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function MultiSeriesLinePlot({ data }: Props) {
  const perSeriesData = useMemo(
    () =>
      data.series.map((s) =>
        s.x.map((xv, i) => ({ x: xv, y: s.y[i] })),
      ),
    [data],
  );
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart margin={{ top: 5, right: 20, left: 10, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="x"
          type="number"
          domain={['auto', 'auto']}
          stroke="#9ca3af"
          tick={{ fontSize: 10 }}
          label={{
            value: data.x_label,
            position: 'insideBottom',
            offset: -10,
            fill: '#9ca3af',
            fontSize: 11,
          }}
        />
        <YAxis
          stroke="#9ca3af"
          tick={{ fontSize: 10 }}
          label={{
            value: data.y_label,
            angle: -90,
            position: 'insideLeft',
            fill: '#9ca3af',
            fontSize: 11,
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '6px',
          }}
          labelStyle={{ color: '#9ca3af' }}
          itemStyle={{ color: '#e5e7eb' }}
        />
        <Legend wrapperStyle={{ fontSize: '11px' }} />
        {data.series.map((s, i) => (
          <Line
            key={s.name}
            data={perSeriesData[i]}
            dataKey="y"
            name={seriesLegendName(s)}
            stroke={PALETTE[i % PALETTE.length]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PlotDataChart({ data }: Props) {
  if (isEmpty(data)) {
    const message =
      typeof data.metadata?.message === 'string'
        ? data.metadata.message
        : 'No data recorded';
    return (
      <div
        className="flex items-center justify-center h-full text-gray-400"
        data-testid="plot-empty-state"
      >
        <p className="text-sm italic">{message}</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col" data-testid="plot-data-chart">
      <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
        {data.title}
      </h4>
      <div className="flex-1 min-h-0">
        {data.metadata?.chart === 'radar' ? (
          <RadarPlot data={data} />
        ) : (
          <MultiSeriesLinePlot data={data} />
        )}
      </div>
    </div>
  );
}
