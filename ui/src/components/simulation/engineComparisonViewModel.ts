import type { ManagedEngine } from '@/stores/useEngineStore';
import type { SimulationFrame } from '@/api/client';

const COMPARISON_CAPABILITY_ALIASES = new Set([
  'forward_simulation',
  'rollout',
  'rigid_body',
  'musculoskeletal',
  'simulation',
]);

const DIVERGENCE_WARNING_THRESHOLD = 0.05;
const DIVERGENCE_CRITICAL_THRESHOLD = 0.2;

export type ComparisonSupport = 'ready' | 'blocked' | 'pending';
export type DivergenceSeverity = 'ok' | 'warning' | 'critical' | 'pending';

export interface EngineComparisonOption {
  name: string;
  displayName: string;
  support: ComparisonSupport;
  disabledReason: string | null;
  capabilities: string[];
}

export interface ComparisonProvenance {
  engine: string;
  version: string;
  frame: number | null;
  time: number | null;
  capabilities: string[];
}

export interface EngineComparisonColumn {
  name: string;
  displayName: string;
  hasFrame: boolean;
  provenance: ComparisonProvenance;
  metrics: Record<string, number>;
}

export interface DivergenceAnnotation {
  metric: string;
  baseline: string;
  compared: string;
  delta: number | null;
  severity: DivergenceSeverity;
  label: string;
}

export interface EngineComparisonViewModel {
  datasetLabel: string;
  canCompare: boolean;
  options: EngineComparisonOption[];
  selectedEngineNames: string[];
  columns: EngineComparisonColumn[];
  annotations: DivergenceAnnotation[];
  emptyMessage: string | null;
}

export function engineSupportsComparison(engine: ManagedEngine): boolean {
  return engine.capabilities.some((capability) =>
    COMPARISON_CAPABILITY_ALIASES.has(capability),
  );
}

export function buildEngineComparisonOptions(
  engines: ManagedEngine[],
): EngineComparisonOption[] {
  return engines.map((engine) => {
    if (engine.loadState !== 'loaded') {
      return {
        name: engine.name,
        displayName: engine.displayName,
        support: engine.available ? 'pending' : 'blocked',
        disabledReason: engine.available ? 'Load engine to compare' : 'Engine unavailable',
        capabilities: engine.capabilities,
      };
    }
    if (!engineSupportsComparison(engine)) {
      return {
        name: engine.name,
        displayName: engine.displayName,
        support: 'blocked',
        disabledReason: 'No comparable rollout capability advertised',
        capabilities: engine.capabilities,
      };
    }
    return {
      name: engine.name,
      displayName: engine.displayName,
      support: 'ready',
      disabledReason: null,
      capabilities: engine.capabilities,
    };
  });
}

export function coerceComparisonSelection(
  selectedEngineNames: string[],
  options: EngineComparisonOption[],
): string[] {
  const readyNames = new Set(
    options.filter((option) => option.support === 'ready').map((option) => option.name),
  );
  const filtered = selectedEngineNames.filter((name) => readyNames.has(name));
  if (
    filtered.length === selectedEngineNames.length &&
    filtered.every((name, index) => name === selectedEngineNames[index])
  ) {
    return selectedEngineNames;
  }
  return filtered;
}

export function toggleComparisonEngine(
  selectedEngineNames: string[],
  engineName: string,
  options: EngineComparisonOption[],
): string[] {
  const option = options.find((candidate) => candidate.name === engineName);
  if (!option || option.support !== 'ready') {
    return selectedEngineNames;
  }
  if (selectedEngineNames.includes(engineName)) {
    return selectedEngineNames.filter((name) => name !== engineName);
  }
  return [...selectedEngineNames, engineName];
}

export function buildEngineComparisonViewModel({
  engines,
  selectedEngineNames,
  framesByEngine,
  datasetLabel,
}: {
  engines: ManagedEngine[];
  selectedEngineNames: string[];
  framesByEngine: Record<string, SimulationFrame | null | undefined>;
  datasetLabel: string;
}): EngineComparisonViewModel {
  const options = buildEngineComparisonOptions(engines);
  const coercedSelection = coerceComparisonSelection(selectedEngineNames, options);
  const columns = buildColumns(engines, coercedSelection, framesByEngine);

  return {
    datasetLabel,
    canCompare: columns.length >= 2,
    options,
    selectedEngineNames: coercedSelection,
    columns,
    annotations: buildAnnotations(columns),
    emptyMessage: buildEmptyMessage(coercedSelection, columns),
  };
}

function buildColumns(
  engines: ManagedEngine[],
  selectedEngineNames: string[],
  framesByEngine: Record<string, SimulationFrame | null | undefined>,
): EngineComparisonColumn[] {
  return selectedEngineNames.flatMap((engineName) => {
    const engine = engines.find((candidate) => candidate.name === engineName);
    if (!engine) return [];
    const frame = framesByEngine[engineName] ?? null;
    return [
      {
        name: engine.name,
        displayName: engine.displayName,
        hasFrame: frame !== null,
        provenance: {
          engine: engine.name,
          version: engine.version ?? 'unknown',
          frame: frame?.frame ?? null,
          time: frame?.time ?? null,
          capabilities: engine.capabilities,
        },
        metrics: frame ? extractComparableMetrics(frame) : {},
      },
    ];
  });
}

function extractComparableMetrics(frame: SimulationFrame): Record<string, number> {
  const metrics: Record<string, number> = {};
  visitMetricValue('state', frame.state, metrics);
  if (frame.analysis) {
    visitMetricValue('analysis', frame.analysis, metrics);
  }
  return metrics;
}

function visitMetricValue(
  prefix: string,
  value: unknown,
  metrics: Record<string, number>,
): void {
  if (typeof value === 'number' && Number.isFinite(value)) {
    metrics[prefix] = value;
    return;
  }
  if (Array.isArray(value)) {
    const finiteNumbers = value.filter(
      (item): item is number => typeof item === 'number' && Number.isFinite(item),
    );
    if (finiteNumbers.length === value.length && value.length > 0) {
      metrics[`${prefix}.rms`] = rms(finiteNumbers);
      metrics[`${prefix}.max`] = Math.max(...finiteNumbers.map((item) => Math.abs(item)));
    }
    return;
  }
  if (value && typeof value === 'object') {
    Object.entries(value as Record<string, unknown>).forEach(([key, child]) => {
      visitMetricValue(`${prefix}.${key}`, child, metrics);
    });
  }
}

function rms(values: number[]): number {
  const sumSquares = values.reduce((total, value) => total + value * value, 0);
  return Math.sqrt(sumSquares / values.length);
}

function buildAnnotations(
  columns: EngineComparisonColumn[],
): DivergenceAnnotation[] {
  if (columns.length < 2) return [];
  const baseline = columns[0];
  return columns.slice(1).flatMap<DivergenceAnnotation>((column) => {
    if (!baseline.hasFrame || !column.hasFrame) {
      return [
        {
          metric: 'run data',
          baseline: baseline.name,
          compared: column.name,
          delta: null,
          severity: 'pending' as const,
          label: 'Run each selected engine on this dataset to populate comparison data',
        },
      ];
    }
    const metricNames = Object.keys(baseline.metrics).filter(
      (metric) => metric in column.metrics,
    );
    if (metricNames.length === 0) {
      return [
        {
          metric: 'metrics',
          baseline: baseline.name,
          compared: column.name,
          delta: null,
          severity: 'pending' as const,
          label: 'No shared numeric outputs yet',
        },
      ];
    }
    return metricNames.map((metric) => {
      const delta = Math.abs(column.metrics[metric] - baseline.metrics[metric]);
      return {
        metric,
        baseline: baseline.name,
        compared: column.name,
        delta,
        severity: classifyDelta(delta),
        label: formatDeltaLabel(metric, delta),
      };
    });
  });
}

function classifyDelta(delta: number): DivergenceSeverity {
  if (delta >= DIVERGENCE_CRITICAL_THRESHOLD) return 'critical';
  if (delta >= DIVERGENCE_WARNING_THRESHOLD) return 'warning';
  return 'ok';
}

function formatDeltaLabel(metric: string, delta: number): string {
  return `${metric}: ${delta.toExponential(2)} absolute delta`;
}

function buildEmptyMessage(
  selectedEngineNames: string[],
  columns: EngineComparisonColumn[],
): string | null {
  if (selectedEngineNames.length < 2) {
    return 'Select at least two loaded comparable engines';
  }
  if (columns.some((column) => !column.hasFrame)) {
    return 'Run each selected engine on this dataset to fill the comparison';
  }
  return null;
}
