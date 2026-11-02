/**
 * Runtime validation schemas for API payloads (issue #7165).
 *
 * The backend and frontend ship separately (Docker vs Tauri vs dev), so
 * version skew is a *when*, not an *if*. `apiFetch<T>` is a compile-time-only
 * assertion: a malformed response is stored as-is and later crashes deep inside
 * a component render (e.g. `capabilities.find is not a function`) or silently
 * corrupts UI state (e.g. a tile missing `order` produces `NaN` sort
 * comparisons). Design-by-Contract at this boundary means *parsing*, not
 * *asserting*.
 *
 * Decision (documented in ui/README.md): we use small hand-rolled type guards
 * rather than adding a runtime-schema dependency (zod). The validated TS types
 * are the existing interfaces; each `parseX` returns the same interface so the
 * compile-time and runtime contracts share one source of truth.
 */

import type {
  CapabilityEntry,
  CapabilityLevel,
  EngineCapabilitiesData,
} from './useEngineCapabilities';
import type {
  LauncherManifest,
  LauncherTile,
  WebLaunchMode,
} from './useLauncherManifest';
import type {
  ActiveTerrain,
  TerrainMaterial,
  TerrainPreset,
  TerrainProperties,
  TerrainTypeInfo,
} from './useTerrain';

/** Error thrown when a backend payload fails runtime validation. */
export class ResponseValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ResponseValidationError';
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function fail(message: string): never {
  throw new ResponseValidationError(`Invalid response format: ${message}`);
}

const CAPABILITY_LEVELS: readonly CapabilityLevel[] = ['full', 'partial', 'none'];

function parseCapabilityEntry(value: unknown, index: number): CapabilityEntry {
  if (!isRecord(value)) {
    fail(`capabilities[${index}] must be an object`);
  }
  const { name, level, supported } = value;
  if (typeof name !== 'string') {
    fail(`capabilities[${index}].name must be a string`);
  }
  if (typeof level !== 'string' || !CAPABILITY_LEVELS.includes(level as CapabilityLevel)) {
    fail(`capabilities[${index}].level must be one of ${CAPABILITY_LEVELS.join('|')}`);
  }
  if (typeof supported !== 'boolean') {
    fail(`capabilities[${index}].supported must be a boolean`);
  }
  return { name, level: level as CapabilityLevel, supported };
}

/**
 * Parse and validate an engine-capabilities payload.
 *
 * @throws ResponseValidationError when the shape does not match the contract.
 */
export function parseEngineCapabilities(value: unknown): EngineCapabilitiesData {
  if (!isRecord(value)) {
    fail('capabilities response must be an object');
  }
  const { engine_name, engine_type, capabilities, summary } = value;
  if (typeof engine_name !== 'string') {
    fail('engine_name must be a string');
  }
  if (typeof engine_type !== 'string') {
    fail('engine_type must be a string');
  }
  if (!Array.isArray(capabilities)) {
    fail('capabilities must be an array');
  }
  if (!isRecord(summary)) {
    fail('summary must be an object');
  }
  for (const key of ['full', 'partial', 'none'] as const) {
    if (typeof summary[key] !== 'number' || !Number.isFinite(summary[key])) {
      fail(`summary.${key} must be a finite number`);
    }
  }
  return {
    engine_name,
    engine_type,
    capabilities: capabilities.map(parseCapabilityEntry),
    summary: {
      full: summary.full as number,
      partial: summary.partial as number,
      none: summary.none as number,
    },
  };
}

const WEB_LAUNCH_MODES: readonly WebLaunchMode[] = [
  'route',
  'native-window',
  'unavailable',
];

/**
 * Validate the optional `web` launch contract (issue #7461). Absence is
 * tolerated (older backends), but a present contract must be internally
 * consistent so the dashboard never renders a dead or dishonest button.
 */
function validateWebContract(value: unknown, index: number): void {
  if (value === undefined) {
    return;
  }
  if (!isRecord(value)) {
    fail(`tiles[${index}].web must be an object`);
  }
  const { mode, route, reason } = value;
  if (typeof mode !== 'string' || !WEB_LAUNCH_MODES.includes(mode as WebLaunchMode)) {
    fail(`tiles[${index}].web.mode must be one of ${WEB_LAUNCH_MODES.join('|')}`);
  }
  if (mode === 'route' && (typeof route !== 'string' || !route.startsWith('/'))) {
    fail(`tiles[${index}].web.route must be a string starting with "/" for mode "route"`);
  }
  if (mode === 'unavailable' && (typeof reason !== 'string' || reason.trim() === '')) {
    fail(`tiles[${index}].web.reason must be a non-empty string for mode "unavailable"`);
  }
}

function parseLauncherTile(value: unknown, index: number): LauncherTile {
  if (!isRecord(value)) {
    fail(`tiles[${index}] must be an object`);
  }
  const tile = value;
  for (const key of ['id', 'name'] as const) {
    if (typeof tile[key] !== 'string') {
      fail(`tiles[${index}].${key} must be a string`);
    }
  }
  // `order` drives a numeric sort; a missing/NaN value silently corrupts
  // tile ordering with no error, so it is the critical invariant here.
  if (
    typeof tile.order !== 'number' ||
    !Number.isFinite(tile.order) ||
    !Number.isInteger(tile.order)
  ) {
    fail(`tiles[${index}].order must be a finite integer`);
  }
  validateWebContract(tile.web, index);
  // Pass through the remaining (already typed) fields unchanged. The two
  // hard invariants above are what prevent the documented failure modes;
  // remaining optional fields are tolerated as the backend evolves.
  return tile as unknown as LauncherTile;
}

/**
 * Parse and validate a launcher-manifest payload.
 *
 * @throws ResponseValidationError when `tiles` is not an array or any tile is
 *   missing a finite-integer `order`.
 */
export function parseLauncherManifest(value: unknown): LauncherManifest {
  if (!isRecord(value)) {
    fail('manifest response must be an object');
  }
  if (!Array.isArray(value.tiles)) {
    fail('tiles must be an array');
  }
  const tiles = value.tiles.map(parseLauncherTile);
  return { ...(value as object), tiles } as LauncherManifest;
}

function terrainCollection(value: unknown, key: string): unknown[] {
  const collection = isRecord(value) ? value[key] : value;
  if (!Array.isArray(collection)) {
    fail(`${key} must be an array`);
  }
  return collection;
}

function terrainString(
  value: Record<string, unknown>,
  key: string,
  path: string,
): string | undefined {
  const field = value[key];
  if (field === undefined || field === null) return undefined;
  if (typeof field !== 'string') fail(`${path}.${key} must be a string`);
  return field;
}

function requiredTerrainString(
  value: Record<string, unknown>,
  key: string,
  path: string,
): string {
  const field = terrainString(value, key, path);
  if (field === undefined || field.trim() === '') {
    fail(`${path}.${key} must be a non-empty string`);
  }
  return field;
}

function terrainStringArray(
  value: unknown,
  path: string,
): string[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    fail(`${path} must be an array of strings`);
  }
  return value;
}

function terrainNumber(
  value: Record<string, unknown>,
  key: string,
  path: string,
): number | undefined {
  const field = value[key];
  if (field === undefined || field === null) return undefined;
  if (typeof field !== 'number' || !Number.isFinite(field)) {
    fail(`${path}.${key} must be a finite number`);
  }
  return field;
}

function terrainNumberArray(
  value: Record<string, unknown>,
  key: string,
  path: string,
): number[] | undefined {
  const field = value[key];
  if (field === undefined || field === null) return undefined;
  if (
    !Array.isArray(field) ||
    field.some((item) => typeof item !== 'number' || !Number.isFinite(item))
  ) {
    fail(`${path}.${key} must be an array of finite numbers`);
  }
  return field;
}

function parseTerrainProperties(value: unknown): TerrainProperties {
  if (value === undefined || value === null) return {};
  if (!isRecord(value)) fail('active terrain.properties must be an object');

  const path = 'active terrain.properties';
  const properties: TerrainProperties = {};
  const dimensions = terrainNumberArray(value, 'dimensions', path);
  const resolution = terrainNumber(value, 'resolution', path);
  const material = terrainString(value, 'material', path);
  const elevationRange = terrainNumberArray(value, 'elevation_range', path);
  const slopeRange = terrainNumberArray(value, 'slope_range', path);
  const features = terrainStringArray(value.features, `${path}.features`);

  if (dimensions !== undefined) properties.dimensions = dimensions;
  if (resolution !== undefined) properties.resolution = resolution;
  if (material !== undefined) properties.material = material;
  if (elevationRange !== undefined) properties.elevation_range = elevationRange;
  if (slopeRange !== undefined) properties.slope_range = slopeRange;
  if (features !== undefined) properties.features = features;
  return properties;
}

/** Normalize old UI records and the current backend's metadata-light payload. */
export function parseActiveTerrain(value: unknown): ActiveTerrain {
  if (!isRecord(value)) fail('active terrain must be an object');
  const name = requiredTerrainString(value, 'name', 'active terrain');
  const properties = parseTerrainProperties(value.properties);

  return {
    id: terrainString(value, 'id', 'active terrain') ?? name,
    name,
    terrain_type:
      terrainString(value, 'terrain_type', 'active terrain') ??
      terrainString(value, 'default_type', 'active terrain') ??
      'N/A',
    material:
      terrainString(value, 'material', 'active terrain') ??
      properties.material ??
      'N/A',
    properties,
    loaded_at: terrainString(value, 'loaded_at', 'active terrain') ?? 'N/A',
  };
}

export function parseTerrainLoadResponse(value: unknown): ActiveTerrain {
  if (!isRecord(value)) fail('terrain load response must be an object');
  if (value.success !== true) {
    const message = terrainString(value, 'error', 'terrain load response');
    fail(message ?? 'Terrain load did not succeed');
  }
  return parseActiveTerrain(value);
}

export function parseTerrainPresets(value: unknown): TerrainPreset[] {
  return terrainCollection(value, 'presets').map((entry, index) => {
    const path = `presets[${index}]`;
    if (!isRecord(entry)) fail(`${path} must be an object`);
    const name = requiredTerrainString(entry, 'name', path);
    const terrainTypes = terrainStringArray(entry.terrain_types, `${path}.terrain_types`);
    const terrainType =
      terrainString(entry, 'terrain_type', path) ?? terrainTypes?.join(', ') ?? 'N/A';
    const description = terrainString(entry, 'description', path) ?? '';
    const id = terrainString(entry, 'id', path) ?? name;
    const width = terrainNumber(entry, 'width_m', path);
    const length = terrainNumber(entry, 'length_m', path);
    const defaults = entry.defaults;
    if (defaults !== undefined && defaults !== null && !isRecord(defaults)) {
      fail(`${path}.defaults must be an object`);
    }

    return {
      id,
      name,
      description,
      terrain_type: terrainType,
      defaults: isRecord(defaults)
        ? defaults
        : {
            ...(width === undefined ? {} : { width_m: width }),
            ...(length === undefined ? {} : { length_m: length }),
          },
    };
  });
}

export function parseTerrainMaterials(value: unknown): TerrainMaterial[] {
  return terrainCollection(value, 'materials').map((entry, index) => {
    const path = `materials[${index}]`;
    if (!isRecord(entry)) fail(`${path} must be an object`);
    const name = requiredTerrainString(entry, 'name', path);
    const id = terrainString(entry, 'id', path) ?? name;
    const friction = terrainNumber(entry, 'friction_coefficient', path);
    const restitution = terrainNumber(entry, 'restitution', path);
    const color = terrainString(entry, 'color', path);

    return {
      id,
      name,
      ...(friction === undefined ? {} : { friction_coefficient: friction }),
      ...(restitution === undefined ? {} : { restitution }),
      ...(color === undefined ? {} : { color }),
    };
  });
}

export function parseTerrainTypes(value: unknown): TerrainTypeInfo[] {
  return terrainCollection(value, 'types').map((entry, index) => {
    if (typeof entry === 'string') {
      if (entry.trim() === '') fail(`types[${index}] must be a non-empty string`);
      return { id: entry, name: entry, description: '' };
    }
    const path = `types[${index}]`;
    if (!isRecord(entry)) fail(`${path} must be an object or string`);
    const name = requiredTerrainString(entry, 'name', path);
    return {
      id: terrainString(entry, 'id', path) ?? name,
      name,
      description: terrainString(entry, 'description', path) ?? '',
    };
  });
}
