/** Mirrors `CanonicalCoreStatus` in `src/api/routes/canonical_core.py`. */
export interface CanonicalCoreStatus {
  tool_id: string;
  mode: string;
  name: string;
  description: string;
  web_route: string;
  capabilities: string[];
  available: boolean;
  reason: string;
  next_step: string;
}

/**
 * Validate a canonical-core status payload at runtime.
 *
 * The backend ships separately from the UI, so a shape mismatch must surface
 * as the page's error state rather than as a render-time TypeError.
 */
export function parseCanonicalCoreStatus(raw: unknown): CanonicalCoreStatus {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Canonical-core status response was not an object');
  }
  const body = raw as Record<string, unknown>;
  if (typeof body.mode !== 'string' || body.mode.length === 0) {
    throw new Error('Canonical-core status response is missing "mode"');
  }
  if (typeof body.available !== 'boolean') {
    throw new Error('Canonical-core status response is missing "available"');
  }
  return {
    tool_id: typeof body.tool_id === 'string' ? body.tool_id : '',
    mode: body.mode,
    name: typeof body.name === 'string' ? body.name : '',
    description: typeof body.description === 'string' ? body.description : '',
    web_route: typeof body.web_route === 'string' ? body.web_route : '',
    capabilities: Array.isArray(body.capabilities)
      ? body.capabilities.filter((capability): capability is string =>
          typeof capability === 'string',
        )
      : [],
    available: body.available,
    reason: typeof body.reason === 'string' ? body.reason : '',
    next_step: typeof body.next_step === 'string' ? body.next_step : '',
  };
}
