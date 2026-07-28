/**
 * ChatContextChip — small header chip showing what the chat assistant can
 * see: active engine, model, and last simulation run.
 *
 * Fetches GET /api/chat/context (shared ChatAppContext schema, issue #7453)
 * and renders e.g. "mujoco · golf_swing.urdf · last run 3.0s". Renders
 * nothing when no live context is available (fetch failure or all-null
 * payload), so it is safe to mount unconditionally.
 */

import { useEffect, useState } from 'react';
import { Eye } from 'lucide-react';
import { apiFetch } from '../../api/fetch';

export interface ChatContextSimulation {
  engine: string | null;
  model: string | null;
  duration_seconds: number | null;
  status: string | null;
}

export interface ChatContextInfo {
  engines_loaded: string[];
  active_engine: string | null;
  active_model: string | null;
  simulation: ChatContextSimulation | null;
}

/**
 * Build the chip label from a context payload.
 *
 * Postcondition: returns null when there is nothing meaningful to show.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function formatContextLabel(ctx: ChatContextInfo | null): string | null {
  if (!ctx) return null;
  const parts: string[] = [];
  if (ctx.active_engine) parts.push(ctx.active_engine);
  if (ctx.active_model) parts.push(ctx.active_model);
  const sim = ctx.simulation;
  if (sim && typeof sim.duration_seconds === 'number') {
    const verb = sim.status === 'running' ? 'running' : 'last run';
    parts.push(`${verb} ${sim.duration_seconds.toFixed(1)}s`);
  } else if (sim && sim.status) {
    parts.push(sim.status);
  }
  return parts.length > 0 ? parts.join(' · ') : null;
}

interface ChatContextChipProps {
  /** Refresh interval in ms; 0 disables polling (tests). Default 15000. */
  refreshMs?: number;
}

export function ChatContextChip({ refreshMs = 15_000 }: ChatContextChipProps = {}) {
  const [context, setContext] = useState<ChatContextInfo | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      if (typeof fetch !== 'function') return;
      try {
        const data = await apiFetch<ChatContextInfo>('/api/chat/context');
        if (!cancelled) setContext(data);
      } catch {
        // Context is best-effort decoration; stay silent on failure.
      }
    };

    void load();
    const timer =
      refreshMs > 0 ? setInterval(() => void load(), refreshMs) : null;
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [refreshMs]);

  const label = formatContextLabel(context);
  if (!label) return null;

  return (
    <span
      className="text-[10px] font-mono flex items-center gap-1 px-2 py-0.5 rounded border"
      data-testid="chat-context-chip"
      title="Live app context visible to the assistant"
      style={{
        borderColor: 'var(--sidekick-color-border)',
        color: 'var(--sidekick-color-text-subtle)',
      }}
    >
      <Eye className="w-3 h-3" aria-hidden="true" />
      {label}
    </span>
  );
}

export default ChatContextChip;
