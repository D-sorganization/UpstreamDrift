/**
 * GlossaryTooltip — inline, contextual glossary tooltip.
 *
 * Wraps any inline content. On hover, fetches `/glossary/{termId}?level=...`
 * and displays a short description. On click, opens the HelpPanel scrolled
 * to the entry via the shared UI store.
 *
 * See issue #3165.
 */

import { useCallback, useState } from 'react';
import { useGlossaryStore, type Definition } from '@/stores/useGlossaryStore';
import { useUIStore, type ExpertiseLevel } from '@/stores/useUIStore';

interface GlossaryTooltipProps {
  termId: string;
  level?: ExpertiseLevel;
  children: React.ReactNode;
}

export function GlossaryTooltip({ termId, level, children }: GlossaryTooltipProps) {
  const fetchDefinition = useGlossaryStore((s) => s.fetch);
  const storeLevel = useUIStore((s) => s.expertiseLevel);
  const openHelpPanel = useUIStore((s) => s.openHelpPanel);

  const effectiveLevel: ExpertiseLevel = level ?? storeLevel;
  const cachedDef = useGlossaryStore((s) => s.cache[`${termId}@${effectiveLevel}`]);

  const [visible, setVisible] = useState(false);
  const [definition, setDefinition] = useState<Definition | null>(cachedDef ?? null);

  const handleEnter = useCallback(async () => {
    setVisible(true);
    if (!definition) {
      const def = await fetchDefinition(termId, effectiveLevel);
      if (def) {
        setDefinition(def);
      }
    }
  }, [definition, fetchDefinition, termId, effectiveLevel]);

  const handleLeave = useCallback(() => {
    setVisible(false);
  }, []);

  const handleClick = useCallback(() => {
    openHelpPanel(termId);
  }, [openHelpPanel, termId]);

  return (
    <span
      className="glossary-tooltip relative inline-flex items-center gap-1 cursor-help underline decoration-dotted decoration-blue-400/60 underline-offset-2"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      onFocus={handleEnter}
      onBlur={handleLeave}
      onClick={handleClick}
      data-term-id={termId}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
      aria-label={`Glossary: ${termId}`}
    >
      {children}
      {visible && definition && (
        <span
          role="tooltip"
          className="absolute left-0 top-full z-50 mt-1 max-w-xs whitespace-normal
                     rounded-md border border-gray-600 bg-gray-900/95 px-3 py-2
                     text-xs text-gray-100 shadow-lg"
        >
          <span className="block font-semibold text-blue-300">{definition.title}</span>
          <span className="block text-gray-200">{definition.short}</span>
        </span>
      )}
    </span>
  );
}
