/**
 * HelpPanel - Searchable, contextual help system for UpstreamDrift.
 *
 * Features:
 * - Searchable help topics with category grouping
 * - Keyboard shortcut (F1) to open/close
 * - Contextual help via topicId prop
 * - Related topics navigation
 * - Quick tips display
 *
 * See issue #1205
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { X, Search, ChevronRight, HelpCircle, Lightbulb } from 'lucide-react';
import {
  HELP_TOPICS,
  FEATURE_HELP,
  CATEGORY_LABELS,
  searchHelp,
  getTopicsByCategory,
  getRelatedTopics,
  type FeatureHelp,
  type HelpCategory,
} from './helpData';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface HelpPanelProps {
  /** If provided, open directly to this topic. */
  initialTopicId?: string;
  /** Controlled open state. When undefined, component manages its own state. */
  isOpen?: boolean;
  /** Callback when the panel is closed. */
  onClose?: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function HelpPanel({ initialTopicId, isOpen: controlledOpen, onClose }: HelpPanelProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isOpen = controlledOpen !== undefined ? controlledOpen : internalOpen;

  // Track user selection separately from initial prop.
  // Once the user clicks a topic, their selection takes priority.
  const [selectedTopicId, setSelectedTopicId] = useState<string | null>(
    initialTopicId ?? null
  );
  const [searchQuery, setSearchQuery] = useState('');

  const searchInputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // -----------------------------------------------------------------------
  // Open/Close helpers
  // -----------------------------------------------------------------------

  const handleClose = useCallback(() => {
    if (onClose) {
      onClose();
    } else {
      setInternalOpen(false);
    }
    setSearchQuery('');
  }, [onClose]);

  const handleOpen = useCallback(() => {
    if (controlledOpen === undefined) {
      setInternalOpen(true);
    }
  }, [controlledOpen]);

  // -----------------------------------------------------------------------
  // F1 keyboard shortcut
  // -----------------------------------------------------------------------

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'F1') {
        e.preventDefault();
        if (isOpen) {
          handleClose();
        } else {
          handleOpen();
        }
      }
      if (e.key === 'Escape' && isOpen) {
        handleClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, handleClose, handleOpen]);

  // Focus search input when panel opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      // Small delay so the panel renders first
      const timer = setTimeout(() => searchInputRef.current?.focus(), 100);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const effectiveTopicId = selectedTopicId;

  // -----------------------------------------------------------------------
  // Search (derived state, no effect needed)
  // -----------------------------------------------------------------------

  const searchResults = useMemo(() => {
    if (searchQuery.trim()) {
      return searchHelp(searchQuery);
    }
    return [];
  }, [searchQuery]);

  // -----------------------------------------------------------------------
  // Topic selection
  // -----------------------------------------------------------------------

  const selectTopic = useCallback((topicId: string) => {
    setSelectedTopicId(topicId);
    setSearchQuery('');
  }, []);

  const selectedContent: FeatureHelp | null = effectiveTopicId
    ? FEATURE_HELP[effectiveTopicId] ?? null
    : null;
  const relatedTopics = effectiveTopicId ? getRelatedTopics(effectiveTopicId) : [];

  // -----------------------------------------------------------------------
  // Sidebar content
  // -----------------------------------------------------------------------

  const topicsByCategory = getTopicsByCategory();

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  if (!isOpen) {
    return null;
  }

  return (
    <div
      ref={panelRef}
      className="fixed inset-0 z-50 flex"
      role="dialog"
      aria-modal="true"
      aria-label="Help panel"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative ml-auto w-full max-w-3xl h-full flex flex-col bg-gray-900 border-l border-gray-700 shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-700 bg-gray-800/80">
          <HelpCircle className="w-5 h-5 text-blue-400 flex-shrink-0" aria-hidden="true" />
          <h2 className="text-lg font-semibold text-gray-100 flex-1">Help</h2>
          <span className="text-xs text-gray-500 hidden sm:inline">Press F1 to toggle</span>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
            aria-label="Close help panel"
          >
            <X className="w-5 h-5" aria-hidden="true" />
          </button>
        </div>

        {/* Search */}
        <div className="px-4 py-2 border-b border-gray-700">
          <div className="relative">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500"
              aria-hidden="true"
            />
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search help topics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg
                         text-gray-200 text-sm placeholder-gray-500
                         focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              aria-label="Search help topics"
            />
          </div>
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-hidden flex">
          {/* Sidebar */}
          <div className="w-56 flex-shrink-0 border-r border-gray-700 overflow-y-auto p-2">
            {searchQuery.trim() ? (
              // Search results
              <div>
                <div className="px-2 py-1 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Search Results ({searchResults.length})
                </div>
                {searchResults.length === 0 ? (
                  <div className="px-2 py-4 text-sm text-gray-500 text-center">
                    No results found
                  </div>
                ) : (
                  searchResults.map((result) => (
                    <button
                      key={result.topicId}
                      onClick={() => selectTopic(result.topicId)}
                      className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors
                        ${
                          effectiveTopicId === result.topicId
                            ? 'bg-blue-900/40 text-blue-300'
                            : 'text-gray-300 hover:bg-gray-800'
                        }`}
                    >
                      {result.title}
                    </button>
                  ))
                )}
              </div>
            ) : (
              // Category-based navigation
              Object.entries(topicsByCategory).map(([category, topics]) => {
                if (topics.length === 0) return null;
                return (
                  <div key={category} className="mb-3">
                    <div className="px-2 py-1 text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {CATEGORY_LABELS[category as HelpCategory]}
                    </div>
                    {topics.map((topic) => (
                      <button
                        key={topic.id}
                        onClick={() => selectTopic(topic.id)}
                        className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors
                          ${
                            effectiveTopicId === topic.id
                              ? 'bg-blue-900/40 text-blue-300'
                              : 'text-gray-300 hover:bg-gray-800'
                          }`}
                      >
                        {topic.title}
                      </button>
                    ))}
                  </div>
                );
              })
            )}
          </div>

          {/* Main content */}
          <div className="flex-1 overflow-y-auto p-6">
            {selectedContent ? (
              <div>
                {/* Topic title */}
                <h3 className="text-xl font-bold text-gray-100 mb-1">
                  {selectedContent.title}
                </h3>
                <p className="text-sm text-gray-400 mb-4">{selectedContent.short}</p>

                {/* Description - render markdown-like content */}
                <div className="prose prose-invert prose-sm max-w-none mb-6">
                  {selectedContent.description.split('\n').map((line, i) => {
                    if (line.startsWith('**') && line.endsWith('**')) {
                      return (
                        <h4 key={i} className="text-base font-semibold text-gray-200 mt-4 mb-1">
                          {line.replace(/\*\*/g, '')}
                        </h4>
                      );
                    }
                    if (line.startsWith('- ')) {
                      return (
                        <div key={i} className="flex gap-2 ml-2 text-gray-300 text-sm">
                          <span className="text-gray-500 flex-shrink-0">-</span>
                          <span>{line.substring(2)}</span>
                        </div>
                      );
                    }
                    if (/^\d+\./.test(line)) {
                      return (
                        <div key={i} className="ml-2 text-gray-300 text-sm">
                          {line}
                        </div>
                      );
                    }
                    if (line.trim() === '') {
                      return <div key={i} className="h-2" />;
                    }
                    return (
                      <p key={i} className="text-gray-300 text-sm">
                        {line}
                      </p>
                    );
                  })}
                </div>

                {/* Tips */}
                {selectedContent.tips.length > 0 && (
                  <div className="mb-6">
                    <div className="flex items-center gap-2 mb-2">
                      <Lightbulb className="w-4 h-4 text-yellow-400" aria-hidden="true" />
                      <h4 className="text-sm font-semibold text-gray-200">Tips</h4>
                    </div>
                    <ul className="space-y-1">
                      {selectedContent.tips.map((tip, i) => (
                        <li
                          key={i}
                          className="text-sm text-gray-400 pl-4 border-l-2 border-yellow-600/40"
                        >
                          {tip}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Related topics */}
                {relatedTopics.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold text-gray-200 mb-2">
                      Related Topics
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {relatedTopics.map((topic) => (
                        <button
                          key={topic.id}
                          onClick={() => selectTopic(topic.id)}
                          className="inline-flex items-center gap-1 px-3 py-1 rounded-full
                                     bg-gray-800 text-gray-300 text-xs border border-gray-700
                                     hover:border-blue-500 hover:text-blue-300 transition-colors"
                        >
                          {topic.title}
                          <ChevronRight className="w-3 h-3" aria-hidden="true" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // No topic selected - show overview
              <div>
                <h3 className="text-xl font-bold text-gray-100 mb-2">
                  UpstreamDrift Help
                </h3>
                <p className="text-sm text-gray-400 mb-6">
                  Select a topic from the sidebar or use the search bar to find help.
                  Press F1 at any time to toggle this panel.
                </p>

                <div className="grid grid-cols-2 gap-3">
                  {Object.values(HELP_TOPICS).map((topic) => (
                    <button
                      key={topic.id}
                      onClick={() => selectTopic(topic.id)}
                      className="text-left p-3 rounded-lg border border-gray-700 bg-gray-800/50
                                 hover:border-blue-500/50 hover:bg-gray-800 transition-colors"
                    >
                      <div className="text-sm font-medium text-gray-200">
                        {topic.title}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {topic.shortDescription}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Contextual Help Button
// ---------------------------------------------------------------------------

interface HelpButtonProps {
  /** The help topic to display when clicked. */
  topicId: string;
  /** Optional tooltip text. */
  tooltip?: string;
  /** Optional class name override. */
  className?: string;
}

export function HelpButton({ topicId, tooltip, className }: HelpButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={
          className ??
          `inline-flex items-center justify-center w-5 h-5 rounded-full
           bg-gray-700 text-blue-400 text-xs font-bold
           hover:bg-blue-600 hover:text-white transition-colors cursor-pointer`
        }
        title={tooltip ?? 'Click for help'}
        aria-label={`Help: ${HELP_TOPICS[topicId]?.title ?? topicId}`}
      >
        ?
      </button>
      {isOpen && (
        <HelpPanel
          isOpen={isOpen}
          initialTopicId={topicId}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
