/**
 * OnboardingOverlay — first-run web onboarding (issue #7459, parity G11).
 *
 * Mirrors the desktop onboarding dialog
 * (src/launchers/onboarding_dialog.py): welcome header, info cards whose
 * copy is single-sourced from src/config/onboarding_cards.json (served by
 * GET /api/v1/about/onboarding), and a "Don't show again" checkbox.
 *
 * Dismissal is persisted through the onboardingPersistence adapter
 * (localStorage today; server-side settings arrive with #7457).
 */

import { useEffect, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { fetchOnboardingCopy, type OnboardingCopy } from '@/api/about';
import { onboardingPersistence } from '@/utils/onboardingStorage';

export function OnboardingOverlay() {
  const [copy, setCopy] = useState<OnboardingCopy | null>(null);
  const [visible, setVisible] = useState(false);
  const [dontShowAgain, setDontShowAgain] = useState(false);

  useEffect(() => {
    if (onboardingPersistence.isDismissed()) return;
    let cancelled = false;
    fetchOnboardingCopy()
      .then((data) => {
        // Guard against malformed payloads (e.g. older backends).
        if (!cancelled && Array.isArray(data?.cards) && data.cards.length > 0) {
          setCopy(data);
          setVisible(true);
        }
      })
      .catch(() => {
        // Backend unreachable — skip onboarding rather than show an
        // empty overlay; the dashboard surfaces connection errors itself.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleClose = () => {
    if (dontShowAgain) {
      onboardingPersistence.dismiss();
    }
    setVisible(false);
  };

  if (!visible || !copy) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={copy.header}
        className="w-full max-w-2xl rounded-xl border border-gray-700 bg-gray-900 p-8 shadow-2xl"
      >
        <h2 className="text-center text-2xl font-bold text-white">{copy.header}</h2>
        <p className="mt-1 text-center text-sm text-gray-400">{copy.subtitle}</p>

        <hr className="my-5 border-gray-700" />

        <div className="grid gap-3 sm:grid-cols-3">
          {copy.cards.map((card) => (
            <div
              key={card.id ?? card.title}
              className="flex flex-col gap-2 rounded-lg border border-gray-700/60 bg-gray-800/50 p-4"
            >
              <h3 className="text-sm font-bold text-white">{card.title}</h3>
              <p className="flex-1 text-xs leading-relaxed text-gray-400">{card.body}</p>
              <a
                href={card.link_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-blue-300 hover:text-blue-200 hover:underline"
              >
                <ExternalLink className="h-3 w-3" aria-hidden="true" />
                {card.link_text}
              </a>
            </div>
          ))}
        </div>

        <div className="mt-6 flex items-center justify-between gap-4">
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={(e) => setDontShowAgain(e.target.checked)}
              className="h-4 w-4 rounded border-gray-600 bg-gray-800"
            />
            Don't show this welcome message again
          </label>
          <button
            onClick={handleClose}
            className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
}
