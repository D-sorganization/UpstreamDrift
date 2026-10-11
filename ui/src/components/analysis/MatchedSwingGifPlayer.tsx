/**
 * MatchedSwingGifPlayer — frame-exact GIF playback for the web Results page
 * (#11987).
 *
 * The desktop Results Browser (`src/tools/matched_swing_browser/gui.py`)
 * plays the GIF artefact with a `QMovie` and Play / Pause / Restart buttons:
 * Pause freezes the current frame, Play resumes from it, and Restart jumps
 * back to frame 0 and plays. A browser `<img>` cannot be paused mid-GIF, so
 * this component instead fetches the frame count/durations and renders the
 * current frame as a separately-served PNG
 * (`GET /matched-swings/{id}/animation/frames[/{index}]`), advancing frames
 * on a `setTimeout` keyed to each frame's duration while playing.
 *
 * The parent keys this component on `runId` (`key={run.id}`) so switching
 * runs remounts it with fresh state, instead of resetting state inside an
 * effect body (`react-hooks/set-state-in-effect`).
 */

import { useEffect, useState } from 'react';
import {
  fetchMatchedSwingAnimationInfo,
  matchedSwingAnimationFrameUrl,
  type MatchedSwingAnimationInfo,
} from '@/api/matchedSwings';

type LoadState = 'loading' | 'ready' | 'error';

const BUTTON_CLASS =
  'text-xs rounded border border-gray-700 bg-gray-800 px-2 py-1 hover:border-gray-500 disabled:opacity-40 disabled:cursor-not-allowed';

export function MatchedSwingGifPlayer({
  runId,
  hasAnimation,
}: {
  runId: string;
  hasAnimation: boolean;
}) {
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<MatchedSwingAnimationInfo | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  // Fetch frame info and preload every frame once. Both setState calls
  // happen inside the `.then`/`.catch` callbacks of the fetch promise, never
  // synchronously in the effect body, so this does not trip
  // `react-hooks/set-state-in-effect`.
  useEffect(() => {
    if (!hasAnimation) return;
    let cancelled = false;
    fetchMatchedSwingAnimationInfo(runId)
      .then((data) => {
        if (cancelled) return;
        setInfo(data);
        setLoadState('ready');
        for (let i = 0; i < data.frame_count; i += 1) {
          const preload = new Image();
          preload.src = matchedSwingAnimationFrameUrl(runId, i);
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoadState('error');
      });
    return () => {
      cancelled = true;
    };
  }, [runId, hasAnimation]);

  // Advance the frame on a timer while playing. The `setFrameIndex` call
  // happens inside the `setTimeout` callback, not synchronously in the
  // effect body.
  useEffect(() => {
    if (!playing || !info || info.frame_count <= 1) return;
    const duration = info.durations_ms[frameIndex] ?? 100;
    const timer = setTimeout(() => {
      setFrameIndex((prev) => (prev + 1) % info.frame_count);
    }, duration);
    return () => clearTimeout(timer);
  }, [playing, info, frameIndex]);

  const controlsDisabled = !hasAnimation || loadState !== 'ready' || !info;

  return (
    <div className="flex flex-col items-center gap-2">
      {!hasAnimation && (
        <p className="text-xs text-gray-400">No animation artefact for this run.</p>
      )}
      {hasAnimation && loadState === 'loading' && (
        <p className="text-xs text-gray-400">Loading animation…</p>
      )}
      {hasAnimation && loadState === 'error' && (
        <p className="text-xs text-red-300">Animation unavailable: {error}</p>
      )}
      {hasAnimation && loadState === 'ready' && info && (
        <img
          src={matchedSwingAnimationFrameUrl(runId, frameIndex)}
          alt="matched swing animation frame"
          className="mx-auto max-h-72 rounded border border-gray-700 bg-black"
        />
      )}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setPlaying(true)}
          disabled={controlsDisabled}
          className={BUTTON_CLASS}
        >
          Play
        </button>
        <button
          type="button"
          onClick={() => setPlaying(false)}
          disabled={controlsDisabled}
          className={BUTTON_CLASS}
        >
          Pause
        </button>
        <button
          type="button"
          onClick={() => {
            setFrameIndex(0);
            setPlaying(true);
          }}
          disabled={controlsDisabled}
          className={BUTTON_CLASS}
        >
          Restart
        </button>
      </div>
      {hasAnimation && info && (
        <p className="text-xs text-gray-400">
          Frame {frameIndex + 1} / {info.frame_count}
        </p>
      )}
    </div>
  );
}

export default MatchedSwingGifPlayer;
