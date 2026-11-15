/**
 * Images and downloads served by the backend API.
 *
 * A plain `<img src>` or `<a href>` request cannot carry the launcher
 * capability header (see `launcherFetch`). These helpers fetch the resource
 * through `launcherFetch` instead and hand the browser a local object URL.
 */

import { useEffect, useState } from 'react';

import { launcherFetch } from './launcherFetch';

/**
 * Fetch `url` with the launcher capability and return an object URL for it.
 *
 * @precondition `url` is a non-empty string.
 * @postcondition The caller owns the returned URL and must revoke it.
 */
export async function fetchLauncherObjectUrl(url: string): Promise<string> {
  if (!url) throw new Error('url must be a non-empty string');
  const response = await launcherFetch(url);
  if (!response.ok) {
    throw new Error(`Request failed with HTTP ${response.status}`);
  }
  return URL.createObjectURL(await response.blob());
}

/** File name from a `Content-Disposition` header, or `null` when it has none. */
export function fileNameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const encoded = /filename\*\s*=\s*[^']*'[^']*'([^;]+)/i.exec(header);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1].trim());
    } catch {
      return null;
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(header);
  return plain ? plain[1].trim() : null;
}

function lastPathSegment(url: string): string {
  const path = url.split(/[?#]/)[0];
  return path.slice(path.lastIndexOf('/') + 1) || 'download';
}

/**
 * Download `url` with the launcher capability and save it through the browser.
 *
 * The saved name is `fileName` when given, else the name the server sends,
 * else the last path segment of `url`.
 */
export async function downloadWithLauncherCapability(
  url: string,
  fileName?: string,
): Promise<void> {
  if (!url) throw new Error('url must be a non-empty string');
  const response = await launcherFetch(url);
  if (!response.ok) {
    throw new Error(`Download failed with HTTP ${response.status}`);
  }
  const objectUrl = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a');
  anchor.href = objectUrl;
  anchor.download =
    fileName ||
    fileNameFromDisposition(response.headers.get('Content-Disposition')) ||
    lastPathSegment(url);
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(objectUrl);
}

/**
 * Object URL for `url`, or `null` while it loads, after it fails, or when
 * `url` is `null`. The URL is revoked when `url` changes or on unmount.
 */
export function useLauncherObjectUrl(url: string | null): string | null {
  const [loaded, setLoaded] = useState<{ url: string; objectUrl: string } | null>(null);

  useEffect(() => {
    if (!url) return undefined;
    let cancelled = false;
    let objectUrl: string | null = null;
    fetchLauncherObjectUrl(url)
      .then((created) => {
        if (cancelled) {
          URL.revokeObjectURL(created);
          return;
        }
        objectUrl = created;
        setLoaded({ url, objectUrl: created });
      })
      .catch(() => {
        // Leave the image empty; the alt text stays visible.
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  return loaded && loaded.url === url ? loaded.objectUrl : null;
}
