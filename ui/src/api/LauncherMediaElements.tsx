/**
 * `<img>` and `<a>` elements for backend API images and downloads.
 *
 * See `launcherMedia.ts` for why these cannot be plain `src`/`href` requests.
 */

import { useState } from 'react';
import type { AnchorHTMLAttributes, ImgHTMLAttributes, MouseEvent } from 'react';

import { downloadWithLauncherCapability, useLauncherObjectUrl } from './launcherMedia';

type LauncherImageProps = Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'> & {
  src: string;
};

/** `<img>` for a backend API image. */
export function LauncherImage({ src, alt, ...rest }: LauncherImageProps) {
  const objectUrl = useLauncherObjectUrl(src);
  return <img {...rest} alt={alt} src={objectUrl ?? undefined} />;
}

type LauncherDownloadLinkProps = Omit<
  AnchorHTMLAttributes<HTMLAnchorElement>,
  'href' | 'onClick'
> & { href: string };

/**
 * `<a>` for a backend API download. It looks and reads like a normal link;
 * activating it downloads the file with the launcher capability.
 */
export function LauncherDownloadLink({
  href,
  download,
  children,
  ...rest
}: LauncherDownloadLinkProps) {
  const [error, setError] = useState<string | null>(null);

  const onClick = (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    setError(null);
    const fileName = typeof download === 'string' ? download : undefined;
    downloadWithLauncherCapability(href, fileName).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err));
    });
  };

  return (
    <>
      <a {...rest} href={href} download={download} onClick={onClick}>
        {children}
      </a>
      {error && (
        <span role="alert" className="text-xs text-red-300">
          {error}
        </span>
      )}
    </>
  );
}
