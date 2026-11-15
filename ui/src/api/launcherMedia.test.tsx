import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';

import { LAUNCHER_CAPABILITY_HEADER } from './launcherFetch';
import {
  downloadWithLauncherCapability,
  fetchLauncherObjectUrl,
  fileNameFromDisposition,
} from './launcherMedia';
import { LauncherDownloadLink, LauncherImage } from './LauncherMediaElements';
import { setLauncherCapabilityToken } from './websocketToken';

const IMAGE_URL = '/api/v1/necromatcher/captures/c1/frames/0/image';
const FILE_URL = '/api/recordings/rec-1/export?format=csv';

function blobResponse(status = 200, headers: Record<string, string> = {}): Response {
  return new Response(new Blob(['data']), { status, headers });
}

function sentHeaders(call: unknown[]): Record<string, string> {
  return ((call[1] as RequestInit | undefined)?.headers ?? {}) as Record<string, string>;
}

describe('launcherMedia', () => {
  const fetchMock = vi.fn();
  const createObjectURL = vi.fn(() => 'blob:local-1');
  const revokeObjectURL = vi.fn();
  let clickedDownloads: string[];

  beforeEach(() => {
    fetchMock.mockReset();
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
    clickedDownloads = [];
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL, revokeObjectURL }));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function click(
      this: HTMLAnchorElement,
    ) {
      clickedDownloads.push(this.download);
    });
    setLauncherCapabilityToken('cap-1');
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    setLauncherCapabilityToken(null);
  });

  it('fetches an object URL with the capability header', async () => {
    fetchMock.mockResolvedValueOnce(blobResponse());

    await expect(fetchLauncherObjectUrl(IMAGE_URL)).resolves.toBe('blob:local-1');

    expect(fetchMock.mock.calls[0][0]).toBe(IMAGE_URL);
    expect(sentHeaders(fetchMock.mock.calls[0])[LAUNCHER_CAPABILITY_HEADER]).toBe('cap-1');
  });

  it('rejects when the backend refuses the request', async () => {
    fetchMock.mockResolvedValue(blobResponse(403));
    await expect(fetchLauncherObjectUrl(IMAGE_URL)).rejects.toThrow('403');
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it('reads the file name from Content-Disposition', () => {
    expect(fileNameFromDisposition('attachment; filename="run 1.zip"')).toBe('run 1.zip');
    expect(fileNameFromDisposition("attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf")).toBe(
      'résumé.pdf',
    );
    expect(fileNameFromDisposition('inline')).toBeNull();
    expect(fileNameFromDisposition(null)).toBeNull();
  });

  it.each([
    ['explicit.csv', {}, 'explicit.csv'],
    [undefined, { 'Content-Disposition': 'attachment; filename="server.csv"' }, 'server.csv'],
    [undefined, {}, 'export'],
  ])('saves a download as %s / %o -> %s', async (fileName, headers, expected) => {
    fetchMock.mockResolvedValueOnce(blobResponse(200, headers as Record<string, string>));

    await downloadWithLauncherCapability(FILE_URL, fileName);

    expect(clickedDownloads).toEqual([expected]);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:local-1');
  });

  it('LauncherImage shows the fetched image and revokes it on unmount', async () => {
    fetchMock.mockResolvedValueOnce(blobResponse());

    const { unmount } = render(<LauncherImage src={IMAGE_URL} alt="Frame" className="w-full" />);
    const image = screen.getByAltText('Frame');
    expect(image).not.toHaveAttribute('src');
    await waitFor(() => expect(image).toHaveAttribute('src', 'blob:local-1'));
    expect(image).toHaveClass('w-full');

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:local-1');
  });

  it('LauncherDownloadLink keeps its href and downloads through the capability', async () => {
    fetchMock.mockResolvedValueOnce(blobResponse());

    render(
      <LauncherDownloadLink href={FILE_URL} download="rec-1.csv" className="link">
        CSV
      </LauncherDownloadLink>,
    );
    const link = screen.getByRole('link', { name: 'CSV' });
    expect(link).toHaveAttribute('href', FILE_URL);
    fireEvent.click(link);

    await waitFor(() => expect(clickedDownloads).toEqual(['rec-1.csv']));
    expect(sentHeaders(fetchMock.mock.calls[0])[LAUNCHER_CAPABILITY_HEADER]).toBe('cap-1');
  });

  it('LauncherDownloadLink reports a failed download', async () => {
    fetchMock.mockResolvedValue(blobResponse(500));

    render(<LauncherDownloadLink href={FILE_URL}>CSV</LauncherDownloadLink>);
    fireEvent.click(screen.getByRole('link', { name: 'CSV' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('500');
    expect(clickedDownloads).toEqual([]);
  });
});
