/** Save a Blob as a file through a temporary download link. */
export type BlobDownload = (blob: Blob, filename: string) => void;

export const saveBlob: BlobDownload = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};
