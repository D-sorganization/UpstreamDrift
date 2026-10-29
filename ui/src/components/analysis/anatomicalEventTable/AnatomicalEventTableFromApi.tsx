/**
 * Anatomical event table fed by the API (HRG-E4, issue #13175).
 *
 * Posts the calibrated bundle to `POST /api/biomechanics/anatomical-event-table`
 * and hands the returned export document to the presentational
 * `AnatomicalEventTable`, so the web cells equal the desktop widget's for the
 * same bundle. Shows a loading status while the request is in flight and the
 * server's refusal message when it fails.
 *
 * `request` is compared by identity: memoise it, or the table refetches on
 * every render.
 */

import { useEffect, useState } from "react";
import {
  fetchAnatomicalEventTable,
  type AnatomicalEventTableRequest,
} from "@/api/anatomicalEventTable";
import { AnatomicalEventTable } from "./AnatomicalEventTable";
import type { AnatomicalEventTableDocument } from "./eventTableView";

export interface AnatomicalEventTableFromApiProps {
  request: AnatomicalEventTableRequest;
}

/** Outcome of the fetch for one `request`; absent until it settles. */
type Settled =
  | { request: AnatomicalEventTableRequest; message: string }
  | { request: AnatomicalEventTableRequest; document: AnatomicalEventTableDocument };

export function AnatomicalEventTableFromApi({
  request,
}: AnatomicalEventTableFromApiProps) {
  const [settled, setSettled] = useState<Settled | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchAnatomicalEventTable(request, controller.signal)
      .then((document) => setSettled({ request, document }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setSettled({
          request,
          message: error instanceof Error ? error.message : String(error),
        });
      });
    return () => controller.abort();
  }, [request]);

  // A result for an earlier request is stale: show loading, not old cells.
  if (settled === null || settled.request !== request) {
    return (
      <p role="status" className="text-sm text-gray-400">
        Loading anatomical event table...
      </p>
    );
  }
  if ("message" in settled) {
    return (
      <p role="alert" className="text-sm text-red-400">
        Anatomical event table unavailable: {settled.message}
      </p>
    );
  }
  return <AnatomicalEventTable document={settled.document} />;
}
