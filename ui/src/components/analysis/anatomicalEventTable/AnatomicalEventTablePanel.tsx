import { useId, useState, type FormEvent } from "react";
import type { AnatomicalEventTableRequest } from "@/api/anatomicalEventTable";
import { AnatomicalEventTableFromApi } from "./AnatomicalEventTableFromApi";

/** Analysis-page entry to the shared API table; never substitutes sample data. */
export function AnatomicalEventTablePanel() {
  const inputId = useId();
  const [source, setSource] = useState("");
  const [request, setRequest] = useState<AnatomicalEventTableRequest | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  function compute(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRequest(null);
    try {
      const parsed: unknown = JSON.parse(source);
      if (
        parsed === null ||
        typeof parsed !== "object" ||
        Array.isArray(parsed)
      ) {
        throw new Error("Calibrated frames must be a JSON object.");
      }
      // The existing API validates the complete frame, event and calibration
      // contract. Keep scientific validation in that single authority.
      setRequest(parsed as AnatomicalEventTableRequest);
      setError(null);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }

  return (
    <section
      aria-label="Anatomical Event Table"
      className="space-y-3 rounded-lg border border-gray-700 bg-gray-800 p-4"
    >
      <h3 className="text-sm font-semibold text-gray-300">
        Anatomical Event Table
      </h3>
      <p className="text-sm text-gray-400">
        Compute wrist and forearm angles at swing events from calibrated frames.
        Paste your calibrated frame bundle, including event indices and
        provenance.
      </p>
      <form onSubmit={compute} className="space-y-2">
        <label htmlFor={inputId} className="block text-sm text-gray-300">
          Calibrated Frames (JSON)
        </label>
        <textarea
          id={inputId}
          value={source}
          onChange={(event) => setSource(event.target.value)}
          rows={5}
          maxLength={1_000_000}
          required
          className="w-full rounded border border-gray-600 bg-gray-900 p-2 font-mono text-sm text-gray-100"
        />
        <button
          type="submit"
          className="rounded bg-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-600"
        >
          Compute Event Table
        </button>
      </form>
      {error && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}
      {request && <AnatomicalEventTableFromApi request={request} />}
    </section>
  );
}
