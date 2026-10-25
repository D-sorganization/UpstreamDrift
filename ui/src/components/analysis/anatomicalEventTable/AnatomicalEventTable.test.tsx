import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import golden from "./fixtures/eventTableView.json";
import { AnatomicalEventTable } from "./AnatomicalEventTable";
import {
  MISSING_TEXT,
  type AnatomicalEventTableDocument,
} from "./eventTableView";

const document = golden.document as AnatomicalEventTableDocument;

afterEach(cleanup);

function bodyCells(): string[][] {
  const table = screen.getByTestId("anatomical-event-table");
  const [, body] = within(table).getAllByRole("rowgroup");
  return within(body)
    .getAllByRole("row")
    .map((row) =>
      within(row)
        .getAllByRole("cell")
        .map((cell) => cell.textContent ?? ""),
    );
}

describe("AnatomicalEventTable (parity with the PyQt widget, #13175)", () => {
  it("renders the golden caption, headers and cells", () => {
    render(<AnatomicalEventTable document={document} />);
    expect(
      screen.getByTestId("anatomical-event-table-caption"),
    ).toHaveTextContent(golden.view.caption);
    const headers = screen
      .getAllByRole("columnheader")
      .map((h) => h.textContent);
    expect(headers).toEqual(golden.view.columns);
    expect(bodyCells()).toEqual(golden.view.rows);
  });

  it("gives a missing value its reason as a title", () => {
    render(<AnatomicalEventTable document={document} />);
    const missing = screen.getAllByText(MISSING_TEXT);
    expect(missing.length).toBeGreaterThan(0);
    for (const cell of missing)
      expect(cell).toHaveAttribute("title", "marker occluded");
  });

  it("shows an error instead of a table for an invalid document", () => {
    render(
      <AnatomicalEventTable document={{ ...document, schema: "other" }} />,
    );
    expect(
      screen.queryByTestId("anatomical-event-table"),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/schema/);
  });
});
