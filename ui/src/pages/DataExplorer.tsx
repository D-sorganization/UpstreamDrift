/**
 * DataExplorer - Dataset browser with filtering, sorting, and chart views.
 *
 * Supports CSV/JSON import, tabular data grid view, summary statistics,
 * and column filtering. Connects to the data-explorer REST API.
 *
 * See issue #1206
 */

import { useState, useCallback, useEffect } from 'react';

/** Dataset info from the API. See issue #1206 */
export interface DatasetInfo {
  name: string;
  path: string;
  format: string;
  size_bytes: number;
  columns: string[];
}

/** Dataset preview response. See issue #1206 */
export interface DatasetPreview {
  name: string;
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
  format: string;
}

/** Column statistics. See issue #1206 */
export interface ColumnStats {
  min: number | null;
  max: number | null;
  mean: number | null;
  count: number;
}

/** Dataset statistics response. See issue #1206 */
export interface DatasetStats {
  name: string;
  columns: string[];
  row_count: number;
  stats: Record<string, ColumnStats>;
}

/**
 * DataTable - Tabular display of dataset rows.
 */
function DataTable({
  columns,
  rows,
  sortColumn,
  sortAscending,
  onSort,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  sortColumn: string | null;
  sortAscending: boolean;
  onSort: (column: string) => void;
}) {
  if (columns.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic text-center py-8">
        No data to display
      </div>
    );
  }

  return (
    <div className="overflow-auto max-h-[calc(100vh-200px)]" data-testid="data-table">
      <table className="w-full text-xs text-left">
        <thead className="sticky top-0 bg-gray-800 z-10">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-gray-400 font-medium border-b border-gray-700 cursor-pointer hover:text-gray-200 select-none"
                onClick={() => onSort(col)}
              >
                <span className="flex items-center gap-1">
                  {col}
                  {sortColumn === col && (
                    <span className="text-blue-400">
                      {sortAscending ? ' ^' : ' v'}
                    </span>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr
              key={idx}
              className="hover:bg-gray-700/30 border-b border-gray-800"
            >
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-3 py-1.5 text-gray-300 font-mono truncate max-w-[200px]"
                >
                  {row[col] != null ? String(row[col]) : ''}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * StatsPanel - Column summary statistics display.
 */
function StatsPanel({ stats }: { stats: DatasetStats | null }) {
  if (!stats) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-4">
        Load a dataset to view statistics
      </div>
    );
  }

  return (
    <div className="space-y-2" data-testid="stats-panel">
      <div className="text-xs text-gray-400 mb-2">
        {stats.row_count} rows, {stats.columns.length} columns
      </div>
      {Object.entries(stats.stats).map(([col, colStats]) => (
        <div
          key={col}
          className="bg-gray-700/30 p-2 rounded"
        >
          <div className="text-xs font-medium text-gray-300 mb-1">{col}</div>
          {colStats.mean != null ? (
            <div className="grid grid-cols-3 gap-1 text-xs">
              <div>
                <span className="text-gray-500">min</span>
                <span className="text-gray-300 font-mono ml-1">
                  {colStats.min?.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-gray-500">avg</span>
                <span className="text-gray-300 font-mono ml-1">
                  {colStats.mean?.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-gray-500">max</span>
                <span className="text-gray-300 font-mono ml-1">
                  {colStats.max?.toFixed(2)}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500">Non-numeric</div>
          )}
        </div>
      ))}
    </div>
  );
}

/**
 * DataExplorerPage - Full data explorer tool page.
 *
 * See issue #1206
 */
export function DataExplorerPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'table' | 'stats'>('table');

  // Sorting state
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortAscending, setSortAscending] = useState(true);

  // Filter state
  const [filterColumn, setFilterColumn] = useState('');
  const [filterOperator, setFilterOperator] = useState('eq');
  const [filterValue, setFilterValue] = useState('');

  // Fetch available datasets
  useEffect(() => {
    async function fetchDatasets() {
      try {
        const response = await fetch('/api/tools/data-explorer/datasets');
        if (!response.ok) return;
        const data = await response.json();
        setDatasets(data.datasets || []);
      } catch {
        // API may not be available
      }
    }
    fetchDatasets();
  }, []);

  // Load dataset preview
  const loadDataset = useCallback(async (name: string) => {
    setLoading(true);
    setError(null);
    setSelectedDataset(name);
    setSortColumn(null);
    setFilterColumn('');
    setFilterValue('');

    try {
      const response = await fetch(
        `/api/tools/data-explorer/datasets/${encodeURIComponent(name)}/preview?limit=100`,
      );
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data: DatasetPreview = await response.json();
      setPreview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset');
      setPreview(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Load statistics
  const loadStats = useCallback(async () => {
    if (!selectedDataset) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/tools/data-explorer/datasets/${encodeURIComponent(selectedDataset)}/stats`,
      );
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data: DatasetStats = await response.json();
      setStats(data);
      setViewMode('stats');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stats');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  // Apply filter
  const applyFilter = useCallback(async () => {
    if (!selectedDataset || !filterColumn || !filterValue) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/tools/data-explorer/datasets/${encodeURIComponent(selectedDataset)}/filter`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            column: filterColumn,
            operator: filterOperator,
            value: filterValue,
            limit: 100,
          }),
        },
      );

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data: DatasetPreview = await response.json();
      setPreview(data);
      setViewMode('table');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Filter failed');
    } finally {
      setLoading(false);
    }
  }, [selectedDataset, filterColumn, filterOperator, filterValue]);

  // Handle file import
  const handleImport = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/tools/data-explorer/import', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add to dataset list and select it
      setDatasets((prev) => [
        ...prev,
        {
          name: data.name,
          path: '(imported)',
          format: data.format,
          size_bytes: 0,
          columns: data.columns,
        },
      ]);

      // Load the imported dataset
      await loadDataset(data.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed');
    } finally {
      setLoading(false);
    }
  }, [loadDataset]);

  // Handle sort
  const handleSort = useCallback(
    (column: string) => {
      if (sortColumn === column) {
        setSortAscending(!sortAscending);
      } else {
        setSortColumn(column);
        setSortAscending(true);
      }
    },
    [sortColumn, sortAscending],
  );

  // Sort rows
  const sortedRows = preview?.rows
    ? sortColumn
      ? [...preview.rows].sort((a, b) => {
          const aVal = a[sortColumn];
          const bVal = b[sortColumn];
          if (aVal == null && bVal == null) return 0;
          if (aVal == null) return 1;
          if (bVal == null) return -1;

          const aNum = Number(aVal);
          const bNum = Number(bVal);
          if (!isNaN(aNum) && !isNaN(bNum)) {
            return sortAscending ? aNum - bNum : bNum - aNum;
          }

          const cmp = String(aVal).localeCompare(String(bVal));
          return sortAscending ? cmp : -cmp;
        })
      : preview.rows
    : [];

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Panel: Dataset List + Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white mb-1">Data Explorer</h2>
          <p className="text-xs text-gray-500">
            Browse and analyze simulation datasets
          </p>
        </div>

        {/* Import */}
        <div className="p-4 border-b border-gray-700">
          <label className="w-full py-2 px-4 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded border border-gray-600 transition-colors cursor-pointer block text-center">
            Import CSV/JSON
            <input
              type="file"
              accept=".csv,.json"
              onChange={handleImport}
              className="hidden"
              data-testid="import-file-input"
            />
          </label>
        </div>

        {/* Dataset List */}
        <div className="flex-1 overflow-y-auto p-2">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
            Datasets ({datasets.length})
          </h3>

          {datasets.length === 0 && (
            <div className="text-xs text-gray-500 italic text-center py-4">
              No datasets found. Import a CSV or JSON file.
            </div>
          )}

          {datasets.map((ds) => (
            <button
              key={ds.name}
              onClick={() => loadDataset(ds.name)}
              className={`w-full text-left p-2 rounded mb-1 transition-colors ${
                selectedDataset === ds.name
                  ? 'bg-blue-900/40 ring-1 ring-blue-500/50'
                  : 'hover:bg-gray-700/50'
              }`}
            >
              <div className="text-xs text-gray-200 truncate">{ds.name}</div>
              <div className="text-xs text-gray-500 flex gap-2">
                <span>{ds.format}</span>
                {ds.size_bytes > 0 && (
                  <span>{(ds.size_bytes / 1024).toFixed(1)} KB</span>
                )}
                {ds.columns.length > 0 && (
                  <span>{ds.columns.length} cols</span>
                )}
              </div>
            </button>
          ))}
        </div>

        {/* Filter Controls */}
        {preview && (
          <div className="p-4 border-t border-gray-700 space-y-2">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Filter
            </h3>
            <select
              value={filterColumn}
              onChange={(e) => setFilterColumn(e.target.value)}
              className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1 text-xs border-none"
            >
              <option value="">Select column...</option>
              {preview.columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
            <div className="flex gap-1">
              <select
                value={filterOperator}
                onChange={(e) => setFilterOperator(e.target.value)}
                className="bg-gray-700 text-gray-200 rounded px-2 py-1 text-xs border-none"
              >
                <option value="eq">=</option>
                <option value="ne">!=</option>
                <option value="gt">&gt;</option>
                <option value="lt">&lt;</option>
                <option value="gte">&gt;=</option>
                <option value="lte">&lt;=</option>
                <option value="contains">contains</option>
              </select>
              <input
                type="text"
                value={filterValue}
                onChange={(e) => setFilterValue(e.target.value)}
                placeholder="Value..."
                className="flex-1 bg-gray-700 text-gray-200 rounded px-2 py-1 text-xs border-none"
              />
            </div>
            <button
              onClick={applyFilter}
              disabled={!filterColumn || !filterValue || loading}
              className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white text-xs rounded transition-colors"
            >
              Apply Filter
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mx-4 mb-4 text-xs text-red-400 bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center gap-4">
          <div className="flex gap-1">
            <button
              onClick={() => setViewMode('table')}
              className={`px-3 py-1 text-xs rounded ${
                viewMode === 'table'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Table View
            </button>
            <button
              onClick={() => {
                setViewMode('stats');
                if (!stats && selectedDataset) loadStats();
              }}
              className={`px-3 py-1 text-xs rounded ${
                viewMode === 'stats'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Statistics
            </button>
          </div>

          {preview && (
            <div className="text-xs text-gray-400 ml-auto">
              {preview.total_rows} rows, {preview.columns.length} columns
              {sortColumn && (
                <span className="ml-2 text-blue-400">
                  sorted by {sortColumn} {sortAscending ? 'asc' : 'desc'}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto">
          {loading && (
            <div className="flex items-center justify-center h-full">
              <div className="text-sm text-gray-400">Loading...</div>
            </div>
          )}

          {!loading && !preview && !stats && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-4xl mb-4 text-gray-600">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-16 h-16 mx-auto text-gray-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1}
                      d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-400 mb-2">
                  Select a Dataset
                </h3>
                <p className="text-sm text-gray-500 max-w-xs">
                  Choose a dataset from the sidebar or import a CSV/JSON file
                  to browse and analyze data.
                </p>
              </div>
            </div>
          )}

          {!loading && viewMode === 'table' && preview && (
            <DataTable
              columns={preview.columns}
              rows={sortedRows}
              sortColumn={sortColumn}
              sortAscending={sortAscending}
              onSort={handleSort}
            />
          )}

          {!loading && viewMode === 'stats' && (
            <div className="p-4 max-w-xl mx-auto">
              <StatsPanel stats={stats} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
