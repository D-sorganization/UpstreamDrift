import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router';
import { ScrollToTop } from './utils/ScrollToTop';
import { RouteTitle } from './utils/RouteTitle';
import { ToastProvider } from './components/ui/Toast';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { useWebSettingsBootstrap } from './api/useWebSettings';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';
import { HelpPanel } from './components/ui/HelpPanel';
import { useUIStore } from './stores';
import { NotFoundPage } from './pages/NotFound';

/**
 * Route-level code splitting (#7433): every page is lazy so the initial bundle
 * no longer ships three.js / @react-three/fiber / drei (pulled in by Scene3D,
 * URDFViewer, ModelPreviewViewport) on routes that render no 3D view. Pages use
 * named exports, so each import is mapped to a `default` for `React.lazy`.
 */
const DashboardPage = lazy(() =>
  import('./pages/Dashboard').then((m) => ({ default: m.DashboardPage })),
);
const SimulationPage = lazy(() =>
  import('./pages/Simulation').then((m) => ({ default: m.SimulationPage })),
);
const ModelExplorerPage = lazy(() =>
  import('./pages/ModelExplorer').then((m) => ({ default: m.ModelExplorerPage })),
);
const PuttingGreenPage = lazy(() =>
  import('./pages/PuttingGreen').then((m) => ({ default: m.PuttingGreenPage })),
);
const VideoAnalyzerPage = lazy(() =>
  import('./pages/VideoAnalyzer').then((m) => ({ default: m.VideoAnalyzerPage })),
);
const DataExplorerPage = lazy(() =>
  import('./pages/DataExplorer').then((m) => ({ default: m.DataExplorerPage })),
);
const MotionCapturePage = lazy(() =>
  import('./pages/MotionCapture').then((m) => ({ default: m.MotionCapturePage })),
);
const ChatPage = lazy(() =>
  import('./pages/Chat').then((m) => ({ default: m.ChatPage })),
);
const TerrainPage = lazy(() =>
  import('./pages/Terrain').then((m) => ({ default: m.TerrainPage })),
);
const DatasetGeneratorPage = lazy(() =>
  import('./pages/DatasetGenerator').then((m) => ({
    default: m.DatasetGeneratorPage,
  })),
);
const AnalysisToolsPage = lazy(() =>
  import('./pages/AnalysisTools').then((m) => ({ default: m.AnalysisToolsPage })),
);
const CharacterBuilderPage = lazy(() =>
  import('./pages/CharacterBuilder').then((m) => ({
    default: m.CharacterBuilderPage,
  })),
);
const CanonicalCoreShellPage = lazy(() =>
  import('./pages/CanonicalCoreShell').then((m) => ({
    default: m.CanonicalCoreShellPage,
  })),
);
const BallFlightPage = lazy(() =>
  import('./pages/BallFlight').then((m) => ({ default: m.BallFlightPage })),
);
const SettingsPage = lazy(() =>
  import('./pages/Settings').then((m) => ({ default: m.SettingsPage })),
);
/** Themed full-viewport fallback shown while a route chunk loads (#7433). */
function PageLoadingFallback() {
  return (
    <div
      className="flex h-screen w-full items-center justify-center bg-gray-900"
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-col items-center gap-3 text-gray-300">
        <span
          className="h-8 w-8 animate-spin rounded-full border-2 border-gray-600 border-t-blue-500 motion-reduce:animate-none"
          aria-hidden="true"
        />
        <span className="text-sm">Loading…</span>
      </div>
    </div>
  );
}

/**
 * Route-level error boundary: a crash on one page is contained and reset when
 * the route changes, so sidebar/browser navigation still recovers the app
 * instead of bricking the whole tree (#7434).
 */
function RoutedContent() {
  const location = useLocation();
  return (
    <ErrorBoundary resetKeys={[location.pathname]} label={location.pathname}>
      <Suspense fallback={<PageLoadingFallback />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/simulation" element={<SimulationPage />} />
          <Route path="/tools/model-explorer" element={<ModelExplorerPage />} />
          {/* Phase 5: Tool pages (#1206) */}
          <Route path="/tools/putting-green" element={<PuttingGreenPage />} />
          <Route path="/tools/video-analyzer" element={<VideoAnalyzerPage />} />
          <Route path="/tools/data-explorer" element={<DataExplorerPage />} />
          <Route path="/tools/motion-capture" element={<MotionCapturePage />} />
          <Route path="/tools/terrain" element={<TerrainPage />} />
          <Route path="/tools/dataset" element={<DatasetGeneratorPage />} />
          <Route path="/tools/analysis" element={<AnalysisToolsPage />} />
          <Route path="/tools/character-builder" element={<CharacterBuilderPage />} />
          <Route
            path="/tools/canonical-core/estimation"
            element={<CanonicalCoreShellPage mode="estimation" />}
          />
          <Route
            path="/tools/canonical-core/comparison"
            element={<CanonicalCoreShellPage mode="comparison" />}
          />
          {/* Shot Tracer / ball-flight comparison (#7456) */}
          <Route path="/ball-flight" element={<BallFlightPage />} />
          {/* Chat (#3505): wires chat_ws backend into the UI */}
          <Route path="/chat" element={<ChatPage />} />
          {/* Settings (#7457): server-persisted preferences surface */}
          <Route path="/settings" element={<SettingsPage />} />
          {/* Catch-all 404 (#7430) — must stay last. */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}

function App() {
  const helpOpen = useUIStore((s) => s.helpOpen);
  const setHelpOpen = useUIStore((s) => s.setHelpOpen);

  // Load persisted web settings (font scale, simulation defaults) at app
  // start; server file is the source of truth, localStorage is a cache (#7457).
  useWebSettingsBootstrap();

  return (
    <BrowserRouter>
      {/* #7441: first focusable element — lets keyboard users jump past the
          per-page sidebars straight to the main content. */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:bg-blue-600 focus:text-white focus:px-3 focus:py-2 focus:rounded"
      >
        Skip to main content
      </a>
      <ScrollToTop />
      <RouteTitle />
      <ToastProvider>
        <RoutedContent />
        <DiagnosticsPanel />
        <HelpPanel isOpen={helpOpen} onClose={() => setHelpOpen(false)} />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
