/**
 * Workspace Navigation Components for React and Tauri (ORG-06, #10516).
 *
 * Implements accessible, responsive sidebar navigation, breadcrumbs,
 * and workspace task views adhering to shared catalog metadata.
 */

import { useCallback } from 'react';
import { NavLink, Link, useNavigate } from 'react-router';
import {
  Camera,
  Cpu,
  Flag,
  Sliders,
  BarChart3,
  Code2,
  Home,
  Star,
  History,
  Grid,
  Monitor,
  ExternalLink,
} from 'lucide-react';
import {
  PRIMARY_WORKSPACES,
  SECONDARY_WORKSPACES,
  type WorkspaceMetadata,
} from '@/types/workspaceNavigation';
import {
  resolveWorkspaceToolAction,
  type WorkspaceToolAction,
} from '@/api/capabilityAdapter';

export interface WorkspaceSidebarProps {
  onNavigate?: (workspaceId: string) => void;
  className?: string;
}

const WORKSPACE_ICONS: Record<string, React.ComponentType<{ className?: string; 'aria-hidden'?: boolean }>> = {
  camera: Camera,
  computer: Cpu,
  sports_golf: Flag,
  build: Sliders,
  assessment: BarChart3,
  code: Code2,
  grid_view: Grid,
  star: Star,
  history: History,
};

export function WorkspaceSidebar({ onNavigate, className = '' }: WorkspaceSidebarProps) {
  const handleNavClick = useCallback(
    (workspaceId: string) => {
      onNavigate?.(workspaceId);
      // Focus recovery to main content for screen readers / keyboard users (#7441)
      const mainEl = document.getElementById('main-content');
      if (mainEl) {
        mainEl.focus();
      }
    },
    [onNavigate],
  );

  return (
    <nav
      aria-label="Workspace Navigation"
      className={`flex flex-col gap-1 p-3 text-sm text-gray-300 ${className}`.trim()}
    >
      {/* Home link */}
      <NavLink
        to="/"
        end
        onClick={() => handleNavClick('home')}
        className={({ isActive }) =>
          `flex items-center gap-2.5 px-3 py-2 rounded-lg font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
            isActive
              ? 'bg-blue-600/30 text-blue-200 border border-blue-500/40'
              : 'hover:bg-gray-700/60 hover:text-white'
          }`
        }
      >
        <Home className="w-4 h-4 text-gray-400" aria-hidden={true} />
        <span>Home</span>
      </NavLink>

      <div className="pt-2 pb-1 px-3 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
        Task Workspaces
      </div>

      {/* Five Primary Task Workspaces */}
      {Object.values(PRIMARY_WORKSPACES).map((ws) => {
        const IconComponent = WORKSPACE_ICONS[ws.iconName] || Grid;
        return (
          <NavLink
            key={ws.id}
            to={ws.route}
            onClick={() => handleNavClick(ws.id)}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2 rounded-lg font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                isActive
                  ? 'bg-blue-600/30 text-blue-200 border border-blue-500/40'
                  : 'hover:bg-gray-700/60 hover:text-white'
              }`
            }
          >
            <IconComponent className="w-4 h-4 text-blue-400" aria-hidden={true} />
            <span className="truncate">{ws.title}</span>
          </NavLink>
        );
      })}

      <div className="pt-3 pb-1 px-3 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
        Utilities & Dev
      </div>

      {/* Secondary Workspaces */}
      {Object.values(SECONDARY_WORKSPACES).map((ws) => {
        const IconComponent = WORKSPACE_ICONS[ws.iconName] || Grid;
        return (
          <NavLink
            key={ws.id}
            to={ws.route}
            onClick={() => handleNavClick(ws.id)}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2 rounded-lg font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                isActive
                  ? 'bg-blue-600/30 text-blue-200 border border-blue-500/40'
                  : 'hover:bg-gray-700/60 hover:text-white'
              }`
            }
          >
            <IconComponent className="w-4 h-4 text-gray-400" aria-hidden={true} />
            <span className="truncate">{ws.title}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}

export interface WorkspaceBreadcrumbProps {
  workspaceId: string;
  toolName: string;
  className?: string;
}

export function WorkspaceBreadcrumb({
  workspaceId,
  toolName,
  className = '',
}: WorkspaceBreadcrumbProps) {
  const ws = PRIMARY_WORKSPACES[workspaceId] || SECONDARY_WORKSPACES[workspaceId];
  const wsTitle = ws ? ws.title : 'Workspace';
  const wsRoute = ws ? ws.route : '/';

  return (
    <nav
      aria-label="Breadcrumb"
      className={`flex items-center gap-2 text-xs text-gray-400 px-4 py-2 bg-gray-850 border-b border-gray-700/60 ${className}`.trim()}
    >
      <Link
        to="/"
        className="hover:text-blue-400 transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-blue-400 rounded"
      >
        Home
      </Link>
      <span aria-hidden="true" className="text-gray-600">
        /
      </span>
      <Link
        to={wsRoute}
        className="hover:text-blue-400 transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-blue-400 rounded"
      >
        {wsTitle}
      </Link>
      <span aria-hidden="true" className="text-gray-600">
        /
      </span>
      <span aria-current="page" className="text-gray-200 font-medium">
        {toolName}
      </span>
    </nav>
  );
}

export interface WorkspaceViewProps {
  workspace: WorkspaceMetadata;
  isTauri?: boolean;
}

export function WorkspaceView({ workspace, isTauri }: WorkspaceViewProps) {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col gap-6 p-6 overflow-y-auto">
      <header className="flex flex-col gap-1 border-b border-gray-700 pb-4">
        <h1 className="text-2xl font-bold text-white tracking-tight">{workspace.title}</h1>
        <p className="text-sm text-gray-400">{workspace.description}</p>
      </header>

      <section
        aria-label={`${workspace.title} Tools`}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
      >
        {workspace.memberToolIds.map((toolId) => {
          const action: WorkspaceToolAction = resolveWorkspaceToolAction(toolId, {
            isTauri,
          });

          return (
            <div
              key={toolId}
              className="flex flex-col justify-between p-4 bg-gray-800/80 border border-gray-700/80 rounded-xl shadow-md hover:border-gray-600 transition-colors"
            >
              <div>
                <h3 className="text-base font-semibold text-white mb-1">{action.title}</h3>
                <p className="text-xs text-gray-400 mb-3">
                  {action.kind === 'native-only-explanation'
                    ? action.explanation
                    : `Access and configure ${action.title} in the ${workspace.title} task space.`}
                </p>
              </div>

              <div className="pt-2 border-t border-gray-700/40 flex items-center justify-between gap-2">
                {action.kind === 'navigate' && (
                  <button
                    type="button"
                    onClick={() => navigate(action.route)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
                  >
                    Open Tool
                    <ExternalLink className="w-3.5 h-3.5" aria-hidden={true} />
                  </button>
                )}

                {action.kind === 'native-launch' && (
                  <button
                    type="button"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
                  >
                    Launch Window
                    <Monitor className="w-3.5 h-3.5" aria-hidden={true} />
                  </button>
                )}

                {action.kind === 'native-only-explanation' && (
                  <div className="flex flex-col gap-1.5 w-full">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-amber-500/20 text-amber-300 border border-amber-600/40 w-fit">
                      <Monitor className="w-3 h-3" aria-hidden={true} />
                      {action.badge}
                    </span>
                    {action.alternativeRoute && (
                      <button
                        type="button"
                        onClick={() => navigate(action.alternativeRoute!)}
                        className="text-xs text-blue-400 hover:underline text-left"
                      >
                        Try web alternative →
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
