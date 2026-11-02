import { useState, type ReactNode } from 'react';
import { useInRouterContext, Link } from 'react-router';
import { ChevronLeft, ChevronRight, ArrowRight, AlertCircle, Layers } from 'lucide-react';
import { PRIMARY_WORKSPACES, SECONDARY_WORKSPACES, getWorkspaceForTool, getWorkspaceById } from '@/types/workspaceNavigation';
import { WorkspaceShell } from './WorkspaceShell';

export interface NavRailStep {
  id: string;
  label: string;
  active?: boolean;
  disabled?: boolean;
  icon?: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean | 'true' | 'false' }>;
  onClick?: () => void;
  href?: string;
}

export interface NextStepConfig {
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  href?: string;
}

export interface ToolShellProps {
  workspaceId?: string;
  toolId?: string;
  toolName: string;
  title?: string;
  description?: string;
  breadcrumb?: ReactNode;
  actions?: ReactNode;
  primaryActionDisabledReason?: string;
  navRailSteps?: NavRailStep[];
  initialNavCollapsed?: boolean;
  nextStep?: NextStepConfig | ReactNode;
  leftPanel?: ReactNode;
  rightPanel?: ReactNode;
  bottomPanel?: ReactNode;
  leftPanelLabel?: string;
  rightPanelLabel?: string;
  leftPanelWidthClass?: string;
  children: ReactNode;
  className?: string;
}

function SafeLink({ to, className, children }: { to: string; className?: string; children: ReactNode }) {
  return useInRouterContext() ? <Link to={to} className={className}>{children}</Link> : <a href={to} className={className}>{children}</a>;
}

/** Canonical ToolShell frame with persistent navigation, breadcrumbs, and next step card (UD #14260). */
export function ToolShell({
  workspaceId,
  toolId,
  toolName,
  title,
  description,
  breadcrumb,
  actions,
  primaryActionDisabledReason,
  navRailSteps,
  initialNavCollapsed = false,
  nextStep,
  leftPanel,
  rightPanel,
  bottomPanel,
  leftPanelLabel = 'Controls',
  rightPanelLabel = 'Details',
  leftPanelWidthClass,
  children,
  className = '',
}: ToolShellProps) {
  const [navCollapsed, setNavCollapsed] = useState(initialNavCollapsed);
  const resolvedWsId = workspaceId ?? (toolId ? getWorkspaceForTool(toolId) ?? undefined : undefined);
  const ws = resolvedWsId ? getWorkspaceById(resolvedWsId) ?? PRIMARY_WORKSPACES[resolvedWsId] ?? SECONDARY_WORKSPACES[resolvedWsId] : undefined;
  const wsTitle = ws?.title ?? 'Workspace';
  const wsRoute = ws?.route ?? '/';

  const breadcrumbNode = breadcrumb ?? (
    <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-xs text-gray-400 px-4 py-2 border-b border-gray-800 bg-gray-900/90">
      <SafeLink to="/" className="hover:text-blue-400 transition-colors rounded">Home</SafeLink>
      <span aria-hidden="true" className="text-gray-600">/</span>
      <SafeLink to={wsRoute} className="hover:text-blue-400 transition-colors rounded">{wsTitle}</SafeLink>
      <span aria-hidden="true" className="text-gray-600">/</span>
      <span aria-current="page" className="text-gray-200 font-medium">{toolName}</span>
    </nav>
  );

  const nextStepNode = nextStep && (
    typeof nextStep === 'object' && 'title' in nextStep ? (
      <section data-testid="next-step-card" aria-label="Next Step" className="my-6 mx-4 p-4 rounded-lg border border-blue-500/30 bg-blue-950/20 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700/40">Next Step</span>
            <h3 className="text-sm font-semibold text-gray-100 truncate">{(nextStep as NextStepConfig).title}</h3>
          </div>
          {(nextStep as NextStepConfig).description && <p className="text-xs text-gray-300">{(nextStep as NextStepConfig).description}</p>}
        </div>
        {(nextStep as NextStepConfig).actionLabel && (
          (nextStep as NextStepConfig).href ? (
            <SafeLink to={(nextStep as NextStepConfig).href!} className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors">
              <span>{(nextStep as NextStepConfig).actionLabel}</span>
              <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
            </SafeLink>
          ) : (
            <button type="button" onClick={(nextStep as NextStepConfig).onAction} className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors">
              <span>{(nextStep as NextStepConfig).actionLabel}</span>
              <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
            </button>
          )
        )}
      </section>
    ) : <section data-testid="next-step-card" aria-label="Next Step" className="my-6 mx-4">{nextStep}</section>
  );

  const toolHeader = (
    <div className="flex flex-col border-b border-gray-800 bg-gray-900">
      {breadcrumbNode}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5">
        <div className="min-w-0">
          <h1 className="text-lg font-bold text-gray-100 tracking-tight">{title ?? toolName}</h1>
          {description && <p className="text-xs text-gray-400 mt-0.5 line-clamp-1">{description}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          {primaryActionDisabledReason && (
            <div role="status" data-testid="primary-action-disabled-reason" className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-amber-950/70 border border-amber-800 text-amber-300 font-medium">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" aria-hidden="true" />
              <span>{primaryActionDisabledReason}</span>
            </div>
          )}
          {actions}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`flex h-screen w-full bg-gray-950 overflow-hidden ${className}`.trim()}>
      <nav aria-label="Tool Navigation Rail" className={`flex flex-col border-r border-gray-800 bg-gray-900 transition-all duration-200 flex-shrink-0 ${navCollapsed ? 'w-12' : 'w-48'}`}>
        <div className="flex items-center justify-between p-2 border-b border-gray-800">
          {!navCollapsed && <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-1">Steps</span>}
          <button type="button" data-testid="nav-rail-toggle" aria-label={navCollapsed ? 'Expand navigation rail' : 'Collapse navigation rail'} aria-expanded={!navCollapsed} onClick={() => setNavCollapsed(!navCollapsed)} className="inline-flex items-center justify-center min-h-[44px] min-w-[44px] rounded hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400">
            {navCollapsed ? <ChevronRight className="w-4 h-4" aria-hidden="true" /> : <ChevronLeft className="w-4 h-4" aria-hidden="true" />}
          </button>
        </div>
        <ul className="flex flex-col gap-1 p-1.5 flex-1 overflow-y-auto">
          {navRailSteps && navRailSteps.length > 0 ? (
            navRailSteps.map((step, idx) => {
              const IconComp = step.icon ?? Layers;
              const item = (
                <div className={`flex items-center gap-2.5 px-2 py-2 rounded text-xs transition-colors ${step.active ? 'bg-blue-600/30 text-blue-200 border border-blue-500/40 font-medium' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'} ${step.disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}>
                  <IconComp className="w-4 h-4 flex-shrink-0" aria-hidden={true} />
                  {!navCollapsed && <span className="truncate">{idx + 1}. {step.label}</span>}
                </div>
              );
              return (
                <li key={step.id}>
                  {step.href && !step.disabled ? <SafeLink to={step.href}>{item}</SafeLink> : <button type="button" disabled={step.disabled} onClick={step.onClick} className="w-full text-left focus:outline-none rounded">{item}</button>}
                </li>
              );
            })
          ) : (
            <li>
              <div className="flex items-center gap-2.5 px-2 py-2 rounded text-xs bg-blue-600/30 text-blue-200 border border-blue-500/40 font-medium">
                <Layers className="w-4 h-4 flex-shrink-0" aria-hidden={true} />
                {!navCollapsed && <span className="truncate">{toolName}</span>}
              </div>
            </li>
          )}
        </ul>
      </nav>
      <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        {toolHeader}
        <WorkspaceShell leftPanel={leftPanel} rightPanel={rightPanel} bottomPanel={bottomPanel} leftPanelLabel={leftPanelLabel} rightPanelLabel={rightPanelLabel} leftPanelWidthClass={leftPanelWidthClass} className="h-full flex-1">
          <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-y-auto">
            {children}
            {nextStepNode}
          </div>
        </WorkspaceShell>
      </div>
    </div>
  );
}
