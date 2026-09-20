/**
 * Workspace Page — Task-oriented workspace view (ORG-06, #10516).
 *
 * Integrates with WorkspaceShell, providing responsive sidebar navigation
 * and bookmarkable task URLs (/workspaces/:slug).
 */

import { useParams, Navigate } from 'react-router';
import { getWorkspaceBySlug } from '@/types/workspaceNavigation';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import {
  WorkspaceSidebar,
  WorkspaceView,
} from '@/components/layout/WorkspaceNavigation';
import { canLaunchNativeWindow } from '@/api/webLaunch';

export function WorkspacePage() {
  const { slug } = useParams<{ slug: string }>();
  const workspace = slug ? getWorkspaceBySlug(slug) : undefined;

  if (!workspace) {
    return <Navigate to="/" replace />;
  }

  return (
    <WorkspaceShell
      leftPanel={<WorkspaceSidebar />}
      leftPanelLabel="Workspaces"
      className="bg-gray-900"
    >
      <WorkspaceView
        workspace={workspace}
        isTauri={canLaunchNativeWindow()}
      />
    </WorkspaceShell>
  );
}
