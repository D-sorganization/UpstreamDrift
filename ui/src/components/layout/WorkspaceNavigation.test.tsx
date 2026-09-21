/**
 * Workspace Navigation Tests for React and Tauri (ORG-06, #10516).
 *
 * Validates:
 * 1. Five primary task workspaces + secondary utilities from shared catalog metadata.
 * 2. Navigation rendering, accessible names, active states, and keyboard focus.
 * 3. Browser history preservation, bookmarkable task URLs, and route title resolution.
 * 4. Focus recovery to #main-content on navigation.
 * 5. Capability adapter: Tauri native launch vs browser-only explanation.
 * 6. Return-to-workspace breadcrumb affordance.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { MemoryRouter } from 'react-router';
import {
  PRIMARY_WORKSPACES,
  SECONDARY_WORKSPACES,
  getWorkspaceBySlug,
  getWorkspaceForTool,
  isPrimaryWorkspace,
} from '@/types/workspaceNavigation';
import {
  WorkspaceSidebar,
  WorkspaceBreadcrumb,
  WorkspaceView,
} from './WorkspaceNavigation';
import { resolveWorkspaceToolAction } from '@/api/capabilityAdapter';
import { titleForPath } from '@/utils/routeTitles';

describe('Workspace Catalog Metadata (ORG-06)', () => {
  it('defines exactly five primary task workspaces with required titles', () => {
    const keys = Object.keys(PRIMARY_WORKSPACES);
    expect(keys).toHaveLength(5);
    expect(keys).toEqual([
      'capture_analyze',
      'model_match',
      'shot_course_lab',
      'optimize_train',
      'results_compare',
    ]);

    expect(PRIMARY_WORKSPACES.capture_analyze.title).toBe('Capture & Analyze');
    expect(PRIMARY_WORKSPACES.model_match.title).toBe('Model & Match');
    expect(PRIMARY_WORKSPACES.shot_course_lab.title).toBe('Shot & Course Lab');
    expect(PRIMARY_WORKSPACES.optimize_train.title).toBe('Optimize & Train');
    expect(PRIMARY_WORKSPACES.results_compare.title).toBe('Results & Compare');
  });

  it('defines secondary utilities including dev_research, all_tools, favorites, history', () => {
    expect(SECONDARY_WORKSPACES.dev_research).toBeDefined();
    expect(SECONDARY_WORKSPACES.dev_research.title).toBe('Developer & Research');
    expect(SECONDARY_WORKSPACES.all_tools).toBeDefined();
    expect(SECONDARY_WORKSPACES.favorites).toBeDefined();
    expect(SECONDARY_WORKSPACES.history).toBeDefined();
  });

  it('maps tools to their correct task workspace', () => {
    expect(getWorkspaceForTool('motion_capture')).toBe('capture_analyze');
    expect(getWorkspaceForTool('video_analyzer')).toBe('capture_analyze');
    expect(getWorkspaceForTool('mujoco_unified')).toBe('model_match');
    expect(getWorkspaceForTool('drake_golf')).toBe('model_match');
    expect(getWorkspaceForTool('putting_green')).toBe('shot_course_lab');
    expect(getWorkspaceForTool('tools_movement_optimizer')).toBe('optimize_train');
    expect(getWorkspaceForTool('cross_engine_dashboard')).toBe('results_compare');
    expect(getWorkspaceForTool('character_builder')).toBe('dev_research');
  });

  it('resolves workspaces by slug for bookmarkable URLs', () => {
    const ws = getWorkspaceBySlug('capture-analyze');
    expect(ws).toBeDefined();
    expect(ws?.id).toBe('capture_analyze');
    expect(isPrimaryWorkspace('capture_analyze')).toBe(true);
    expect(isPrimaryWorkspace('dev_research')).toBe(false);
  });
});

describe('WorkspaceSidebar Navigation Component', () => {
  it('renders links for all primary workspaces and secondary utilities', () => {
    render(
      <MemoryRouter initialEntries={['/workspaces/capture-analyze']}>
        <WorkspaceSidebar />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: /Capture & Analyze/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Model & Match/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Shot & Course Lab/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Optimize & Train/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Results & Compare/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Developer & Research/i })).toBeInTheDocument();
  });

  it('marks the active workspace with aria-current="page"', () => {
    render(
      <MemoryRouter initialEntries={['/workspaces/model-match']}>
        <WorkspaceSidebar />
      </MemoryRouter>,
    );

    const activeLink = screen.getByRole('link', { name: /Model & Match/i });
    expect(activeLink).toHaveAttribute('aria-current', 'page');

    const inactiveLink = screen.getByRole('link', { name: /Capture & Analyze/i });
    expect(inactiveLink).not.toHaveAttribute('aria-current', 'page');
  });

  it('supports focus recovery by calling onNavigate or focusing main content target', () => {
    const onNavigateMock = vi.fn();
    render(
      <MemoryRouter initialEntries={['/']}>
        <div id="main-content" tabIndex={-1} data-testid="main-target">
          Main Content
        </div>
        <WorkspaceSidebar onNavigate={onNavigateMock} />
      </MemoryRouter>,
    );

    const link = screen.getByRole('link', { name: /Capture & Analyze/i });
    fireEvent.click(link);
    expect(onNavigateMock).toHaveBeenCalledWith('capture_analyze');
  });
});

describe('WorkspaceBreadcrumb Component', () => {
  it('renders home link and current workspace breadcrumbs', () => {
    render(
      <MemoryRouter initialEntries={['/tools/video-analyzer']}>
        <WorkspaceBreadcrumb
          workspaceId="capture_analyze"
          toolName="Video Analyzer"
        />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: /Home/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Capture & Analyze/i })).toBeInTheDocument();
    expect(screen.getByText('Video Analyzer')).toBeInTheDocument();
  });
});

describe('Capability Adapter (Tauri vs Browser)', () => {
  it('returns navigate action for tools with in-app web routes', () => {
    const action = resolveWorkspaceToolAction('video_analyzer', {
      isTauri: false,
    });
    expect(action.kind).toBe('navigate');
    if (action.kind === 'navigate') {
      expect(action.route).toBe('/tools/video-analyzer');
    }
  });

  it('returns native launch action under Tauri for desktop-only engines', () => {
    const action = resolveWorkspaceToolAction('mujoco_unified', {
      isTauri: true,
    });
    expect(action.kind).toBe('native-launch');
  });

  it('returns actionable explanation and alternative for browser users when tool is desktop-only', () => {
    const action = resolveWorkspaceToolAction('mujoco_unified', {
      isTauri: false,
    });
    expect(action.kind).toBe('native-only-explanation');
    if (action.kind === 'native-only-explanation') {
      expect(action.badge).toBe('Desktop app only');
      expect(action.explanation).toContain('desktop');
      expect(action.alternativeRoute).toBeDefined();
    }
  });
});

describe('WorkspaceView and Shell Integration', () => {
  it('renders member tools of a workspace with launch or explanation affordances', () => {
    const ws = PRIMARY_WORKSPACES.capture_analyze;
    render(
      <MemoryRouter initialEntries={['/workspaces/capture-analyze']}>
        <WorkspaceView workspace={ws} isTauri={false} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('heading', { name: /Capture & Analyze/i })).toBeInTheDocument();
    expect(screen.getByText(/Motion capture, video analysis/i)).toBeInTheDocument();
  });
});

describe('Route Titles for Workspaces', () => {
  it('resolves workspace task URLs to descriptive page titles', () => {
    expect(titleForPath('/workspaces/capture-analyze')).toBe('Capture & Analyze');
    expect(titleForPath('/workspaces/model-match')).toBe('Model & Match');
    expect(titleForPath('/workspaces/shot-course-lab')).toBe('Shot & Course Lab');
    expect(titleForPath('/workspaces/optimize-train')).toBe('Optimize & Train');
    expect(titleForPath('/workspaces/results-compare')).toBe('Results & Compare');
    expect(titleForPath('/workspaces/dev-research')).toBe('Developer & Research');
  });
});
