/**
 * About/onboarding API client (issue #7459, parity G11).
 *
 * Thin typed wrappers around GET /api/v1/about and
 * GET /api/v1/about/onboarding. The backend shares its version-resolution
 * chain with the desktop About dialog, and the onboarding card copy is
 * single-sourced from src/config/onboarding_cards.json.
 */

import { apiFetch } from './fetch';

export interface AboutInfo {
  app_name: string;
  app_version: string;
  python_version: string;
  platform: string;
  git_commit: string | null;
  dependencies: Record<string, string>;
  links: {
    repository: string;
    report_bug: string;
    user_guide: string;
  };
}

export interface OnboardingCard {
  id?: string;
  title: string;
  body: string;
  link_text: string;
  link_url: string;
}

export interface OnboardingCopy {
  header: string;
  subtitle: string;
  cards: OnboardingCard[];
}

export function fetchAboutInfo(): Promise<AboutInfo> {
  return apiFetch<AboutInfo>('/api/v1/about');
}

export function fetchOnboardingCopy(): Promise<OnboardingCopy> {
  return apiFetch<OnboardingCopy>('/api/v1/about/onboarding');
}
