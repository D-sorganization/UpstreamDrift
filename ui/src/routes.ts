/**
 * Canonical React routes table for UpstreamDrift web/Tauri frontend (ORG-04, #10513).
 *
 * Single source of truth for in-app route paths, keeping route-mode launcher
 * contracts synchronized with actual router definitions.
 */

export const KNOWN_APP_ROUTES = [
    '/',
    '/simulation',
    '/tools/model-explorer',
    '/tools/impact-explorer',
    '/tools/putting-green',
    '/tools/video-analyzer',
    '/tools/data-explorer',
    '/tools/motion-capture',
    '/tools/terrain',
    '/tools/dataset',
    '/tools/analysis',
    '/tools/character-builder',
    '/tools/canonical-core/estimation',
    '/tools/canonical-core/comparison',
    '/ball-flight',
    '/tools/swing-objective-lab',
    '/tools/golf-simulator',
    '/chat',
    '/settings',
] as const;

export type AppRoute = (typeof KNOWN_APP_ROUTES)[number];

export function isKnownAppRoute(route: string): route is AppRoute {
    return (KNOWN_APP_ROUTES as readonly string[]).includes(route);
}
