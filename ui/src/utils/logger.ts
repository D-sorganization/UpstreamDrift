/**
 * Lightweight client logger (issue #7434).
 *
 * Wraps the console so diagnostic noise is suppressed in production builds
 * while remaining available during development and tests. Use this instead of
 * calling `console.*` directly in `ui/src` runtime code.
 *
 * Vite exposes the build mode via `import.meta.env.PROD` (a static boolean
 * replaced at build time), so the calls tree-shake away in production.
 */

const isProd = import.meta.env.PROD;

export const logger = {
  error(...args: unknown[]): void {
    if (!isProd) {
      console.error(...args);
    }
  },
  warn(...args: unknown[]): void {
    if (!isProd) {
      console.warn(...args);
    }
  },
  info(...args: unknown[]): void {
    if (!isProd) {
      console.info(...args);
    }
  },
};
