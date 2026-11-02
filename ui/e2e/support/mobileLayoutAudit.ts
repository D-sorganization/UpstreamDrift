import type { Page } from "@playwright/test";

/** Fingerprints for clipped nodes reported from the browser audit. */
export type ClipViolation = {
  tag: string;
  id: string;
  className: string;
  right: number;
  bottom: number;
};

export type SmallTargetViolation = {
  tag: string;
  role: string;
  name: string;
  width: number;
  height: number;
};

export type RouteAllowlist = {
  /** When true, any clip violation on this route is permitted (bootstrap only). */
  allowAllClipping?: boolean;
  /** When true, skip the 44px primary-target gate (bootstrap only). */
  allowAllSmallTargets?: boolean;
  clipViolations?: string[];
  smallTargets?: string[];
};

export type MobileAllowlistFile = {
  version: 1;
  routes: Record<string, RouteAllowlist>;
};

export function violationKey(v: ClipViolation): string {
  return `${v.tag}#${v.id}.${v.className}|r${Math.round(v.right)}`;
}

export function smallTargetKey(v: SmallTargetViolation): string {
  const label = v.name || v.role || "unnamed";
  return `${v.tag}:${label}|${Math.round(v.width)}x${Math.round(v.height)}`;
}

export async function auditHorizontalClip(
  page: Page,
  viewportWidth: number,
): Promise<ClipViolation[]> {
  return page.evaluate((width) => {
    const tol = 1;
    const violations: ClipViolation[] = [];

    const hasHorizontalScrollAncestor = (el: Element): boolean => {
      let node: Element | null = el.parentElement;
      while (node && node !== document.documentElement) {
        if (!(node instanceof HTMLElement)) {
          node = node.parentElement;
          continue;
        }
        const style = getComputedStyle(node);
        const scrollable =
          style.overflowX === "auto" ||
          style.overflowX === "scroll" ||
          style.overflow === "auto" ||
          style.overflow === "scroll";
        if (scrollable && node.scrollWidth > node.clientWidth + tol) {
          return true;
        }
        node = node.parentElement;
      }
      return false;
    };

    for (const el of document.querySelectorAll("body *")) {
      if (!(el instanceof HTMLElement)) {
        continue;
      }
      const style = getComputedStyle(el);
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        style.opacity === "0"
      ) {
        continue;
      }
      const rect = el.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) {
        continue;
      }
      if (rect.right <= width + tol) {
        continue;
      }
      if (hasHorizontalScrollAncestor(el)) {
        continue;
      }
      violations.push({
        tag: el.tagName.toLowerCase(),
        id: el.id,
        className: typeof el.className === "string" ? el.className : "",
        right: rect.right,
        bottom: rect.bottom,
      });
    }
    return violations;
  }, viewportWidth);
}

export async function auditSmallPrimaryTargets(
  page: Page,
  minSizePx: number,
): Promise<SmallTargetViolation[]> {
  return page.evaluate((minSize) => {
    const violations: SmallTargetViolation[] = [];
    const selectors =
      'button, a[href], input, select, textarea, [role="button"], [role="link"]';
    for (const el of document.querySelectorAll(selectors)) {
      if (!(el instanceof HTMLElement)) {
        continue;
      }
      const style = getComputedStyle(el);
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        style.pointerEvents === "none"
      ) {
        continue;
      }
      const rect = el.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) {
        continue;
      }
      if (rect.width >= minSize && rect.height >= minSize) {
        continue;
      }
      const name =
        el.getAttribute("aria-label") ||
        el.textContent?.trim().slice(0, 40) ||
        "";
      violations.push({
        tag: el.tagName.toLowerCase(),
        role: el.getAttribute("role") || "",
        name,
        width: rect.width,
        height: rect.height,
      });
    }
    return violations;
  }, minSizePx);
}

export function assertClipViolationsAllowlisted(
  route: string,
  observed: ClipViolation[],
  allowlist: MobileAllowlistFile,
): void {
  const entry = allowlist.routes[route];
  if (!entry) {
    throw new Error(`Missing mobile allowlist entry for route ${route}`);
  }
  if (entry.allowAllClipping) {
    return;
  }
  const allowed = new Set(entry.clipViolations ?? []);
  const unexpected = observed
    .map(violationKey)
    .filter((key) => !allowed.has(key));
  if (unexpected.length > 0) {
    throw new Error(
      `New horizontal clip violations on ${route}: ${unexpected.slice(0, 5).join(", ")}`,
    );
  }
}

export function assertSmallTargetsAllowlisted(
  route: string,
  observed: SmallTargetViolation[],
  allowlist: MobileAllowlistFile,
): void {
  const entry = allowlist.routes[route];
  if (!entry) {
    throw new Error(`Missing mobile allowlist entry for route ${route}`);
  }
  if (entry.allowAllSmallTargets) {
    return;
  }
  const allowed = new Set(entry.smallTargets ?? []);
  const unexpected = observed
    .map(smallTargetKey)
    .filter((key) => !allowed.has(key));
  if (unexpected.length > 0) {
    throw new Error(
      `New sub-44px primary targets on ${route}: ${unexpected.slice(0, 5).join(", ")}`,
    );
  }
}
