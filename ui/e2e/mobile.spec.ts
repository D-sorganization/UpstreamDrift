import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { test, expect } from "@playwright/test";
import {
  assertClipViolationsAllowlisted,
  assertSmallTargetsAllowlisted,
  auditHorizontalClip,
  auditSmallPrimaryTargets,
  type MobileAllowlistFile,
} from "./support/mobileLayoutAudit";
import { installLauncherApiMocks } from "./support/mockLauncherApi";

const allowlist = JSON.parse(
  readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "mobile_clip_allowlist.json"),
    "utf-8",
  ),
) as MobileAllowlistFile;

const MOBILE_ROUTES = [
  "/",
  "/simulation",
  "/tools/model-explorer",
  "/tools/lift-baseline",
  "/tools/analysis",
  "/tools/golf-simulator",
] as const;

const VIEWPORTS = [
  { name: "phone", width: 390, height: 844 },
  { name: "tablet", width: 768, height: 1024 },
] as const;

test.describe("mobile layout smoke (#14283)", () => {
  test.beforeEach(async ({ page }) => {
    await installLauncherApiMocks(page);
  });

  for (const viewport of VIEWPORTS) {
    test.describe(`${viewport.name} ${viewport.width}x${viewport.height}`, () => {
      test.use({
        viewport: { width: viewport.width, height: viewport.height },
        isMobile: viewport.width < 500,
        hasTouch: true,
        deviceScaleFactor: 2,
      });

      for (const route of MOBILE_ROUTES) {
        test(`clip and touch targets ${route}`, async ({ page }) => {
          await page.goto(route, { waitUntil: "networkidle" });
          await expect(page.locator("#root")).toBeVisible();

          const clipViolations = await auditHorizontalClip(page, viewport.width);
          assertClipViolationsAllowlisted(route, clipViolations, allowlist);

          const smallTargets = await auditSmallPrimaryTargets(page, 44);
          assertSmallTargetsAllowlisted(route, smallTargets, allowlist);
        });
      }
    });
  }
});
