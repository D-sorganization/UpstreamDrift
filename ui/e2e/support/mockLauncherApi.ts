import type { Page } from "@playwright/test";

/** Minimal launcher manifest so dashboard routes render without a live API. */
const MINIMAL_MANIFEST = {
  version: "1.0.0",
  description: "playwright-fixture",
  launcher_csrf_token: "playwright-token",
  launcher_csrf_header: "X-Launcher-CSRF-Token",
  tiles: [
    {
      id: "chat_assistant",
      name: "Sidekick",
      description: "Chat assistant",
      category: "tool",
      type: "special_app",
      logo: "sidekick.svg",
      status: "ready",
      web_route: "/chat",
    },
  ],
  category_labels: {
    tool: "Tools",
    physics_engine: "Physics Engines",
  },
};

export async function installLauncherApiMocks(page: Page): Promise<void> {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith("/api/launcher/manifest")) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MINIMAL_MANIFEST),
      });
      return;
    }
    if (url.pathname.includes("/api/launcher/logos/")) {
      await route.fulfill({
        status: 200,
        contentType: "image/svg+xml",
        body: "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'/>",
      });
      return;
    }
    if (url.pathname.includes("/api/web/settings")) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ theme: "dark" }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "{}",
    });
  });
}
