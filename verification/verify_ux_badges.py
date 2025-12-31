import os

from playwright.sync_api import sync_playwright


def verify_ux() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the file directly from the filesystem
        cwd = os.getcwd()
        file_path = f"file://{cwd}/engines/physics_engines/pinocchio/python/double_pendulum_model/visualization/double_pendulum_web/index.html"
        page.goto(file_path)

        # Take a screenshot of the controls area where the badges are
        controls = page.locator(".controls")
        controls.screenshot(path="verification/ux_verification.png")

        print("Screenshot saved to verification/ux_verification.png")

        browser.close()


if __name__ == "__main__":
    verify_ux()
