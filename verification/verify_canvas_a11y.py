import os

from playwright.sync_api import expect, sync_playwright


def run():
    # Use absolute path for file URL
    file_path = os.path.abspath(
        "engines/physics_engines/pinocchio/python/double_pendulum_model/visualization/double_pendulum_web/index.html"
    )
    url = f"file://{file_path}"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # 1. Load the page
        print(f"Loading {url}")
        page.goto(url)

        # 2. Check that the canvas is focusable (tabindex="0")
        canvas = page.locator("#canvas")
        expect(canvas).to_have_attribute("tabindex", "0")
        print("Verified: Canvas has tabindex='0'")

        # 3. Check aria-describedby
        expect(canvas).to_have_attribute("aria-describedby", "shortcuts")
        print("Verified: Canvas has aria-describedby='shortcuts'")

        # 4. Focus the canvas and verify focus state
        canvas.focus()

        # Check if focus is on canvas
        expect(canvas).to_be_focused()
        print("Verified: Canvas is focusable and focused")

        # 5. Take screenshot of focused canvas to verify visual ring
        screenshot_path = "verification/canvas_focus.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        # 6. Verify keyboard interaction (Space)
        # We need to see if it triggers the announcer
        # First, ensure simulation is initially paused or running?
        # Default state: Reset calls pause() -> updateButtonStates(false)
        # Wait, app.js calls reset() at the end.

        # Let's press Space on the canvas
        page.keyboard.press("Space")

        # The announcer should say "Simulation started"
        announcer = page.locator("#status-announcer")
        expect(announcer).to_have_text("Simulation started")
        print("Verified: Space on canvas started simulation")

        browser.close()


if __name__ == "__main__":
    run()
