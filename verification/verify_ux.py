from playwright.sync_api import sync_playwright, expect
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Use absolute path to the HTML file
        url = f"file://{os.getcwd()}/engines/physics_engines/pinocchio/python/double_pendulum_model/visualization/double_pendulum_web/index.html"
        page.goto(url)

        # Check if the hint exists
        expect(page.locator("#math-hint")).to_be_visible()

        # Check if the torque inputs have aria-describedby
        tau1 = page.locator("#tau1")
        expect(tau1).to_have_attribute("aria-describedby", "math-hint")

        # Test validation logic
        # Default value is "0" which is valid
        # We need to trigger the input event to run validation
        tau1.fill("Math.sin(t)")
        # This should be valid
        expect(tau1).to_have_attribute("aria-invalid", "false")

        # Invalid input
        tau1.fill("Math.sin(t")
        # Should be invalid
        expect(tau1).to_have_attribute("aria-invalid", "true")

        # Take screenshot of invalid state
        page.screenshot(path="verification/verification.png")
        print("Verification successful")

if __name__ == "__main__":
    run()
