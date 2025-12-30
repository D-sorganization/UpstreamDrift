import os

from playwright.sync_api import expect, sync_playwright


def run() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the file directly from disk
        cwd = os.getcwd()
        filepath = os.path.join(
            cwd,
            "engines/physics_engines/pinocchio/python/double_pendulum_model/visualization/double_pendulum_web/index.html",
        )
        page.goto(f"file://{filepath}")

        # 1. Verify tooltip on Start button (initial state)
        start_btn = page.get_by_role("button", name="Start")
        title_attr = start_btn.get_attribute("title")
        print(f"Initial title: {title_attr}")
        assert title_attr == "Start simulation (Space)"

        # 2. Start the simulation
        start_btn.click()
        page.wait_for_timeout(500)  # Let it run for a bit

        # 3. Verify tooltip changes to Resume after pause
        # Click pause
        pause_btn = page.get_by_role("button", name="Pause")
        pause_btn.click()

        # Now Start button should be Resume
        resume_btn = page.get_by_role("button", name="Resume")
        expect(resume_btn).to_be_visible()

        new_title = resume_btn.get_attribute("title")
        print(f"Resume title: {new_title}")
        assert new_title == "Resume simulation (Space)"

        # 4. Verify keyboard exclusion
        # Focus a text input
        input_field = page.get_by_label("Shoulder torque f(t)")
        input_field.focus()
        input_field.fill("")  # Clear it

        # Press Space
        page.keyboard.press("Space")

        # Check if simulation started (it should NOT have started)
        # If space triggered start, the Resume button would become hidden/disabled and Pause enabled
        # But we are paused. If space worked, we would be running.
        # Check Pause button state. If running, Pause is enabled. If paused, Pause is disabled.
        # We expect it to remain disabled (paused) because Space was swallowed by input

        expect(pause_btn).to_be_disabled()
        print("Space in input did not trigger simulation start (Correct)")

        # 5. Verify Space on Button (previously conflicting)
        # Focus Reset button
        reset_btn = page.get_by_role("button", name="Reset")
        reset_btn.focus()

        # Press Space. This should trigger the button click (Reset)
        # Reset sets time to 0, so Resume button becomes Start button
        page.keyboard.press("Space")

        expect(page.get_by_role("button", name="Start")).to_be_visible()
        print("Space on Reset button triggered Reset (Correct)")

        page.screenshot(path="verification/app_state.png")
        print("Verification complete")
        browser.close()


if __name__ == "__main__":
    run()
