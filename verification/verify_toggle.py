
from playwright.sync_api import sync_playwright
import os

def run():
    # Use absolute path for file:// URL
    path = os.path.abspath('engines/physics_engines/pinocchio/python/double_pendulum_model/visualization/double_pendulum_web/index.html')
    url = f'file://{path}'

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # Take screenshot of initial state (Play button)
        page.screenshot(path='verification/initial_state.png')

        # Click Play/Pause button
        page.click('#play-pause')

        # Take screenshot of running state (Pause button)
        page.screenshot(path='verification/running_state.png')

        browser.close()

if __name__ == '__main__':
    run()
