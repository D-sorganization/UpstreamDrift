
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

        # Take initial screenshot
        page.screenshot(path='verification/before_click.png')
        print('Initial screenshot taken')

        # Click the play-pause button
        # First click: Start -> Pause
        page.click('#play-pause')
        page.wait_for_timeout(100) # Wait for UI update
        page.screenshot(path='verification/after_first_click.png')
        print('After first click (Start -> Pause) screenshot taken')

        # Second click: Pause -> Resume
        page.click('#play-pause')
        page.wait_for_timeout(100)
        page.screenshot(path='verification/after_second_click.png')
        print('After second click (Pause -> Resume) screenshot taken')

        browser.close()

if __name__ == '__main__':
    run()
