# About/version info and first-run onboarding missing from web app

## Gap (PyQt6 = model)

Desktop has an About dialog (`src/launchers/about_dialog.py` — live versions of Python, Qt, NumPy, ezc3d, UpstreamDrift; links to user guide and bug reporting) and a first-run onboarding dialog (`src/launchers/onboarding_dialog.py` — what-is-this, install engines, select & launch, "don't show again"). The web app has neither: no version visibility anywhere (which also makes bug reports from web users unactionable) and no first-run orientation.

## Proposed fix

1. `GET /api/v1/about` returning app version (VERSION file → importlib.metadata fallback, same resolution chain as the desktop dialog), Python version, key dependency versions, git commit; web About modal/page rendering it plus ui package version and links (user guide, report bug).
2. First-run onboarding: lightweight web variant (3 cards mirroring desktop content), dismissal persisted via the settings endpoint (settings parity issue) so it doesn't reappear per-browser.
3. Keep content single-sourced where possible: onboarding card copy could live in a shared JSON consumed by both UIs (same pattern as `launcher_manifest.json`).

## Acceptance criteria

- [ ] Web user can see exact backend+frontend versions (and include them in bug reports)
- [ ] Onboarding shows once, dismissal persists server-side
- [ ] Version resolution shared with the desktop About dialog (one implementation)

## References

- `src/launchers/about_dialog.py`, `src/launchers/onboarding_dialog.py`
- Related: #7432 (branding/titles) from the UI/UX epic — adjacent, not duplicate
