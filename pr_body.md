## Description

This PR addresses significant technical debt (техническая задолженность) identified in the repository, specifically around code duplication (DRY violations) and tight coupling (Orthogonality violations).

### Key Changes

- **Modular GUI Launcher**: Refactored the monolithic launcher into modular components.
- **Shared Utilities**: Created `src/shared/python/image_utils.py` and `scripts/script_utils.py` to consolidate recurring logic.
- **Parameterized Testing**: Improved test orthogonality and reduced boilerplate in engineering and analytical tests.
- **Standardized Script Lifecycle**: Unified entry points for maintenance scripts.

### Impact

- Reduced duplicate code by ~1.5k lines.
- Improved CI/CD resilience through standardized logging and error handling.
- Enhanced developer productivity by centralizing core utilities.

Part of a systematic architectural cleanup.
