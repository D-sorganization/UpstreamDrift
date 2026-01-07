## 2024-05-23 - [Inline Error Validation]
**Learning:** In vanilla JS visualizations, `aria-invalid` provides good semantic feedback but lacks explanatory power for sighted users and those needing specific guidance. Relying solely on `new Function` validation without exposing the error message leaves users guessing why their input is red.
**Action:** When validating complex user inputs (like math expressions), capture the specific error message (e.g., from `try-catch`) and display it in a dedicated `role="alert"` container linked via `aria-describedby`. This provides immediate, accessible, and actionable feedback.

## 2026-01-26 - [Custom Widget Accessibility]
**Learning:** Custom Qt widgets inheriting from `QFrame` (like clickable cards) are invisible to screen readers by default. They need explicit `accessibleName` and `accessibleDescription` properties to be discoverable, even if they handle focus correctly.
**Action:** Always set `setAccessibleName` and `setAccessibleDescription` for interactive custom widgets during initialization, ensuring dynamic status information is included in the description.
