## 2024-05-23 - [Inline Error Validation]
**Learning:** In vanilla JS visualizations, `aria-invalid` provides good semantic feedback but lacks explanatory power for sighted users and those needing specific guidance. Relying solely on `new Function` validation without exposing the error message leaves users guessing why their input is red.
**Action:** When validating complex user inputs (like math expressions), capture the specific error message (e.g., from `try-catch`) and display it in a dedicated `role="alert"` container linked via `aria-describedby`. This provides immediate, accessible, and actionable feedback.

## 2026-01-26 - [Custom Widget Accessibility]
**Learning:** Custom Qt widgets inheriting from `QFrame` (like clickable cards) are invisible to screen readers by default. They need explicit `accessibleName` and `accessibleDescription` properties to be discoverable, even if they handle focus correctly.
**Action:** Always set `setAccessibleName` and `setAccessibleDescription` for interactive custom widgets during initialization, ensuring dynamic status information is included in the description.

## 2026-02-12 - [Drag and Drop UX in Desktop Apps]
**Learning:** Adding drag-and-drop support to a main window is a high-value, low-effort "delight" feature, but it requires careful handling of MIME types and visual feedback (e.g., cursor changes) to feel native. Without `dragEnterEvent` filtering for specific file extensions (like `.c3d`), users might try to drop unsupported files, leading to silent failures or confusion.
**Action:** Always implement `dragEnterEvent` to filter compatible file types and provide immediate visual feedback (accepting the action) only when valid files are hovered. Combine this with a unified `load_file` method that handles both dialog-based and drop-based loading to ensure consistent security and validation logic.
