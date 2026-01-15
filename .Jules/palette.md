## 2024-05-23 - [Inline Error Validation]
**Learning:** In vanilla JS visualizations, `aria-invalid` provides good semantic feedback but lacks explanatory power for sighted users and those needing specific guidance. Relying solely on `new Function` validation without exposing the error message leaves users guessing why their input is red.
**Action:** When validating complex user inputs (like math expressions), capture the specific error message (e.g., from `try-catch`) and display it in a dedicated `role="alert"` container linked via `aria-describedby`. This provides immediate, accessible, and actionable feedback.

## 2026-01-26 - [Custom Widget Accessibility]
**Learning:** Custom Qt widgets inheriting from `QFrame` (like clickable cards) are invisible to screen readers by default. They need explicit `accessibleName` and `accessibleDescription` properties to be discoverable, even if they handle focus correctly.
**Action:** Always set `setAccessibleName` and `setAccessibleDescription` for interactive custom widgets during initialization, ensuring dynamic status information is included in the description.

## 2026-02-12 - [Drag and Drop UX in Desktop Apps]
**Learning:** Adding drag-and-drop support to a main window is a high-value, low-effort "delight" feature, but it requires careful handling of MIME types and visual feedback (e.g., cursor changes) to feel native. Without `dragEnterEvent` filtering for specific file extensions (like `.c3d`), users might try to drop unsupported files, leading to silent failures or confusion.
**Action:** Always implement `dragEnterEvent` to filter compatible file types and provide immediate visual feedback (accepting the action) only when valid files are hovered. Combine this with a unified `load_file` method that handles both dialog-based and drop-based loading to ensure consistent security and validation logic.

## 2026-06-15 - [Micro-feedback in Desktop Apps]
**Learning:** In desktop GUI applications, immediate feedback for invisible actions (like "Copy to Clipboard") significantly reduces user uncertainty. A simple status bar message is often missed; temporarily changing the button's own state (text/icon) captures attention exactly where the user is looking.
**Action:** For "fire-and-forget" actions like copying to clipboard, implement a temporary "success state" on the triggering element (e.g., change "Copy" to "Copied!" with a checkmark) that reverts automatically after a short delay (e.g., 2s).

## 2026-06-15 - [Search Bar in Grid Layouts]
**Learning:** When displaying a large grid of items (like models or apps), users struggle to find specific items quickly without a search function. Adding a real-time filter improves discoverability significantly. However, implementing it in a grid layout requires clearing and rebuilding the grid to avoid gaps, rather than just hiding widgets.
**Action:** When adding search filtering to a `QGridLayout`, implement a robust rebuild method that re-flows the visible items into the grid coordinates from scratch, ensuring a dense, gap-free layout for the filtered results.
## 2024-05-23 - Web Button Feedback Race Conditions
**Learning:** When implementing temporary UI feedback (like 'Copied!' status), relying on capturing and restoring `textContent` is dangerous if the user can trigger the action again during the feedback window. It can lead to the temporary state being captured as the 'original' state, corrupting the UI.
**Action:** Always hardcode the restoration value or check current state before applying feedback in simple UI interactions, rather than dynamically capturing state.

## 2026-10-24 - [Micro-feedback in Desktop Apps]
**Learning:** Implementing the "Snapshot" button with self-reverting state ("Snapshot" -> "Copied!" -> "Snapshot") proved to be a simple yet highly effective way to provide confirmation for an invisible action (clipboard copy) without cluttering the UI with dialogs.
**Action:** Apply this pattern to all clipboard actions or background tasks that complete instantly but lack inherent visual side effects.

## 2026-10-25 - [Status Chip Contrast]
**Learning:** Using standard Bootstrap-like colors (Success Green, Info Cyan, Warning Orange) as background for status chips often fails WCAG AA contrast requirements when paired with white text. Green (#28a745) and Cyan (#17a2b8) specifically require black text to pass AAA/AA standards.
**Action:** When implementing status indicators using standard color palettes, always verify text contrast. Implement logic to switch between black and white text based on the background luminance, rather than defaulting to white.
