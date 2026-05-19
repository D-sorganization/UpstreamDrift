# EPIC: AffineDrift Website & Offline Documentation Integration

## Overview

As the UpstreamDrift software suite scales, providing robust, high-quality, and instant access to documentation becomes critical. The AffineDrift website already hosts comprehensive articles, theory, and tutorials that complement the software. Relying on an active internet connection limits the utility of the software in field environments, secure networks, or during deployment.

This Epic aims to bundle the complete AffineDrift website into the application as a localized, offline resource. By utilizing a localized web bundle and an embedded WebView within the Library architecture, users will experience instantaneous access to theory and tutorials directly adjacent to the models they are running.

## Goals

1. **Local Documentation Bundling**: Introduce a build step to export the AffineDrift static site (HTML, CSS, JS, Images) into a localized `vendor/docs` folder during the CI/CD release process.
2. **Embedded WebView Integration**: Enhance the newly established "Library" tab architecture (or create a dedicated "Documentation" workspace tab) using `QtWebEngineWidgets.QWebEngineView` to host the local site seamlessly.
3. **Cross-Navigation and Deep Linking**: Enable the application to route specific Contextual Help requests (e.g., clicking "?" on a specific Physics Engine tile) to the corresponding offline article URL.
4. **Offline Search**: Provide a local search index (e.g., lunr.js or an SQLite FTS index) to allow full-text querying of the AffineDrift website directly from the UpstreamDrift search bar.

## Feasibility and Architectural Fit

**Is this a reasonable addition?**
Absolutely. This is the industry standard approach for professional engineering and scientific software (e.g., MATLAB's offline documentation, Qt Creator's Assistant, Apple's Xcode DocC). It guarantees documentation continuity, prevents link rot, and ensures that the specific version of the documentation perfectly matches the specific version of the software installed.

It is highly recommended that we host this embedded site inside the `QTabWidget` (Workspace Tabs) we just established for the `LibraryWidget` and `SettingsWidget`.

## Technical Requirements

### 1. Static Site Export

- The AffineDrift website must support exporting as a purely static site (e.g., via Next.js `next export`, Jekyll, Hugo, or Docusaurus).
- Assets (images, fonts, stylesheets) must use relative paths instead of absolute URLs to function correctly under the `file://` protocol.

### 2. Frontend (PyQt6)

- **Dependency Addition**: Ensure `PyQt6-WebEngine` is added to the `pyproject.toml` or `requirements.txt`.
- **WebView Widget**: Create `DocumentationWidget` inheriting from `QWebEngineView`.
- **Security**: Configure `QWebEngineSettings` to allow local file access (`QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls` if necessary, though pure offline is preferred).

### 3. Application State & Context Routing

- Establish an `UrlRouter` in `launcher_dialogs.py` or the `ToolsSidebar` to map application components to specific local URLs (e.g., `file:///.../docs/physics_engines/drake.html`).

## Proposed Sub-Tasks

### Phase 1: Dependency & Foundation Spike

- [ ] Add `PyQt6-WebEngine` to the dependencies list.
- [ ] Create a prototype `DocumentationWidget` that can load a local dummy HTML file.
- [ ] Add a "Documentation" tab to the `workspace_tabs` array in `launcher_ui_setup.py`.

### Phase 2: Content Bundling Pipeline

- [ ] Update the AffineDrift website's build pipeline to produce a fully self-contained static HTML bundle.
- [ ] Create a Python script in `scripts/update_docs.py` that pulls the latest release artifact from the website repository and extracts it into `vendor/docs/`.
- [ ] Add the `vendor/docs/` folder to `.gitignore` if pulling dynamically, or commit it if versioning alongside the code is preferred.

### Phase 3: Deep Linking & UI Integration

- [ ] Map all "Help" buttons across the application (e.g., in the Main Launcher tiles, Sidekick panel, and Settings) to dispatch signals to the `DocumentationWidget`.
- [ ] Intercept external links clicked _inside_ the documentation (e.g., links to GitHub or external papers) and open them in the user's default system browser using `QDesktopServices.openUrl`, while keeping internal links inside the `QWebEngineView`.

## Acceptance Criteria

- The application successfully launches a fully functional instance of the AffineDrift website without an internet connection.
- CSS, Javascript, and media assets load correctly within the embedded WebView.
- Contextual Help buttons successfully navigate the embedded browser to the correct topic.
- Clicking an external link within the documentation safely delegates to the user's default system browser.

## Known Risks & Mitigation

- **Bundle Size**: A heavy website could significantly bloat the software installer size.
  _Mitigation_: Compress images extensively before export and exclude large media (e.g., heavy video tutorials) from the offline bundle, substituting them with placeholder links to the live site.
- **Dependency Weight**: `PyQt6-WebEngine` relies on the Chromium engine and is a substantial dependency.
  _Mitigation_: Assess the installer size impact. If deemed too large, an alternative is to host the static files via a lightweight background local HTTP server (e.g., `http.server`) and open the user's default system browser to `localhost:8000`, though embedding is far superior for UX.
