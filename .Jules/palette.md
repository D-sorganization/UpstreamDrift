## 2024-05-23 - Standard Icons for Primary Actions
**Learning:** Using standard `QStyle.StandardPixmap` icons provides immediate visual affordance and consistency with the host OS, which is critical for primary actions like Play, Pause, and Save.
**Action:** Always prefer `self.style().standardIcon(...)` over custom assets for universal actions like media playback and file operations to ensure platform-native look and feel without adding asset dependencies.
