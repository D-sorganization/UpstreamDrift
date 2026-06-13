/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  // #7423: class-based theming. The root carries `class="dark"` so today's
  // appearance is preserved; flipping the class to light activates the
  // `:root:not(.dark)` token block in index.css.
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Semantic aliases (UI/UX #7421). Reference these tokens in new code
        // instead of raw palette names so the app has one recognizable accent
        // and consistent status colors.
        primary: {
          DEFAULT: '#2563eb', // blue-600
          hover: '#1d4ed8', // blue-700
        },
        success: {
          DEFAULT: '#16a34a', // green-600
          hover: '#15803d', // green-700
        },
        warning: {
          DEFAULT: '#d97706', // amber-600
          hover: '#b45309', // amber-700
        },
        danger: {
          DEFAULT: '#dc2626', // red-600
          hover: '#b91c1c', // red-700
        },
        // #7423: token-backed utilities (bg-surface, text-token-muted…) map to
        // the runtime-themeable --sidekick-color-* variables so the whole app
        // can be re-themed by swapping the variable block, not by editing
        // class strings.
        canvas: 'var(--sidekick-color-canvas)',
        surface: {
          DEFAULT: 'var(--sidekick-color-surface)',
          raised: 'var(--sidekick-color-surface-raised)',
          muted: 'var(--sidekick-color-surface-muted)',
        },
        token: {
          border: 'var(--sidekick-color-border)',
          text: 'var(--sidekick-color-text)',
          muted: 'var(--sidekick-color-text-muted)',
          subtle: 'var(--sidekick-color-text-subtle)',
        },
      },
    },
  },
  plugins: [],
};
