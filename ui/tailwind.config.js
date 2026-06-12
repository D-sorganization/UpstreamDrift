/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
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
      },
    },
  },
  plugins: [],
};
