import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // #6897: every `/api` call must route through apiFetch/apiFetchForm so it
      // resolves against getApiBase() — bare fetch('/api…') breaks the Tauri
      // desktop build (UI and backend live on different origins there). Banning
      // the literal at lint time keeps the wrapper the single source of truth.
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "CallExpression[callee.name='fetch'] > Literal.arguments:first-child[value=/^\\/api/]",
          message:
            "Do not call fetch('/api…') directly — use apiFetch/apiFetchForm from '@/api/fetch' so the URL resolves via getApiBase() (Tauri-safe). See issue #6897.",
        },
        {
          selector:
            "CallExpression[callee.name='fetch'] > TemplateLiteral.arguments:first-child[quasis.0.value.raw=/^\\/api/]",
          message:
            "Do not call fetch(`/api…`) directly — use apiFetch/apiFetchForm from '@/api/fetch' so the URL resolves via getApiBase() (Tauri-safe). See issue #6897.",
        },
      ],
    },
  },
  {
    // Tests stub a global `fetch` mock and assert on its calls; the backend
    // module cannot import apiFetch (fetch.ts → backend.ts circular dep). These
    // are the only sanctioned bare-fetch sites, so exempt them from the ban.
    files: ["**/*.test.{ts,tsx}", "src/api/backend.ts"],
    rules: {
      "no-restricted-syntax": "off",
    },
  },
]);
