export interface ThemeColors {
  bg: string;
  group_bg: string;
  border: string;
  text: string;
  text_secondary: string;
  label: string;
  focus: string;
  input_bg: string;
  accent: string;
  title_bg: string;
  title_border: string;
  table_header: string;
  table_alt: string;
  button_hover: string;
}

export interface ActiveThemeResponse {
  name: string;
  is_builtin: boolean;
  colors: ThemeColors;
}

export async function fetchActiveTheme(): Promise<ActiveThemeResponse> {
  const response = await fetch('/api/v1/themes/active');
  if (!response.ok) {
    throw new Error('Failed to fetch active theme');
  }
  return response.json();
}

export function applyThemeToCSSVariables(colors: ThemeColors) {
  const root = document.documentElement;
  Object.entries(colors).forEach(([key, value]) => {
    // Convert python snake_case to css kebab-case
    const cssKey = key.replace(/_/g, '-');
    root.style.setProperty(`--theme-${cssKey}`, value);
  });
}
