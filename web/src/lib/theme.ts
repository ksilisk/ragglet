import { useCallback, useState } from "react";
import { readPref, writePref } from "./storage";

export type Theme = "system" | "light" | "dark";
export const THEME_KEY = "ragglet.theme";

export function readTheme(): Theme {
  const v = readPref(THEME_KEY);
  return v === "light" || v === "dark" ? v : "system";
}

/** data-theme на <html> — явный выбор; без атрибута тема следует системе (tokens.css). */
export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
}

export function useTheme(): [Theme, (theme: Theme) => void] {
  const [theme, setState] = useState<Theme>(readTheme);
  const set = useCallback((next: Theme) => {
    writePref(THEME_KEY, next);
    applyTheme(next);
    setState(next);
  }, []);
  return [theme, set];
}
