import { useEffect, useMemo, useState } from "react";
import { useLang, type Lang } from "../i18n/i18n";

/**
 * Цвета и шрифты графика из токенов tokens.css (ragglet-32 §2): графики перекрашиваются вместе с
 * темой UI, своих цветов у компонентов нет. lang — чтобы опция пересобиралась при смене языка:
 * форматтеры подписей берут числа и единицы по языку UI.
 */
export interface ChartTheme {
  dark: boolean;
  lang: Lang;
  text: string;
  muted: string;
  faint: string;
  grid: string;
  axis: string;
  surface: string;
  tooltipBg: string;
  tooltipText: string;
  ok: string;
  warning: string;
  danger: string;
  series: string[];
  fontSans: string;
  fontMono: string;
}

type Palette = Omit<ChartTheme, "dark" | "lang">;

const VARS: Record<Exclude<keyof Palette, "series">, string> = {
  text: "--text",
  muted: "--muted",
  faint: "--faint",
  grid: "--border-soft",
  axis: "--border",
  surface: "--surface",
  tooltipBg: "--tooltip-bg",
  tooltipText: "--tooltip-text",
  ok: "--ok",
  warning: "--warning",
  danger: "--danger",
  fontSans: "--font-sans",
  fontMono: "--font-mono",
};
export const SERIES_SLOTS = 6;

/**
 * Светлая тема как в tokens.css (тест сверяет): экспорт графиков всегда в ней, на белом фоне,
 * в какой бы теме ни был открыт UI (решение 3 ревью).
 */
export const LIGHT: Palette = {
  text: "#15181d",
  muted: "#5f6873",
  faint: "#9aa3ae",
  grid: "#eef0f3",
  axis: "#e2e5ea",
  surface: "#ffffff",
  tooltipBg: "#15181d",
  tooltipText: "#ffffff",
  ok: "#16a34a",
  warning: "#d97706",
  danger: "#dc2626",
  series: ["#285cc2", "#f2823b", "#0a978b", "#6331a0", "#e0ab22", "#d4679f"],
  fontSans: '"IBM Plex Sans", system-ui, sans-serif',
  fontMono: '"IBM Plex Mono", ui-monospace, monospace',
};

export const lightTheme = (lang: Lang): ChartTheme => ({ ...LIGHT, dark: false, lang });

export function readChartTheme(lang: Lang): ChartTheme {
  const css = getComputedStyle(document.documentElement);
  const get = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  const palette = Object.fromEntries(Object.entries(VARS).map(([key, name]) => [key, get(name, LIGHT[key as keyof typeof VARS])])) as Omit<Palette, "series">;
  return {
    ...palette,
    series: LIGHT.series.map((fallback, i) => get(`--series-${i + 1}`, fallback)),
    dark: get("color-scheme", "light") === "dark",
    lang,
  };
}

/** Тема графика; перечитывается при смене data-theme на <html>, системной темы и языка. */
export function useChartTheme(): ChartTheme {
  const { lang } = useLang();
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    media?.addEventListener("change", bump);
    return () => {
      observer.disconnect();
      media?.removeEventListener("change", bump);
    };
  }, []);
  // version — сигнал «токены сменились»: getComputedStyle читается заново
  return useMemo(() => ({ ...readChartTheme(lang), version }), [lang, version]);
}
