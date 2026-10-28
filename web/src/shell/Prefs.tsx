import type { ReactNode } from "react";
import { msg, useLang, useT, type Lang } from "../i18n/i18n";
import { useTheme, type Theme } from "../lib/theme";
import { cx } from "../ui/cx";
import s from "./shell.module.css";

const ICONS: Record<Theme, ReactNode> = {
  light: (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
    </svg>
  ),
  dark: (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
      <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7Z" />
    </svg>
  ),
  system: (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
      <rect x="2" y="3" width="12" height="8" rx="1.5" />
      <path d="M6 14h4M8 11v3" />
    </svg>
  ),
};

const THEMES: Array<{ id: Theme; label: string }> = [
  { id: "light", label: msg("Светлая") },
  { id: "dark", label: msg("Тёмная") },
  { id: "system", label: msg("Как в системе") },
];
const LANGS: Lang[] = ["ru", "en"];

/** Тема и язык в шапке: сегментированные переключатели; выбор хранится в localStorage. */
export function Prefs() {
  const t = useT();
  const [theme, setTheme] = useTheme();
  const { lang, setLang } = useLang();
  return (
    <div className={s.prefs}>
      <div className={s.seg} role="radiogroup" aria-label={t("Тема")}>
        {THEMES.map((th) => (
          <button
            key={th.id}
            type="button"
            role="radio"
            aria-checked={theme === th.id}
            aria-label={t(th.label)}
            title={t(th.label)}
            className={cx(theme === th.id && s.segOn)}
            onClick={() => setTheme(th.id)}
          >
            {ICONS[th.id]}
          </button>
        ))}
      </div>
      <div className={s.seg} role="radiogroup" aria-label={t("Язык")}>
        {LANGS.map((l) => (
          <button key={l} type="button" role="radio" aria-checked={lang === l} className={cx(lang === l && s.segOn)} onClick={() => setLang(l)}>
            {l.toUpperCase()}
          </button>
        ))}
      </div>
    </div>
  );
}
