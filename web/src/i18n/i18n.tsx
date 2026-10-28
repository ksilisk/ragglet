import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { readPref, writePref } from "../lib/storage";
import { en } from "./en";

/**
 * Языки UI. Ключ перевода — русский текст из кода: t("Сохранить") → «Save» в en. Строки-константы вне
 * компонентов помечаются msg("…") и переводятся при выводе через t(). Подстановки — {name}, формы
 * числа — «запрос|запроса|запросов» (en: «query|queries»). Чего нет в en, показывается по-русски;
 * тест i18n.test.ts требует перевода для каждой русской строки в src.
 */
export type Lang = "ru" | "en";
export const LANG_KEY = "ragglet.lang";
export type Vars = Record<string, string | number>;

function readLang(): Lang {
  return readPref(LANG_KEY) === "en" ? "en" : "ru";
}

let current: Lang = typeof window === "undefined" ? "ru" : readLang();

export function currentLang(): Lang {
  return current;
}

/** Локаль чисел и дат: «0,331» и «9 696» — ru, «0.331» и «9,696» — en. */
export function numberLocale(lang: Lang = current): string {
  return lang === "en" ? "en-US" : "ru-RU";
}

/** Пометка строки-константы как переводимой: возвращает её как есть, перевод — t() при выводе. */
export function msg<T extends string>(text: T): T {
  return text;
}

function fill(template: string, vars?: Vars): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (match, key: string) => (key in vars ? String(vars[key]) : match));
}

export function translate(text: string, vars?: Vars, lang: Lang = current): string {
  return fill(lang === "en" ? (en[text] ?? text) : text, vars);
}

/** Форма слова по числу: forms — «одна|две|пять» по-русски, перевод — «one|other». */
export function pluralize(n: number, forms: string, lang: Lang = current): string {
  const parts = (lang === "en" ? (en[forms] ?? forms) : forms).split("|");
  if (lang === "en") return (Math.abs(n) === 1 ? parts[0] : (parts[1] ?? parts[0]))!;
  const mod10 = Math.abs(n) % 10;
  const mod100 = Math.abs(n) % 100;
  if (mod10 === 1 && mod100 !== 11) return parts[0]!;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return (parts[1] ?? parts[0])!;
  return (parts[2] ?? parts[1] ?? parts[0])!;
}

export interface T {
  (text: string, vars?: Vars): string;
  plural(n: number, forms: string): string;
  lang: Lang;
}

function makeT(lang: Lang): T {
  const t = ((text: string, vars?: Vars) => translate(text, vars, lang)) as T;
  t.plural = (n, forms) => pluralize(n, forms, lang);
  t.lang = lang;
  return t;
}

const LangContext = createContext<{ lang: Lang; setLang: (lang: Lang) => void }>({ lang: "ru", setLang: () => {} });

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setState] = useState<Lang>(current);
  const setLang = useCallback((next: Lang) => {
    current = next;
    writePref(LANG_KEY, next);
    document.documentElement.lang = next;
    setState(next);
  }, []);
  const value = useMemo(() => ({ lang, setLang }), [lang, setLang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  return useContext(LangContext);
}

export function useT(): T {
  const { lang } = useContext(LangContext);
  return useMemo(() => makeT(lang), [lang]);
}
