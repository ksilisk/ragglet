import { NavLink, Outlet, useLocation } from "react-router";
import { msg, useT } from "../i18n/i18n";
import { cx } from "../ui/cx";
import { Prefs } from "./Prefs";
import { StandStrip } from "./StandStrip";
import s from "./shell.module.css";

const SECTIONS = [
  { to: "/", label: msg("Стенд"), match: (p: string) => p === "/" },
  { to: "/scenarios", label: msg("Сценарии"), match: (p: string) => p.startsWith("/scenarios") },
  { to: "/indexes", label: msg("Индексы"), match: (p: string) => p.startsWith("/indexes") || p.startsWith("/builds") },
  { to: "/runs", label: msg("Прогоны"), match: (p: string) => p.startsWith("/runs") },
];

export function AppShell() {
  const t = useT();
  const { pathname } = useLocation();
  return (
    <>
      <nav className={s.nav} aria-label={t("Разделы")}>
        <NavLink className={s.brand} to="/">
          ragglet
        </NavLink>
        {SECTIONS.map((section) => (
          <NavLink key={section.to} to={section.to} className={cx(s.item, section.match(pathname) && s.on)}>
            {t(section.label)}
          </NavLink>
        ))}
        <StandStrip />
        <Prefs />
      </nav>
      <Outlet />
    </>
  );
}
