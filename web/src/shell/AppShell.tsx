import { NavLink, Outlet, useLocation } from "react-router";
import { cx } from "../ui/cx";
import { StandStrip } from "./StandStrip";
import s from "./shell.module.css";

const SECTIONS = [
  { to: "/", label: "Стенд", match: (p: string) => p === "/" },
  { to: "/scenarios", label: "Сценарии", match: (p: string) => p.startsWith("/scenarios") },
  { to: "/indexes", label: "Индексы", match: (p: string) => p.startsWith("/indexes") || p.startsWith("/builds") },
  { to: "/runs", label: "Прогоны", match: (p: string) => p.startsWith("/runs") },
];

export function AppShell() {
  const { pathname } = useLocation();
  return (
    <>
      <nav className={s.nav} aria-label="Разделы">
        <NavLink className={s.brand} to="/">
          ragglet
        </NavLink>
        {SECTIONS.map((section) => (
          <NavLink key={section.to} to={section.to} className={cx(s.item, section.match(pathname) && s.on)}>
            {section.label}
          </NavLink>
        ))}
        <StandStrip />
      </nav>
      <Outlet />
    </>
  );
}
