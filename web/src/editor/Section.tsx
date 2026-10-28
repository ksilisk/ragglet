import { useEffect, useState, type ReactNode } from "react";
import { Chip } from "../ui/Chip";
import { cx } from "../ui/cx";
import { useSectionMarks, useVisibleIssues } from "./derived";
import s from "./editor.module.css";
import { useEditor, useEditorContext } from "./store";

export const sectionDomId = (id: string) => `s-${id}`;

/** Секция формы (.sec): свёрнутая показывает одну строку сводки; включатель стадии — в заголовке. */
export function Section({ id, title, summary, toggle, children }: {
  id: string;
  title: string;
  summary?: ReactNode;
  /** Чип «включён / выключен» и чекбокс стадии в заголовке. */
  toggle?: { on: boolean; set: (on: boolean) => void; disabled?: boolean };
  children: ReactNode;
}) {
  const { store } = useEditorContext();
  const open = useEditor((st) => st.open[id] ?? false);
  const marks = useSectionMarks();
  const issues = useVisibleIssues().filter((i) => i.section === id);
  const errors = issues.filter((i) => i.severity === "error").length;
  const warnings = issues.length - errors;
  const readOnly = useEditor((st) => st.parsed.error !== null);
  const isOff = toggle ? !toggle.on : false;
  useEffect(() => {
    store.getState().setOff(id, isOff);
  }, [store, id, isOff]);
  return (
    <section className={cx(s.sec, marks.get(id) === "error" && s.err)} id={sectionDomId(id)} aria-label={title}>
      <div style={{ display: "flex", alignItems: "center" }}>
        <button type="button" className={s.secH} aria-expanded={open} onClick={() => store.getState().setOpen(id, !open)}>
          <span className={cx(s.caret, !open && s.c)} aria-hidden />
          <h2>{title}</h2>
          {toggle?.on && <Chip tone="ok">включён</Chip>}
          {errors > 0 && <Chip tone="fail">{errors === 1 ? "1 ошибка" : `${errors} ошибки`}</Chip>}
          {warnings > 0 && <Chip tone="warn">{warnings === 1 ? "1 предупреждение" : `${warnings} предупр.`}</Chip>}
          {!open && summary && <span className={s.sum}>{summary}</span>}
        </button>
        {toggle && (
          <label className={s.chk} style={{ padding: "0 16px", whiteSpace: "nowrap" }}>
            <input type="checkbox" aria-label={`Включить ${title}`} checked={toggle.on} disabled={readOnly || toggle.disabled} onChange={(e) => {
                toggle.set(e.target.checked);
                if (e.target.checked) store.getState().setOpen(id, true);
              }}
            />
            включить
          </label>
        )}
      </div>
      {open && <div className={s.secB}>{children}</div>}
    </section>
  );
}

/**
 * Навигация по секциям — оглавление: подсвечена одна строка, секция, которая сейчас вверху экрана
 * (а не все раскрытые). Точки — проблемы проверки, «выкл» — выключенная стадия.
 */
export function SectionNav({ sections }: { sections: ReadonlyArray<{ id: string; title: string }> }) {
  const { store } = useEditorContext();
  const marks = useSectionMarks();
  const off = useEditor((st) => st.off);
  const [active, setActive] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      let current = sections[0]?.id ?? "";
      for (const sec of sections) {
        const el = document.getElementById(sectionDomId(sec.id));
        if (el && el.getBoundingClientRect().top <= SPY_OFFSET) current = sec.id;
      }
      // докрутили до конца — последняя секция, даже если её верх ниже линии
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        const last = [...sections].reverse().find((sec) => document.getElementById(sectionDomId(sec.id)));
        if (last) current = last.id;
      }
      setActive(current);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [sections]);

  return (
    <nav className={s.secnav} aria-label="Секции">
      {sections.map((sec) => {
        const mark = marks.get(sec.id);
        return (
          <button
            key={sec.id}
            type="button"
            className={cx(active === sec.id && s.on, off[sec.id] && s.offStage)}
            aria-current={active === sec.id ? "location" : undefined}
            onClick={() => {
              store.getState().setOpen(sec.id, true);
              setActive(sec.id);
              requestAnimationFrame(() => document.getElementById(sectionDomId(sec.id))?.scrollIntoView({ behavior: "smooth", block: "start" }));
            }}
          >
            <span>
              {sec.title}
              {off[sec.id] && <span className={s.offLabel}> · выкл</span>}
            </span>
            {mark && <span className={cx(s.sdot, mark === "error" ? s.e : s.w)} title={mark === "error" ? "есть ошибки" : "есть предупреждения"} />}
          </button>
        );
      })}
    </nav>
  );
}

/** Линия «вверху экрана» для оглавления: секция, чей верх выше неё, считается текущей. */
const SPY_OFFSET = 120;
