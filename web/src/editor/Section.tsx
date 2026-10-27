import type { ReactNode } from "react";
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

export function SectionNav({ sections }: { sections: ReadonlyArray<{ id: string; title: string }> }) {
  const { store } = useEditorContext();
  const marks = useSectionMarks();
  const open = useEditor((st) => st.open);
  return (
    <nav className={s.secnav} aria-label="Секции">
      {sections.map((sec) => {
        const mark = marks.get(sec.id);
        return (
          <button
            key={sec.id}
            type="button"
            className={cx(open[sec.id] && s.on)}
            onClick={() => {
              store.getState().setOpen(sec.id, true);
              requestAnimationFrame(() => document.getElementById(sectionDomId(sec.id))?.scrollIntoView({ behavior: "smooth", block: "start" }));
            }}
          >
            <span>{sec.title}</span>
            {mark && <span className={cx(s.sdot, mark === "error" ? s.e : s.w)} title={mark === "error" ? "есть ошибки" : "есть предупреждения"} />}
          </button>
        );
      })}
    </nav>
  );
}
