import type { ReactNode } from "react";
import type { ProblemIssue } from "../api/client";
import { fmtNumber, fmtTime } from "../lib/format";
import { Chip } from "../ui/Chip";
import { cx } from "../ui/cx";
import k from "../ui/kit.module.css";
import { useVisibleIssues } from "./derived";
import s from "./editor.module.css";
import { useEditor, useEditorContext, type CheckResult } from "./store";

const SOURCE: Record<CheckResult["source"], string> = {
  save: "сохранение",
  check: "dry-run",
  launch: "постановка",
  task: "старт задачи",
};

/** Панель «Проверка» (как Problems в IDE): клик по строке — к полю или к строке YAML. */
export function ProblemsPanel({ report }: { report?: (check: CheckResult) => ReactNode }) {
  const { store } = useEditorContext();
  const check = useEditor((st) => st.check);
  const stale = useEditor((st) => st.check !== null && st.checkedText !== st.text);
  const visible = useVisibleIssues();
  const all = check?.issues ?? [];
  const errors = all.filter((i) => i.severity === "error").length;
  const warnings = all.length - errors;
  const visibleSet = new Set(visible.map((v) => `${v.code}|${v.path}|${v.message}`));

  const pick = (issue: ProblemIssue) => {
    const m = issue.where ? /line (\d+)/.exec(issue.where) : null;
    store.getState().focusOn(issue.path ?? null, m ? Number(m[1]) : null);
  };

  return (
    <section className={s.problems} aria-label="Проверка">
      <div className={s.problemsH}>
        <span>Проверка</span>
        {check ? (
          <>
            <Chip tone={errors ? "fail" : "q"}>{errors ? `${fmtNumber(errors)} ${errors === 1 ? "ошибка" : "ошибки"}` : "0 ошибок"}</Chip>
            <Chip tone={warnings ? "warn" : "q"}>{`${fmtNumber(warnings)} предупр.`}</Chip>
            {stale && <span className="small muted">текст изменён после проверки</span>}
            <span className={s.problemsMeta}>
              {SOURCE[check.source]} {fmtTime(new Date(check.at))}
              {check.durationMs !== null && ` · ${fmtNumber(check.durationMs / 1000, 1)} с`}
            </span>
          </>
        ) : (
          <span className={s.problemsMeta}>ещё не было: «Проверить» — dry-run текущего текста без сохранения</span>
        )}
      </div>
      {check?.title && all.length > 0 && (
        <div className={s.report}>
          <b>{check.title}</b>
          {check.hint && <span className="muted">{check.hint}</span>}
        </div>
      )}
      {all.map((issue, i) => {
        const hidden = !visibleSet.has(`${issue.code}|${issue.path}|${issue.message}`);
        return (
          <div key={i} className={cx(s.prow, hidden && s.stale)} title={hidden ? "поле изменено после проверки" : undefined}>
            <span className={issue.severity === "error" ? k.x : k.w}>{issue.severity === "error" ? "✕" : "!"}</span>
            {issue.path || issue.where ? (
              <button type="button" className={s.plink} onClick={() => pick(issue)}>
                {issue.path ?? issue.where}
              </button>
            ) : (
              <span className="mono muted">{issue.code}</span>
            )}
            <span>
              {issue.message}{" "}
              <span className="small muted mono">
                {issue.path && issue.where ? `${issue.where} · ` : ""}
                {issue.code}
              </span>
            </span>
          </div>
        );
      })}
      {check?.ok && report && <div className={s.report}>{report(check)}</div>}
    </section>
  );
}
