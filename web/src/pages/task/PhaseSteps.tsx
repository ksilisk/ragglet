import type { ReactNode } from "react";
import type { S } from "../../api/client";
import { useT } from "../../i18n/i18n";
import { PHASE_LABEL, phaseOf, type Phase } from "../../lib/task";
import { cx } from "../../ui/cx";
import s from "./task.module.css";

const ORDER: Array<Exclude<Phase, "done">> = ["queued", "preparing", "progress", "finishing"];

/** Шаги «В очереди → Подготовка → Запросы → Завершение» (артборд RunRunning). */
export function PhaseSteps({ kind, state, progress, details }: {
  kind: S["TaskKind"];
  state: S["TaskState"];
  progress: S["ProgressOut"] | null | undefined;
  details: Partial<Record<Exclude<Phase, "done">, ReactNode>>;
}) {
  const t = useT();
  const phase = phaseOf(state, progress);
  const current = phase === "done" ? ORDER.length : ORDER.indexOf(phase);
  return (
    <div className={s.steps}>
      {ORDER.map((step, i) => {
        const status = i < current ? "done" : i === current ? "now" : "next";
        return (
          <div key={step} style={{ display: "contents" }}>
            {i > 0 && <div className={cx(s.line, i <= current && s.done)} />}
            <div className={s.step}>
              <div className={s.stepT}>
                <span className={cx(s.ic, s[status])} aria-hidden>
                  {status === "done" ? "✓" : status === "now" ? "●" : i + 1}
                </span>
                {t(PHASE_LABEL[kind][step])}
              </div>
              <div className={s.stepD}>{details[step]}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
