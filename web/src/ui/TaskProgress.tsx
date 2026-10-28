import type { S } from "../api/client";
import { useT } from "../i18n/i18n";
import { progressView } from "../lib/eta";
import { fmtDuration, fmtNumber, fmtQps } from "../lib/format";
import { PHASE_HINT, PHASE_LABEL, UNIT, phaseOf } from "../lib/task";
import { useNow } from "../lib/useNow";
import { ProgressBar } from "./Layout";

/** Строка «Запросы · 120 / 323 · повтор 1 из 1 … прошло · осталось ≈» и полоса — Стенд и шапка задачи. */
export function TaskProgress({ kind, state, progress, plan, startedAt, big }: {
  kind: S["TaskKind"];
  state: S["TaskState"];
  progress: S["ProgressOut"] | null | undefined;
  plan?: S["RunPlanOut"] | null;
  startedAt: string | null | undefined;
  big?: boolean;
}) {
  const t = useT();
  const now = useNow(1000, state === "running");
  const phase = phaseOf(state, progress);
  const view = progressView(progress, plan, startedAt, now);
  const elapsed = view?.elapsedS ?? (startedAt ? (now - new Date(startedAt).getTime()) / 1000 : null);

  if (phase === "queued" || phase === "done") return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: big ? 8 : 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", fontSize: big ? 14 : 13 }}>
        <span>
          {t(PHASE_LABEL[kind][phase])}
          {view ? (
            <>
              {" · "}
              <span className="num">
                {fmtNumber(view.done)} / {fmtNumber(view.total)}
              </span>
              {kind === "run" && view.repeat && " · " + t("повтор {index} из {of}", { index: view.repeat.index, of: view.repeat.of })}
            </>
          ) : (
            <span className="muted"> · {t(PHASE_HINT[kind].preparing)}</span>
          )}
          {phase === "finishing" && <span className="muted"> · {t(PHASE_HINT[kind].finishing)}</span>}
        </span>
        <span className="muted">
          {t("прошло")} <b className="num">{fmtDuration(elapsed)}</b>
          {phase === "progress" && (
            <>
              {" · " + t("осталось") + " "}
              <b className="num">{view?.remainingS === null || view?.remainingS === undefined ? t("оценивается…") : `≈ ${fmtDuration(view.remainingS)}`}</b>
              {view?.rate ? (
                <>
                  {" · "}
                  <span className="num">≈ {fmtQps(view.rate)}</span> {kind === "run" ? t("запр./с") : t("док./с")}
                </>
              ) : null}
            </>
          )}
        </span>
      </div>
      <ProgressBar
        big={big}
        fraction={phase === "preparing" ? null : view ? view.fraction : null}
        label={view ? t("Прогресс: {done} из {total} {unit}", { done: view.done, total: view.total, unit: t(UNIT[kind]) }) : t("Прогресс: {phase}", { phase: t(PHASE_LABEL[kind][phase]) })}
      />
    </div>
  );
}
