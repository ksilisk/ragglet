import type { S } from "../api/client";
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
  const now = useNow(1000, state === "running");
  const phase = phaseOf(state, progress);
  const view = progressView(progress, plan, startedAt, now);
  const elapsed = view?.elapsedS ?? (startedAt ? (now - new Date(startedAt).getTime()) / 1000 : null);

  if (phase === "queued" || phase === "done") return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: big ? 8 : 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", fontSize: big ? 14 : 13 }}>
        <span>
          {PHASE_LABEL[kind][phase]}
          {view ? (
            <>
              {" · "}
              <span className="num">
                {fmtNumber(view.done)} / {fmtNumber(view.total)}
              </span>
              {kind === "run" && view.repeat && ` · повтор ${view.repeat.index} из ${view.repeat.of}`}
            </>
          ) : (
            <span className="muted"> · {PHASE_HINT[kind].preparing}</span>
          )}
          {phase === "finishing" && <span className="muted"> · {PHASE_HINT[kind].finishing}</span>}
        </span>
        <span className="muted">
          прошло <b className="num">{fmtDuration(elapsed)}</b>
          {phase === "progress" && (
            <>
              {" · осталось "}
              <b className="num">{view?.remainingS === null || view?.remainingS === undefined ? "оценивается…" : `≈ ${fmtDuration(view.remainingS)}`}</b>
              {view?.rate ? (
                <>
                  {" · "}
                  <span className="num">≈ {fmtQps(view.rate)}</span> {kind === "run" ? "запр./с" : "док./с"}
                </>
              ) : null}
            </>
          )}
        </span>
      </div>
      <ProgressBar
        big={big}
        fraction={phase === "preparing" ? null : view ? view.fraction : null}
        label={`Прогресс: ${view ? `${view.done} из ${view.total} ${UNIT[kind]}` : PHASE_LABEL[kind][phase]}`}
      />
    </div>
  );
}
