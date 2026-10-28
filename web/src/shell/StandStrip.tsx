import { Link } from "react-router";
import { ApiProblem } from "../api/client";
import { POLL, useStand, useTasks } from "../api/queries";
import { rich, useT } from "../i18n/i18n";
import { fmtNumber } from "../lib/format";
import { PHASE_LABEL } from "../lib/task";
import { Dot } from "../ui/Chip";
import { KIND_LABEL, taskHref } from "../ui/states";
import s from "./shell.module.css";

/** Полоса стенда в шапке: текущая задача, очередь, доступность Qdrant и Redis. */
export function StandStrip() {
  const t = useT();
  const tasks = useTasks();
  const stand = useStand(POLL.standStrip);

  if (tasks.error instanceof ApiProblem && tasks.error.status === 0) {
    return (
      <div className={s.strip}>
        <span className={s.down}>
          <Dot tone="down" />
          {rich(t("API недоступен: запущен ли {cmd}?"), { cmd: <span className="mono">python3 ragglet start</span> })}
        </span>
      </div>
    );
  }

  const list = tasks.data ?? [];
  const current = list.find((t) => t.state === "running") ?? null;
  const waiting = list.filter((t) => t.state === "queued").length;
  const qdrant = stand.data?.qdrant ?? [];
  const qdrantUp = qdrant.length > 0 && qdrant.every((q) => q.available);
  const redis = stand.data?.redis;

  return (
    <div className={s.strip}>
      {current ? (
        <Link to={taskHref(current.kind, current.id)}>
          <Dot tone="busy" />
          {t(KIND_LABEL[current.kind])} <b>{current.name}</b>
          {" · "}
          {current.progress ? (
            <span className="num">
              {fmtNumber(current.progress.done)} / {fmtNumber(current.progress.total)}
            </span>
          ) : (
            t(PHASE_LABEL[current.kind].preparing).toLowerCase()
          )}
          {waiting > 0 && " · " + t("ещё {n} в очереди", { n: waiting })}
        </Link>
      ) : waiting > 0 ? (
        <Link to="/">
          <Dot tone="busy" />
          {t("В очереди: {n}", { n: waiting })}
        </Link>
      ) : (
        <span>
          <Dot tone={tasks.data ? "ok" : "idle"} />
          {tasks.data ? t("Стенд свободен") : "…"}
        </span>
      )}
      {stand.data && (
        <span title={qdrant.map((q) => q.url + ": " + (q.available ? q.version ?? t("доступен") : q.error ?? t("недоступен"))).join("\n")}>
          <Dot tone={qdrantUp ? "ok" : "down"} />
          Qdrant
        </span>
      )}
      {redis && (
        <span title={redis.url + ": " + (redis.available ? redis.version ?? t("доступен") : redis.error ?? t("недоступен"))}>
          <Dot tone={redis.available ? "ok" : "down"} />
          Redis
        </span>
      )}
    </div>
  );
}
