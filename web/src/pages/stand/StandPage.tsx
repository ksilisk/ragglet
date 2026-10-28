import type { ReactNode } from "react";
import { Link } from "react-router";
import type { S } from "../../api/client";
import { useBuild, useBuilds, useCancelTask, useRun, useRuns, useStand, useTasks } from "../../api/queries";
import { useT } from "../../i18n/i18n";
import { fmtBytes, fmtDataset, fmtMetric, fmtMs, fmtNumber, fmtQps, fmtTime, fmtWhen } from "../../lib/format";
import { Button, LinkButton } from "../../ui/Button";
import { Chip, Dot } from "../../ui/Chip";
import { useConfirm } from "../../ui/Dialog";
import { Actions, Card, Empty, KeyValue, Page, PageHead } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { KIND_LABEL, StateChip, taskHref } from "../../ui/states";
import { TaskProgress } from "../../ui/TaskProgress";
import k from "../../ui/kit.module.css";
import s from "./stand.module.css";

export default function StandPage() {
  const t = useT();
  const tasks = useTasks();
  const current = tasks.data?.find((task) => task.state === "running") ?? null;
  const queue = (tasks.data ?? []).filter((task) => task.state === "queued");

  return (
    <Page gap={20}>
      <PageHead
        title={t("Стенд")}
        actions={
          <>
            <LinkButton to="/indexes?tab=profiles">{t("Собрать индекс")}</LinkButton>
            <LinkButton to="/scenarios" variant="primary">
              {t("Запустить сценарий")}
            </LinkButton>
          </>
        }
      />
      {tasks.error && <ErrorBox error={tasks.error} />}
      <div className={s.split}>
        {current ? <NowCard task={current} /> : <IdleCard loading={tasks.isPending} />}
        <QueueCard queue={queue} />
      </div>
      <Services />
      <div className={s.split}>
        <RecentRuns />
        <RecentBuilds />
      </div>
    </Page>
  );
}

function IdleCard({ loading }: { loading: boolean }) {
  const t = useT();
  return (
    <Card title={t("Сейчас")}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <Dot tone={loading ? "idle" : "ok"} />
        {loading ? "…" : t("Стенд свободен")}
      </div>
      <Actions>
        <LinkButton to="/scenarios" variant="primary">
          {t("Запустить сценарий")}
        </LinkButton>
        <LinkButton to="/indexes?tab=profiles">{t("Собрать индекс")}</LinkButton>
      </Actions>
    </Card>
  );
}

function NowCard({ task }: { task: S["TaskItem"] }) {
  const t = useT();
  const run = useRun(task.kind === "run" ? task.id : null);
  const build = useBuild(task.kind === "build" ? task.id : null);
  const cancel = useCancelTask();
  const confirm = useConfirm();
  const status = run.data ?? build.data;
  const href = taskHref(task.kind, task.id);

  const onCancel = async () => {
    const ok = await confirm({
      title: task.kind === "run" ? t("Отменить прогон {name}?", { name: task.name }) : t("Отменить сборку {name}?", { name: task.name }),
      body: t("Процесс задачи будет остановлен. Уже записанные файлы останутся в каталоге."),
      confirm: t("Отменить задачу"),
      cancel: t("Не отменять"),
      danger: true,
    });
    if (ok) cancel.mutate({ kind: task.kind, id: task.id });
  };

  return (
    <Card>
      <div className={s.nowHead}>
        <h2>{t("Сейчас")}</h2>
        <StateChip state={task.state} />
        <span className="muted">{t(KIND_LABEL[task.kind])}</span>
        <Link className="mono" to={href} style={{ fontWeight: 500 }}>
          {task.name}
        </Link>
        <span className="muted small">
          {t("поставлен {enqueued} · старт {started}", { enqueued: fmtTime(task.enqueued_at, false), started: fmtTime(task.started_at) })}
        </span>
      </div>
      <TaskProgress kind={task.kind} state={task.state} progress={task.progress} plan={run.data?.plan} startedAt={task.started_at} />
      {status && (
        <div className={s.facts}>
          <span>{fmtDataset(status.dataset)}</span>
          {run.data && (
            <>
              <span>·</span>
              <span className="mono">{run.data.collection}</span>
              {run.data.plan?.load?.concurrency !== undefined && (
                <>
                  <span>·</span>
                  <span>concurrency {String(run.data.plan.load.concurrency)}</span>
                </>
              )}
            </>
          )}
          {build.data && (
            <>
              <span>·</span>
              <span className="mono">{build.data.targets.map((target) => target.collection).join(", ")}</span>
            </>
          )}
          {status.warnings.map((w, i) => (
            <Chip key={i} tone="warn" title={w.message}>
              {w.code}
            </Chip>
          ))}
        </div>
      )}
      <Actions>
        <LinkButton to={href}>{t("Открыть")}</LinkButton>
        <Button variant="danger" onClick={onCancel} disabled={cancel.isPending}>
          {t("Отменить")}
        </Button>
      </Actions>
    </Card>
  );
}

function QueueCard({ queue }: { queue: S["TaskItem"][] }) {
  const t = useT();
  const cancel = useCancelTask();
  return (
    <Card title={t("Очередь · {n}", { n: queue.length })}>
      {queue.length === 0 ? (
        <span className="muted small">{t("Очередь пуста.")}</span>
      ) : (
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Очередь задач")}>
          <thead>
            <tr>
              <th>#</th>
              <th>{t("Тип")}</th>
              <th>{t("Имя")}</th>
              <th>{t("Поставлена")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {queue.map((task) => (
              <tr key={`${task.kind}:${task.id}`}>
                <td className="num">{task.position}</td>
                <td>{t(KIND_LABEL[task.kind])}</td>
                <td>
                  <Link className={`${k.rowLink} mono`} to={taskHref(task.kind, task.id)}>
                    {task.name}
                  </Link>
                </td>
                <td className="muted">{fmtWhen(task.enqueued_at)}</td>
                <td className="r">
                  <Button small onClick={() => cancel.mutate({ kind: task.kind, id: task.id })} disabled={cancel.isPending}>
                    {t("Убрать")}
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}

function Services() {
  const t = useT();
  const stand = useStand();
  const data = stand.data;
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <h2>{t("Сервисы")}</h2>
      {stand.error && <ErrorBox error={stand.error} />}
      <div className={s.services}>
        <Card>
          {data?.qdrant.length ? (
            data.qdrant.map((q, i) => (
              <div key={q.url} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div className={s.serviceHead}>
                  <Dot tone={q.available ? "ok" : "down"} />
                  <b>Qdrant{data.qdrant.length > 1 ? ` ${i + 1}` : ""}</b>
                  <span className="muted small">{q.version}</span>
                </div>
                <KeyValue
                  items={[
                    ["URL", <span className="mono">{q.url}</span>],
                    q.available ? [t("Коллекции"), <span className="num">{fmtNumber(q.collections)}</span>] : [t("Ошибка"), <span className="small">{q.error}</span>],
                  ]}
                />
              </div>
            ))
          ) : (
            <ServiceStub title="Qdrant" loading={stand.isPending} />
          )}
        </Card>
        <Card>
          {data ? (
            <>
              <div className={s.serviceHead}>
                <Dot tone={data.redis ? (data.redis.available ? "ok" : "down") : "idle"} />
                <b>Redis</b>
                <span className="muted small">{data.redis?.version}</span>
              </div>
              {data.redis ? (
                <KeyValue
                  items={[
                    ["URL", <span className="mono">{data.redis.url}</span>],
                    [t("Роль"), "embedding cache"],
                    !data.redis.available && [t("Ошибка"), <span className="small">{data.redis.error}</span>],
                  ]}
                />
              ) : (
                <span className="muted small">{t("не задан (RAGGLET_REDIS_URL)")}</span>
              )}
            </>
          ) : (
            <ServiceStub title="Redis" loading={stand.isPending} />
          )}
        </Card>
        <Card>
          <div className={s.serviceHead}>
            <b>{t("Вычисления")}</b>
          </div>
          {data && (
            <KeyValue
              items={[
                [t("Устройство"), <span className="mono">{String(data.compute.device ?? "—")}</span>],
                ["CPU", `${String(data.system.cpu ?? "—")} · ${String(data.system.cpus ?? "—")}`],
                ["torch threads", <span className="num">{String(data.compute.torch_threads ?? "—")}</span>],
                typeof data.system.ram_bytes === "number" && ["RAM", fmtBytes(data.system.ram_bytes)],
              ]}
            />
          )}
        </Card>
        <Card>
          <div className={s.serviceHead}>
            <b>{t("Код и окружение")}</b>
          </div>
          {data && (
            <KeyValue
              items={[
                ["commit", <span className="mono">{data.commit ? data.commit.slice(0, 8) : "—"}</span>],
                ["Python", <span className="num">{data.python}</span>],
                ...Object.entries(data.env).map(([name, set]): [string, ReactNode] => [
                  name,
                  <Chip tone={set ? "ok" : "warn"}>{set ? t("задана") : t("не задана")}</Chip>,
                ]),
              ]}
            />
          )}
        </Card>
      </div>
    </section>
  );
}

function ServiceStub({ title, loading }: { title: string; loading: boolean }) {
  const t = useT();
  return (
    <div className={s.serviceHead}>
      <Dot tone="idle" />
      <b>{title}</b>
      <span className="muted small">{loading ? "…" : t("нет данных")}</span>
    </div>
  );
}

function RecentRuns() {
  const t = useT();
  const runs = useRuns({ limit: 5, sort: "-enqueued_at", k: 10 });
  return (
    <Card
      title={t("Недавние прогоны")}
      aside={
        <Link to="/runs" className="small">
          {t("Все прогоны →")}
        </Link>
      }
    >
      {runs.error && <ErrorBox error={runs.error} />}
      {runs.data && runs.data.items.length === 0 ? (
        <Empty>{t("Прогонов ещё нет.")}</Empty>
      ) : (
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Недавние прогоны")}>
          <thead>
            <tr>
              <th>{t("Состояние")}</th>
              <th>{t("Сценарий")}</th>
              <th>{t("Когда")}</th>
              <th className="r">nDCG@10</th>
              <th className="r">{t("p95, мс")}</th>
              <th className="r">QPS</th>
            </tr>
          </thead>
          <tbody>
            {runs.data?.items.map((r) => (
              <tr key={r.id}>
                <td>
                  <StateChip state={r.state} />
                </td>
                <td>
                  <Link className={`${k.rowLink} mono`} to={`/runs/${encodeURIComponent(r.id)}`}>
                    {r.scenario}
                  </Link>
                </td>
                <td className="muted">{fmtWhen(r.started_at ?? r.enqueued_at)}</td>
                <Metric value={r.ndcg} format={fmtMetric} />
                <Metric value={r.p95_ms} format={fmtMs} />
                <Metric value={r.qps} format={fmtQps} />
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}

function Metric({ value, format }: { value: number | null; format: (x: number | null) => string }) {
  return <td className={value === null ? "r muted" : "r num"}>{format(value)}</td>;
}

function RecentBuilds() {
  const t = useT();
  const builds = useBuilds({ limit: 3, sort: "-enqueued_at" });
  return (
    <Card
      title={t("Недавние сборки")}
      aside={
        <Link to="/indexes" className="small">
          {t("Индексы →")}
        </Link>
      }
    >
      {builds.error && <ErrorBox error={builds.error} />}
      {builds.data && builds.data.items.length === 0 ? (
        <Empty>{t("Сборок через API ещё не было.")}</Empty>
      ) : (
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Недавние сборки")}>
          <thead>
            <tr>
              <th>{t("Состояние")}</th>
              <th>{t("Профиль")}</th>
              <th>{t("Коллекции")}</th>
              <th>{t("Когда")}</th>
            </tr>
          </thead>
          <tbody>
            {builds.data?.items.map((b) => (
              <tr key={b.id}>
                <td>
                  <StateChip state={b.state} />
                </td>
                <td>
                  <Link className={`${k.rowLink} mono`} to={`/builds/${encodeURIComponent(b.id)}`}>
                    {b.profile}
                  </Link>
                </td>
                <td className="mono">{b.targets.map((target) => target.collection).join(", ")}</td>
                <td className="muted">{fmtWhen(b.started_at ?? b.enqueued_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
