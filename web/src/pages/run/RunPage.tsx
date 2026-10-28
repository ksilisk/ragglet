import { useQueryClient } from "@tanstack/react-query";
import { useState, type ComponentType } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { ApiProblem, api, unwrap, type S } from "../../api/client";
import { keys, useCancelTask, useRun, useRunArtifacts, useRunInclude, useTasks } from "../../api/queries";
import { launch } from "../../editor/launch";
import { msg, rich, useT } from "../../i18n/i18n";
import { YamlEditor } from "../../editor/YamlEditor";
import { fmtBytes, fmtDuration, fmtNumber, fmtTime, fmtWhen, splitDataset } from "../../lib/format";
import { PHASE_HINT } from "../../lib/task";
import { useNow } from "../../lib/useNow";
import { Button, FileButton } from "../../ui/Button";
import { Chip, Tag } from "../../ui/Chip";
import { useConfirm } from "../../ui/Dialog";
import k from "../../ui/kit.module.css";
import { Card, Crumbs, Empty, KeyValue, Page } from "../../ui/Layout";
import { Menu } from "../../ui/Menu";
import { ErrorBox, IssueList, ProblemBox } from "../../ui/ProblemView";
import { Tabs } from "../../ui/Tabs";
import { ACTIVE, KIND_LABEL, StateChip, taskHref } from "../../ui/states";
import { TaskProgress } from "../../ui/TaskProgress";
import { NameDialog } from "../library/LibraryDialogs";
import { fromRun } from "../library/library";
import { WarningsBody, checkStep, dryRunScenario, enqueueScenario } from "../runs/enqueue";
import { LogView } from "../task/LogView";
import { PhaseSteps } from "../task/PhaseSteps";
import s from "../task/task.module.css";
import { RunOverview } from "./RunOverview";

type TabId = "overview" | "quality" | "latency" | "queries" | "conditions" | "config" | "log" | "files";

/**
 * Реестр вкладок прогона (ragglet-31 §8): 32 добавляет свои записи вместо заглушек «soon», не трогая
 * страницу. available решает, есть ли вкладка у прогона в этом состоянии. title — msg(), перевод при выводе.
 */
export const RUN_TABS: Array<{ id: TabId; title: string; available: (run: S["RunStatus"]) => boolean | "soon"; Component?: ComponentType<{ run: S["RunStatus"] }> }> = [
  { id: "overview", title: msg("Обзор"), available: (r) => r.state === "completed", Component: RunOverview },
  { id: "quality", title: msg("Качество"), available: () => "soon" },
  { id: "latency", title: msg("Задержки и нагрузка"), available: () => "soon" },
  { id: "queries", title: msg("Запросы"), available: () => "soon" },
  { id: "conditions", title: msg("Условия"), available: () => "soon" },
  { id: "config", title: msg("Конфиг"), available: () => true, Component: RunConfig },
  { id: "log", title: msg("Лог"), available: (r) => r.state !== "queued", Component: RunLog },
  { id: "files", title: msg("Файлы"), available: (r) => r.state !== "queued", Component: RunFiles },
];

export default function RunPage() {
  const t = useT();
  const { id = "" } = useParams();
  const run = useRun(id);
  if (run.error) {
    const notFound = run.error instanceof ApiProblem && run.error.status === 404;
    return (
      <Page>
        <Crumbs items={[{ label: t("Прогоны"), to: "/runs" }, { label: id }]} />
        {notFound ? <Empty>{t("Прогона {id} нет в реестре.", { id })}</Empty> : <ErrorBox error={run.error} />}
      </Page>
    );
  }
  if (!run.data) {
    return (
      <Page>
        <Crumbs items={[{ label: t("Прогоны"), to: "/runs" }, { label: id }]} />
        <span className="muted">{t("Загрузка…")}</span>
      </Page>
    );
  }
  return <RunView run={run.data} />;
}

function RunView({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const [params, setParams] = useSearchParams();
  const active = ACTIVE.has(run.state);
  const tabs = RUN_TABS.filter((x) => x.available(run) !== false);
  const fallback: TabId = run.state === "completed" ? "overview" : run.state === "queued" ? "config" : "log";
  const requested = params.get("tab") as TabId | null;
  const tab = requested && tabs.some((x) => x.id === requested && x.available(run) === true) ? requested : fallback;
  const current = RUN_TABS.find((x) => x.id === tab)!;

  return (
    <Page>
      <Crumbs items={[{ label: t("Прогоны"), to: "/runs" }, { label: run.id }]} />
      <RunHeader run={run} />
      {active && <ActiveCard run={run} />}
      {run.state === "failed" && <FailedBox run={run} />}
      {run.state === "cancelled" && <CancelledBox run={run} />}
      <Tabs<TabId>
        label={t("Вкладки прогона")}
        value={tab}
        onChange={(next) => setParams(next === fallback ? {} : { tab: next }, { replace: true })}
        items={tabs.map((x) => ({ id: x.id, label: t(x.title), disabled: x.available(run) === "soon", title: x.available(run) === "soon" ? "ragglet-32" : undefined }))}
      />
      {current.Component && <current.Component run={run} />}
    </Page>
  );
}

function RunHeader({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const confirm = useConfirm();
  const cancel = useCancelTask();
  const [asScenario, setAsScenario] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const active = ACTIVE.has(run.state);
  const artifacts = useRunArtifacts(run.id, run.state !== "queued");
  const now = useNow(1000, run.state === "running");
  const dataset = splitDataset(run.dataset);
  const duration = run.started_at ? ((run.finished_at ? new Date(run.finished_at).getTime() : now) - new Date(run.started_at).getTime()) / 1000 : null;

  const onCancel = async () => {
    const ok = await confirm({
      title: t("Отменить прогон {scenario}?", { scenario: run.scenario }),
      body: run.state === "queued" ? t("Прогон уберут из очереди.") : t("Процесс прогона будет остановлен; уже записанные файлы останутся."),
      confirm: t("Отменить прогон"),
      cancel: t("Не отменять"),
      danger: true,
    });
    if (ok) cancel.mutate({ kind: "run", id: run.id }, { onError: (e) => !(e instanceof ApiProblem && e.status === 409) && setError(e) });
  };

  // Повторить: тот же snapshot содержимым и та же ссылка на библиотеку (ragglet-28 §9) — через dry-run.
  const onRepeat = async () => {
    setBusy(true);
    setError(null);
    try {
      const { scenario_yaml } = await unwrap(api.GET("/v1/runs/{run_id}", { params: { path: { run_id: run.id }, query: { include: "scenario" } } }));
      const text = scenario_yaml ?? "";
      let problem: S["Problem"] | null = null;
      const result = await launch<S["RunCheck"]>({
        check: async () => {
          const out = await dryRunScenario(text, run.scenario_id ?? null);
          if (!out.ok) problem = out.problem;
          return checkStep(out);
        },
        confirm: (warnings, check) => confirm({ title: t("Повторить всё равно?"), body: <WarningsBody warnings={warnings} check={check} />, confirm: t("Повторить"), wide: true }),
        submit: async () => {
          const out = await enqueueScenario(text, run.scenario_id ?? null);
          if (out.ok) return { kind: "accepted", id: out.accepted.id, position: out.accepted.position ?? null };
          problem = out.problem;
          return { kind: "problems" };
        },
      });
      if (result.kind === "started") {
        void qc.invalidateQueries({ queryKey: keys.tasks });
        void qc.invalidateQueries({ queryKey: keys.runs });
        navigate(`/runs/${encodeURIComponent(result.id)}`);
      } else if (problem) {
        setError(new ApiProblem(422, problem));
      }
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const files = new Set((artifacts.data ?? []).map((a) => a.name));
  const fileHref = (name: string) => `/v1/runs/${encodeURIComponent(run.id)}/artifacts/${encodeURIComponent(name)}`;

  return (
    <>
      <div className={s.head}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
          <div className={s.titleRow}>
            <h1 className="mono">{run.scenario}</h1>
            <StateChip state={run.state} />
            <span className="muted small mono">{run.id}</span>
          </div>
          <div className={s.facts}>
            <span>{t("поставлен {when}", { when: fmtWhen(run.enqueued_at) })}</span>
            {run.started_at && (
              <>
                <span>·</span>
                <span>{t("старт {time}", { time: fmtTime(run.started_at) })}</span>
              </>
            )}
            {duration !== null && (
              <>
                <span>·</span>
                <span className="num">{fmtDuration(duration)}</span>
              </>
            )}
            <span>·</span>
            <span>
              {dataset.source ? `${dataset.source} · ` : ""}
              {dataset.id} {dataset.revision && <span className="mono">@{dataset.revision.slice(0, 8)}</span>}
            </span>
            <span>·</span>
            <span className="mono">{run.collection}</span>
            <span>·</span>
            <ScenarioLink run={run} />
            {run.origin === "import" && (
              <Chip tone="tag" title={t("каталог без записи реестра: прогон из CLI или старый")}>
                {t("из CLI")}
              </Chip>
            )}
            {run.tags.map((tag) => (
              <Tag key={tag}>{tag}</Tag>
            ))}
          </div>
        </div>
        <div className={k.actions}>
          <Button onClick={() => void onRepeat()} disabled={active || busy} title={active ? t("доступно после завершения") : t("новый прогон из того же snapshot")}>
            {busy ? t("Проверка…") : t("Повторить")}
          </Button>
          <Button onClick={() => setAsScenario(true)}>{t("Открыть как сценарий")}</Button>
          {files.size > 0 && (
            <Menu
              trigger={<Button>{t("Скачать ▾")}</Button>}
              items={[
                { label: t("Все файлы (zip)"), onSelect: () => (window.location.href = `/v1/runs/${encodeURIComponent(run.id)}/artifacts.zip`) },
                ...["summary.json", "manifest.json", "scenario.snapshot.yaml", "task.log"]
                  .filter((name) => files.has(name))
                  .map((name) => ({ label: <span className="mono">{name}</span>, onSelect: () => (window.location.href = fileHref(name)) })),
              ]}
            />
          )}
          {active && (
            <Button variant="danger" onClick={() => void onCancel()} disabled={cancel.isPending}>
              {t("Отменить")}
            </Button>
          )}
        </div>
      </div>
      {error !== null && <ErrorBox error={error} />}
      {asScenario && (
        <NameDialog
          kind="scenario"
          open
          onClose={() => setAsScenario(false)}
          title={t("Открыть как сценарий")}
          initial={`${run.scenario}_from_run`}
          confirm={t("Создать сценарий")}
          onSubmit={(name) => fromRun(run.id, name)}
        >
          <span className="small muted">{t("Новый сценарий библиотеки из snapshot прогона, без run.artifacts_dir.")}</span>
        </NameDialog>
      )}
    </>
  );
}

function ScenarioLink({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  if (!run.scenario_id) return <span>{t("сценарий не из библиотеки")}</span>;
  if (run.scenario_file === "deleted") return <span>{t("сценарий {id} удалён, snapshot остался", { id: run.scenario_id })}</span>;
  return (
    <span>
      {rich(t("сценарий {link}"), { link: <Link to={`/scenarios/${encodeURIComponent(run.scenario_id)}`}>{run.scenario_id}</Link> })}
      {run.scenario_file === "changed" && <span title={t("файл сценария изменён после прогона")}> · {t("изменён после прогона")}</span>}
    </span>
  );
}

/** Активный прогон: шаги фаз, прогресс, предупреждения постановки, очередь (артборд RunRunning). */
function ActiveCard({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const tasks = useTasks();
  const plan = run.plan;
  const concurrency = plan?.load?.concurrency;
  const repeat = plan && run.progress && plan.queries_per_repeat > 0 ? Math.min(plan.repeats, Math.floor(Math.max(0, run.progress.done - 1) / plan.queries_per_repeat) + 1) : 1;
  const ahead = (tasks.data ?? []).filter((x) => x.state === "running" || (x.position !== null && run.position !== null && run.position !== undefined && x.position < run.position));

  return (
    <Card>
      <PhaseSteps
        kind="run"
        state={run.state}
        progress={run.progress}
        details={{
          queued:
            run.state === "queued"
              ? t("поставлен {time} · позиция {position}", { time: fmtTime(run.enqueued_at), position: run.position ?? "—" })
              : `${fmtTime(run.enqueued_at)} → ${fmtTime(run.started_at)}`,
          preparing: t(PHASE_HINT.run.preparing),
          progress: plan ? t("повтор {index} из {of}", { index: repeat, of: plan.repeats }) + (concurrency !== undefined ? ` · concurrency ${String(concurrency)}` : "") : undefined,
          finishing: t(PHASE_HINT.run.finishing),
        }}
      />
      {run.state === "running" && (
        <>
          <TaskProgress kind="run" state={run.state} progress={run.progress} plan={plan} startedAt={run.started_at} big />
          {run.phase === "progress" && (
            <span className="small muted">
              {t("Скорость — по событиям прогресса, оценка. Итоговый throughput_qps появится в summary. Метрик по ходу прогона нет: ядро сообщает только прогресс.")}
            </span>
          )}
        </>
      )}
      {run.state === "queued" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 13 }}>
          {ahead.length === 0 ? (
            <span>{t("Следующий в очереди — стартует, как только освободится стенд.")}</span>
          ) : (
            <>
              <span>{t("Впереди:")}</span>
              {ahead.map((task) => (
                <span key={`${task.kind}:${task.id}`}>
                  {t(KIND_LABEL[task.kind])}{" "}
                  <Link className="mono" to={taskHref(task.kind, task.id)}>
                    {task.name}
                  </Link>{" "}
                  <StateChip state={task.state} />
                  {task.progress && (
                    <span className="num muted">
                      {" "}
                      {fmtNumber(task.progress.done)} / {fmtNumber(task.progress.total)}
                    </span>
                  )}
                </span>
              ))}
            </>
          )}
        </div>
      )}
      {run.warnings.length > 0 && (
        <div style={{ display: "flex", gap: 8, alignItems: "baseline", flexWrap: "wrap" }}>
          <span className="small muted">{t("Preflight при постановке:")}</span>
          {run.warnings.map((w, i) => (
            <span key={i} className="small">
              <Chip tone="warn">{w.code}</Chip> <span className="muted">{w.message}</span>
            </span>
          ))}
        </div>
      )}
    </Card>
  );
}

function FailedBox({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const navigate = useNavigate();
  const error = run.error;
  if (!error) return <ProblemBox problem={{ title: t("Прогон упал"), hint: null, issues: [] }} />;
  return (
    <div className={k.problem} role="alert">
      <div className={k.problemTitle}>
        {error.type}: {error.message.split("\n")[0]}
      </div>
      {error.qid && (
        <span>{rich(t("На запросе {qid}."), { qid: <span className="mono">{error.qid}</span> })}</span>
      )}
      {error.problem && (
        <>
          <span>{rich(t("{title} — второй preflight при старте задачи."), { title: <b>{error.problem.title}</b> })}</span>
          {error.problem.hint && <span className="muted">{error.problem.hint}</span>}
          <IssueList issues={error.problem.issues} />
          {run.scenario_id && run.scenario_file !== "deleted" && (
            <div>
              <Button small onClick={() => navigate(`/scenarios/${encodeURIComponent(run.scenario_id!)}`, { state: { problem: error.problem } })}>
                {t("Открыть в редакторе")}
              </Button>
            </div>
          )}
        </>
      )}
      <span className="muted small">
        {run.finished_at ? t("Подробности — во вкладке «Лог» · упал в {time}.", { time: fmtTime(run.finished_at) }) : t("Подробности — во вкладке «Лог».")}
      </span>
    </div>
  );
}

function CancelledBox({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  return (
    <div className={s.banner}>
      <span>
        <b>{t("Отменён")}</b>
        {run.finished_at && <> {t("в {time}", { time: fmtTime(run.finished_at) })}</>}
        {run.progress ? (
          <>, {t("выполнено {done} / {total}", { done: fmtNumber(run.progress.done), total: fmtNumber(run.progress.total) })}</>
        ) : run.started_at ? (
          ""
        ) : (
          <> {t("до старта")}</>
        )}
        .
      </span>
    </div>
  );
}

function RunLog({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const active = ACTIVE.has(run.state);
  const tasks = useTasks();
  const next = (tasks.data ?? []).find((x) => x.state === "queued");
  return (
    <div className={s.split}>
      <LogView kind="run" id={run.id} active={active} tall={!active} downloadHref={`/v1/runs/${encodeURIComponent(run.id)}/artifacts/task.log`} />
      <Card title={active ? t("Условия на старте") : t("Условия")}>
        <KeyValue
          items={[
            [t("Датасет"), <span className="mono">{run.dataset}</span>],
            [t("Коллекция"), <span className="mono">{run.collection}</span>],
            run.plan && [
              t("План"),
              t("{queries} запросов × {repeats} = {total}", { queries: fmtNumber(run.plan.queries_per_repeat), repeats: run.plan.repeats, total: fmtNumber(run.plan.total) }),
            ],
            run.plan && [
              t("Нагрузка"),
              Object.entries(run.plan.load)
                .map(([key, v]) => (key === "type" ? String(v) : `${key} ${String(v)}`))
                .join(" · "),
            ],
            [t("Предупреждения"), run.warnings.length ? run.warnings.map((w) => w.code).join(", ") : t("нет")],
            active && next && [t("Следом"), <span className="mono">{next.name}</span>],
          ]}
        />
        <span className="small muted">{t("Полный manifest — во вкладке «Условия» (ragglet-32) и в файле manifest.json.")}</span>
      </Card>
    </div>
  );
}

function RunConfig({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const snapshot = useRunInclude(run.id, "scenario");
  if (snapshot.error) return <ErrorBox error={snapshot.error} />;
  if (!snapshot.data) return <span className="muted">{t("Загрузка snapshot…")}</span>;
  return (
    <Card
      title="scenario.snapshot"
      aside={
        <span className="small muted">
          {t("только чтение")}
          {run.scenario_file === "changed" && <> · {t("сценарий библиотеки изменён после прогона")}</>}
        </span>
      }
    >
      <div className={k.card} style={{ padding: 0, overflow: "hidden" }}>
        <YamlEditor label={t("Snapshot сценария")} value={snapshot.data.scenario_yaml ?? ""} readOnly minHeight={320} />
      </div>
    </Card>
  );
}

function RunFiles({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const artifacts = useRunArtifacts(run.id);
  const href = (name: string) => `/v1/runs/${encodeURIComponent(run.id)}/artifacts/${encodeURIComponent(name)}`;
  return (
    <Card
      title={t("Файлы")}
      aside={
        <FileButton href={`/v1/runs/${encodeURIComponent(run.id)}/artifacts.zip`} small>
          {t("Скачать всё (zip)")}
        </FileButton>
      }
    >
      {artifacts.error && <ErrorBox error={artifacts.error} />}
      {artifacts.data && artifacts.data.length === 0 ? (
        <Empty>{t("В каталоге прогона пока нет файлов.")}</Empty>
      ) : (
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Файлы прогона")}>
          <thead>
            <tr>
              <th>{t("Файл")}</th>
              <th className="r">{t("Размер")}</th>
              <th>{t("Изменён")}</th>
            </tr>
          </thead>
          <tbody>
            {artifacts.data?.map((a) => (
              <tr key={a.name}>
                <td>
                  <a className="mono" href={href(a.name)} download>
                    {a.name}
                  </a>
                </td>
                <td className="r num">{fmtBytes(a.size)}</td>
                <td className="muted">{fmtWhen(a.modified_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
