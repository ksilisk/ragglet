import type { ReactNode } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { useBuild, useBuildProfileYaml, useCancelTask, useTasks } from "../../api/queries";
import { YamlEditor } from "../../editor/YamlEditor";
import { fmtDataset, fmtDuration, fmtNumber, fmtTime, fmtWhen } from "../../lib/format";
import { compactNames } from "../../lib/names";
import { PHASE_HINT } from "../../lib/task";
import { useNow } from "../../lib/useNow";
import { Button } from "../../ui/Button";
import { Chip } from "../../ui/Chip";
import { useConfirm } from "../../ui/Dialog";
import k from "../../ui/kit.module.css";
import { Card, Crumbs, Empty, KeyValue, Page } from "../../ui/Layout";
import { ErrorBox, IssueList } from "../../ui/ProblemView";
import { Tabs } from "../../ui/Tabs";
import { ACTIVE, KIND_LABEL, StateChip } from "../../ui/states";
import { TaskProgress } from "../../ui/TaskProgress";
import { hostOf } from "../indexes/IndexesPage";
import { LIBRARY } from "../library/library";
import { LogView } from "../task/LogView";
import { PhaseSteps } from "../task/PhaseSteps";
import s from "../task/task.module.css";

type TabId = "result" | "log" | "profile";

/** Страница сборки — младшая сестра страницы прогона (ragglet-15): состояние, фазы, лог, результат, ошибка. */
export default function BuildPage() {
  const { id = "" } = useParams();
  const build = useBuild(id);
  const crumbs = <Crumbs items={[{ label: "Индексы", to: "/indexes" }, { label: "Сборки" }, { label: id }]} />;
  if (build.error) {
    const notFound = build.error instanceof ApiProblem && build.error.status === 404;
    return (
      <Page>
        {crumbs}
        {notFound ? <Empty>Сборки {id} нет в реестре.</Empty> : <ErrorBox error={build.error} />}
      </Page>
    );
  }
  if (!build.data) {
    return (
      <Page>
        {crumbs}
        <span className="muted">Загрузка…</span>
      </Page>
    );
  }
  return <BuildView build={build.data} />;
}

function BuildView({ build }: { build: S["BuildStatus"] }) {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const confirm = useConfirm();
  const cancel = useCancelTask();
  const tasks = useTasks();
  const active = ACTIVE.has(build.state);
  const now = useNow(1000, build.state === "running");
  const fallback: TabId = build.state === "completed" ? "result" : "log";
  const tab = (["result", "log", "profile"] as const).find((t) => t === params.get("tab") && (t !== "result" || build.state === "completed")) ?? fallback;
  const duration = build.started_at ? ((build.finished_at ? new Date(build.finished_at).getTime() : now) - new Date(build.started_at).getTime()) / 1000 : null;
  const profileRoute = build.profile_id ? LIBRARY.profile.route(build.profile_id) : null;
  const ahead = (tasks.data ?? []).filter((t) => t.state === "running" || (t.position !== null && build.position !== null && build.position !== undefined && t.position < build.position));

  const onCancel = async () => {
    const ok = await confirm({
      title: `Отменить сборку ${build.profile}?`,
      body: build.state === "queued" ? "Сборку уберут из очереди." : "Процесс сборки будет остановлен; коллекция останется недособранной, без метаданных ragglet.",
      confirm: "Отменить сборку",
      cancel: "Не отменять",
      danger: true,
    });
    if (ok) cancel.mutate({ kind: "build", id: build.id });
  };

  return (
    <Page>
      <Crumbs items={[{ label: "Индексы", to: "/indexes" }, { label: "Сборки" }, { label: build.id }]} />
      <div className={s.head}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
          <div className={s.titleRow}>
            <h1 className="mono">{build.profile}</h1>
            <StateChip state={build.state} />
            <span className="muted small mono">{build.id}</span>
          </div>
          <div className={s.facts}>
            <span>поставлена {fmtWhen(build.enqueued_at)}</span>
            {build.started_at && (
              <>
                <span>·</span>
                <span>старт {fmtTime(build.started_at)}</span>
              </>
            )}
            {duration !== null && (
              <>
                <span>·</span>
                <span className="num">{fmtDuration(duration)}</span>
              </>
            )}
            <span>·</span>
            <span>{fmtDataset(build.dataset)}</span>
            <span>·</span>
            <span className="mono">{compactNames(build.targets.map((t) => t.collection))}</span>
          </div>
        </div>
        <div className={k.actions}>
          {profileRoute && <Button onClick={() => navigate(profileRoute)}>Открыть профиль</Button>}
          {profileRoute && !active && <Button onClick={() => navigate(`${profileRoute}?build=1`)}>Пересобрать</Button>}
          {active && (
            <Button variant="danger" onClick={() => void onCancel()} disabled={cancel.isPending}>
              Отменить
            </Button>
          )}
        </div>
      </div>

      {active && (
        <Card>
          <PhaseSteps
            kind="build"
            state={build.state}
            progress={build.progress}
            details={{
              queued: build.state === "queued" ? `поставлена ${fmtTime(build.enqueued_at)} · позиция ${build.position ?? "—"}` : `${fmtTime(build.enqueued_at)} → ${fmtTime(build.started_at)}`,
              preparing: PHASE_HINT.build.preparing,
              progress: "эмбеддинг и запись в Qdrant",
              finishing: PHASE_HINT.build.finishing,
            }}
          />
          {build.state === "running" && <TaskProgress kind="build" state={build.state} progress={build.progress} startedAt={build.started_at} big />}
          {build.state === "queued" && ahead.length > 0 && (
            <span style={{ fontSize: 13 }}>
              Впереди: {ahead.map((t) => `${KIND_LABEL[t.kind].toLowerCase()} ${t.name}`).join(", ")}
            </span>
          )}
        </Card>
      )}
      {build.state === "failed" && (
        <div className={k.problem} role="alert">
          <div className={k.problemTitle}>
            {build.error?.type ?? "Ошибка"}: {build.error?.message.split("\n")[0]}
          </div>
          {build.error?.problem && <IssueList issues={build.error.problem.issues} />}
          <span>Упавшая сборка оставляет коллекцию без метаданных ragglet: прогоны на ней не пройдут preflight. Пересоберите индекс.</span>
          <span className="muted small">Подробности — во вкладке «Лог».</span>
        </div>
      )}
      {build.state === "cancelled" && (
        <div className={s.banner}>
          <span>
            <b>Отменена</b>
            {build.finished_at ? ` в ${fmtTime(build.finished_at)}` : ""}
            {build.progress ? `, обработано ${fmtNumber(build.progress.done)} / ${fmtNumber(build.progress.total)} документов` : ""}. Коллекция могла остаться
            недособранной.
          </span>
        </div>
      )}
      {build.warnings.length > 0 && (
        <div style={{ display: "flex", gap: 8, alignItems: "baseline", flexWrap: "wrap" }}>
          <span className="small muted">Preflight при постановке:</span>
          {build.warnings.map((w, i) => (
            <span key={i} className="small">
              <Chip tone="warn">{w.code}</Chip> <span className="muted">{w.message}</span>
            </span>
          ))}
        </div>
      )}

      <Tabs<TabId>
        label="Вкладки сборки"
        value={tab}
        onChange={(next) => setParams(next === fallback ? {} : { tab: next }, { replace: true })}
        items={[
          ...(build.state === "completed" ? [{ id: "result" as const, label: "Результат" }] : []),
          { id: "log", label: "Лог" },
          { id: "profile", label: "Профиль" },
        ]}
      />
      {tab === "result" && <BuildResult build={build} />}
      {tab === "log" && <LogView kind="build" id={build.id} active={active} tall={!active} />}
      {tab === "profile" && <BuildProfile build={build} />}
    </Page>
  );
}

function BuildResult({ build }: { build: S["BuildStatus"] }) {
  const result = build.result ?? {};
  const documents = result.documents as number | undefined;
  const { documents: _documents, ...rest } = result;
  void _documents;
  return (
    <div className={s.split}>
      <Card title="Результат">
        <KeyValue
          items={[
            ["Датасет", fmtDataset(build.dataset)],
            documents !== undefined && ["Документы", <span className="num">{fmtNumber(documents)}</span>],
            ...build.targets.map((t): [string, ReactNode] => ["Цель", <span className="mono">{`${hostOf(t.endpoint)} / ${t.collection}`}</span>]),
          ]}
        />
      </Card>
      <Card title="Записанные метаданные">
        {Object.keys(rest).length ? (
          <pre className="mono small" style={{ margin: 0, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
            {JSON.stringify(rest, null, 2)}
          </pre>
        ) : (
          <span className="muted small">Нет данных.</span>
        )}
      </Card>
    </div>
  );
}

function BuildProfile({ build }: { build: S["BuildStatus"] }) {
  const query = useBuildProfileYaml(build.id);
  if (query.error) return <ErrorBox error={query.error} />;
  if (!query.data) return <span className="muted">Загрузка…</span>;
  return (
    <Card title="Профиль сборки" aside={<span className="small muted">только чтение</span>}>
      <div className={k.card} style={{ padding: 0, overflow: "hidden" }}>
        <YamlEditor label="Профиль сборки" value={query.data.profile_yaml ?? ""} readOnly minHeight={240} />
      </div>
    </Card>
  );
}
