import { useMemo } from "react";
import { Link, useSearchParams } from "react-router";
import type { S } from "../../api/client";
import { useDatasets, useIndexes, useRuns, useScenarios, useTasks, type RunsQuery } from "../../api/queries";
import { fmtDataset, fmtDuration, fmtMetric, fmtMs, fmtNumber, fmtPercent, fmtQps, fmtWhen } from "../../lib/format";
import { useNow } from "../../lib/useNow";
import { Button } from "../../ui/Button";
import { FilterChip, Tag } from "../../ui/Chip";
import { cx } from "../../ui/cx";
import k from "../../ui/kit.module.css";
import { Empty, Page, PageHead } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { ACTIVE, StateChip } from "../../ui/states";
import s from "../scenarios/scenarios.module.css";

const LIMIT = 50;
const STATES: Array<{ id: S["TaskState"] | ""; label: string }> = [
  { id: "", label: "все состояния" },
  { id: "completed", label: "готово" },
  { id: "failed", label: "ошибка" },
  { id: "cancelled", label: "отменён" },
  { id: "running", label: "выполняется" },
  { id: "queued", label: "в очереди" },
];
const PERIODS: Array<{ id: string; label: string; days: number | null }> = [
  { id: "", label: "всё время", days: null },
  { id: "30d", label: "за 30 дней", days: 30 },
  { id: "7d", label: "за 7 дней", days: 7 },
];
// колонки, по которым сортирует сервер (RUN_SORT_KEYS реестра)
const SORTABLE: Record<string, string> = {
  "Старт": "enqueued_at",
  "Длит.": "duration",
  "Сценарий": "scenario",
  nDCG: "ndcg",
  Rec: "recall",
  p50: "p50",
  p95: "p95",
  QPS: "qps",
  timeout: "timeout_rate",
};

/** Реестр прогонов по артборду Runs: фильтры и сортировка — в URL и на сервере, активные сверху. */
export default function RunsPage() {
  const [params, setParams] = useSearchParams();
  const kv = Number(params.get("k") ?? 10) || 10;
  const offset = Number(params.get("offset") ?? 0) || 0;
  const sort = params.get("sort") ?? "-enqueued_at";
  const period = PERIODS.find((p) => p.id === (params.get("period") ?? "")) ?? PERIODS[0]!;
  const since = useMemo(() => (period.days ? new Date(Date.now() - period.days * 86_400_000).toISOString() : undefined), [period.days]);

  const query: RunsQuery = {
    k: kv,
    offset,
    limit: LIMIT,
    sort,
    state: (params.get("state") as S["TaskState"] | null) || undefined,
    scenario: params.get("scenario") || undefined,
    dataset: params.get("dataset") || undefined,
    collection: params.get("collection") || undefined,
    q: params.get("q") || undefined,
    since,
  };
  const runs = useRuns(query);
  const tasks = useTasks();
  const scenarios = useScenarios();
  const datasets = useDatasets();
  const indexes = useIndexes();

  const set = (patch: Record<string, string | null>, keepOffset = false) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        for (const [key, value] of Object.entries(patch)) {
          if (value) next.set(key, value);
          else next.delete(key);
        }
        if (!keepOffset) next.delete("offset");
        return next;
      },
      { replace: true },
    );

  const progress = new Map((tasks.data ?? []).filter((t) => t.kind === "run").map((t) => [t.id, t]));
  const items = runs.data?.items ?? [];
  // активные закреплены сверху страницы
  const rows = [...items.filter((r) => ACTIVE.has(r.state)), ...items.filter((r) => !ACTIVE.has(r.state))];
  const total = runs.data?.total ?? 0;
  const scenarioNames = [...new Set((scenarios.data ?? []).map((x) => x.id))].sort();
  const datasetOptions = [
    ...(datasets.data?.hf ?? []).map((d) => `hf:${d.id}`),
    ...(datasets.data?.local ?? []).map((d) => `local:${d.id}`),
  ];
  const collections = [...new Set((indexes.data?.collections ?? []).map((c) => c.name))].sort();

  const th = (label: string, right = false, title = label) => {
    const key = SORTABLE[label];
    if (!key) return <th className={right ? "r" : undefined}>{title}</th>;
    const dir = sort === key ? "↑" : sort === `-${key}` ? "↓" : "";
    return (
      <th className={cx(right && "r", k.sortable)} aria-sort={dir === "↑" ? "ascending" : dir === "↓" ? "descending" : undefined} onClick={() => set({ sort: sort === `-${key}` ? key : `-${key}` })}>
        {title}
        {dir && ` ${dir}`}
      </th>
    );
  };

  return (
    <Page>
      <PageHead
        title="Прогоны"
        count={runs.data ? total : undefined}
        actions={
          <>
            <label htmlFor="runs-k" className="small muted">
              k для колонок
            </label>
            <select id="runs-k" className={`${k.inp} ${k.sm} num`} style={{ width: "auto" }} value={kv} onChange={(e) => set({ k: e.target.value === "10" ? null : e.target.value }, true)}>
              {[1, 3, 5, 10, 20, 50, 100].map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
            <Button disabled title="ragglet-32">
              Сравнить · 0
            </Button>
          </>
        }
      />
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        {STATES.map((st) => (
          <FilterChip key={st.id || "all"} on={(params.get("state") ?? "") === st.id} onClick={() => set({ state: st.id || null })}>
            {st.label}
          </FilterChip>
        ))}
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label="Сценарий" value={params.get("scenario") ?? ""} onChange={(e) => set({ scenario: e.target.value })}>
          <option value="">сценарий: любой</option>
          {scenarioNames.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label="Датасет" value={params.get("dataset") ?? ""} onChange={(e) => set({ dataset: e.target.value })}>
          <option value="">датасет: любой</option>
          {datasetOptions.map((d) => (
            <option key={d} value={d}>
              {fmtDataset(d)}
            </option>
          ))}
        </select>
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label="Коллекция" value={params.get("collection") ?? ""} onChange={(e) => set({ collection: e.target.value })}>
          <option value="">коллекция: любая</option>
          {collections.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label="Период" value={period.id} onChange={(e) => set({ period: e.target.value })}>
          {PERIODS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <input className={`${k.inp} ${k.sm}`} style={{ width: 220 }} placeholder="тег или текст" aria-label="Поиск" defaultValue={params.get("q") ?? ""} onChange={(e) => set({ q: e.target.value })} />
      </div>
      {runs.error && <ErrorBox error={runs.error} />}
      <section className={`${k.card} ${k.flush}`}>
        <table className={`${k.tbl} ${k.compact} ${s.tight}`} aria-label="Реестр прогонов">
          <thead>
            <tr>
              <th>
                <input type="checkbox" disabled aria-label="Выбрать все" title="Сравнение — ragglet-32" />
              </th>
              <th>Состояние</th>
              {th("Сценарий")}
              {th("Старт")}
              {th("Длит.", true)}
              <th>Датасет</th>
              <th>Коллекция</th>
              {th("nDCG", true, `nDCG@${kv}`)}
              {th("Rec", true, `Rec@${kv}`)}
              {th("p50", true)}
              {th("p95", true)}
              {th("QPS", true)}
              {th("timeout", true)}
              <th>Теги</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <RunRow key={r.id} r={r} live={progress.get(r.id)} />
            ))}
          </tbody>
        </table>
        {runs.data && rows.length === 0 && <Empty>Прогонов с такими фильтрами нет.</Empty>}
        {runs.data && total > 0 && (
          <div className={k.tblFoot}>
            <span>
              {fmtNumber(offset + 1)}–{fmtNumber(Math.min(offset + LIMIT, total))} из {fmtNumber(total)} · активные закреплены сверху
            </span>
            <span style={{ display: "flex", gap: 6 }}>
              <Button small disabled={offset === 0} onClick={() => set({ offset: String(Math.max(0, offset - LIMIT)) }, true)}>
                ‹ назад
              </Button>
              <Button small disabled={offset + LIMIT >= total} onClick={() => set({ offset: String(offset + LIMIT) }, true)}>
                дальше ›
              </Button>
            </span>
          </div>
        )}
      </section>
      <span className="small muted">Старые прогоны (до ragglet-21) без nDCG показывают «—»: колонка берётся из summary как есть, ничего не пересчитывается.</span>
    </Page>
  );
}

function RunRow({ r, live }: { r: S["RunItem"]; live?: S["TaskItem"] }) {
  const running = r.state === "running";
  const now = useNow(1000, running);
  const elapsed = running && r.started_at ? (now - new Date(r.started_at).getTime()) / 1000 : r.duration_s;
  const fraction = live?.progress && live.progress.total > 0 ? live.progress.done / live.progress.total : null;
  const href = `/runs/${encodeURIComponent(r.id)}`;
  const metric = (value: number | null | undefined, format: (x: number | null) => string) => (
    <td className={value === null || value === undefined ? "r muted" : "r num"}>{format(value ?? null)}</td>
  );
  return (
    <tr className={ACTIVE.has(r.state) ? k.act : undefined}>
      <td>
        <input type="checkbox" disabled aria-label={`Выбрать ${r.id}`} title="Сравнение — ragglet-32" />
      </td>
      <td>
        <StateChip state={r.state} suffix={r.state === "queued" && live?.position ? String(live.position) : undefined} />
      </td>
      <td>
        <Link className={`${k.rowLink} mono`} to={href} title={r.id}>
          {r.scenario}
        </Link>
      </td>
      <td className="muted nowrap">{r.started_at ? fmtWhen(r.started_at) : r.state === "queued" ? "—" : fmtWhen(r.enqueued_at)}</td>
      <td className="r num nowrap">
        {running && fraction !== null && (
          <span className={k.bar} style={{ display: "inline-block", width: 60, height: 6, marginRight: 6, verticalAlign: "middle" }}>
            <div style={{ width: `${Math.round(fraction * 100)}%` }} />
          </span>
        )}
        {fmtDuration(elapsed)}
      </td>
      <td className="nowrap">{fmtDataset(r.dataset)}</td>
      <td className="mono">{r.collection}</td>
      {metric(r.ndcg, fmtMetric)}
      {metric(r.recall, fmtMetric)}
      {metric(r.p50_ms, fmtMs)}
      {metric(r.p95_ms, fmtMs)}
      {metric(r.qps, fmtQps)}
      {metric(r.timeout_rate, fmtPercent)}
      <td>
        <span style={{ display: "inline-flex", gap: 4, flexWrap: "wrap" }}>
          {r.tags.slice(0, 3).map((t) => (
            <Tag key={t}>{t}</Tag>
          ))}
          {r.tags.length > 3 && <span className="small muted">+{r.tags.length - 3}</span>}
        </span>
      </td>
    </tr>
  );
}
