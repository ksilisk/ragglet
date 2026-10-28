import { useMemo } from "react";
import { Link, useSearchParams } from "react-router";
import type { S } from "../../api/client";
import { useDatasets, useIndexes, useRuns, useScenarios, useTasks, type RunsQuery } from "../../api/queries";
import { msg, useT } from "../../i18n/i18n";
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
// подписи — msg(), перевод при выводе
const STATES: Array<{ id: S["TaskState"] | ""; label: string }> = [
  { id: "", label: msg("все состояния") },
  { id: "completed", label: msg("готово") },
  { id: "failed", label: msg("ошибка") },
  { id: "cancelled", label: msg("отменён") },
  { id: "running", label: msg("выполняется") },
  { id: "queued", label: msg("в очереди") },
];
const PERIODS: Array<{ id: string; label: string; days: number | null }> = [
  { id: "", label: msg("всё время"), days: null },
  { id: "30d", label: msg("за 30 дней"), days: 30 },
  { id: "7d", label: msg("за 7 дней"), days: 7 },
];
// колонки, по которым сортирует сервер (RUN_SORT_KEYS реестра): id колонки → ключ сортировки
const SORTABLE: Record<string, string> = {
  start: "enqueued_at",
  duration: "duration",
  scenario: "scenario",
  ndcg: "ndcg",
  recall: "recall",
  p50: "p50",
  p95: "p95",
  qps: "qps",
  timeout: "timeout_rate",
};

/** Реестр прогонов по артборду Runs: фильтры и сортировка — в URL и на сервере, активные сверху. */
export default function RunsPage() {
  const t = useT();
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

  const th = (id: string, title: string, right = false) => {
    const key = SORTABLE[id];
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
        title={t("Прогоны")}
        count={runs.data ? total : undefined}
        actions={
          <>
            <label htmlFor="runs-k" className="small muted">
              {t("k для колонок")}
            </label>
            <select id="runs-k" className={`${k.inp} ${k.sm} num`} style={{ width: "auto" }} value={kv} onChange={(e) => set({ k: e.target.value === "10" ? null : e.target.value }, true)}>
              {[1, 3, 5, 10, 20, 50, 100].map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
            <Button disabled title="ragglet-32">
              {t("Сравнить · {count}", { count: 0 })}
            </Button>
          </>
        }
      />
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        {STATES.map((st) => (
          <FilterChip key={st.id || "all"} on={(params.get("state") ?? "") === st.id} onClick={() => set({ state: st.id || null })}>
            {t(st.label)}
          </FilterChip>
        ))}
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label={t("Сценарий")} value={params.get("scenario") ?? ""} onChange={(e) => set({ scenario: e.target.value })}>
          <option value="">{t("сценарий: любой")}</option>
          {scenarioNames.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label={t("Датасет")} value={params.get("dataset") ?? ""} onChange={(e) => set({ dataset: e.target.value })}>
          <option value="">{t("датасет: любой")}</option>
          {datasetOptions.map((d) => (
            <option key={d} value={d}>
              {fmtDataset(d)}
            </option>
          ))}
        </select>
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label={t("Коллекция")} value={params.get("collection") ?? ""} onChange={(e) => set({ collection: e.target.value })}>
          <option value="">{t("коллекция: любая")}</option>
          {collections.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label={t("Период")} value={period.id} onChange={(e) => set({ period: e.target.value })}>
          {PERIODS.map((p) => (
            <option key={p.id} value={p.id}>
              {t(p.label)}
            </option>
          ))}
        </select>
        <input className={`${k.inp} ${k.sm}`} style={{ width: 220 }} placeholder={t("тег или текст")} aria-label={t("Поиск")} defaultValue={params.get("q") ?? ""} onChange={(e) => set({ q: e.target.value })} />
      </div>
      {runs.error && <ErrorBox error={runs.error} />}
      <section className={`${k.card} ${k.flush}`}>
        <table className={`${k.tbl} ${k.compact} ${s.tight}`} aria-label={t("Реестр прогонов")}>
          <thead>
            <tr>
              <th>
                <input type="checkbox" disabled aria-label={t("Выбрать все")} title={t("Сравнение — ragglet-32")} />
              </th>
              <th>{t("Состояние")}</th>
              {th("scenario", t("Сценарий"))}
              {th("start", t("Старт"))}
              {th("duration", t("Длит."), true)}
              <th>{t("Датасет")}</th>
              <th>{t("Коллекция")}</th>
              {th("ndcg", `nDCG@${kv}`, true)}
              {th("recall", `Rec@${kv}`, true)}
              {th("p50", "p50", true)}
              {th("p95", "p95", true)}
              {th("qps", "QPS", true)}
              {th("timeout", "timeout", true)}
              <th>{t("Теги")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <RunRow key={r.id} r={r} live={progress.get(r.id)} />
            ))}
          </tbody>
        </table>
        {runs.data && rows.length === 0 && <Empty>{t("Прогонов с такими фильтрами нет.")}</Empty>}
        {runs.data && total > 0 && (
          <div className={k.tblFoot}>
            <span>
              {t("{from}–{to} из {total} · активные закреплены сверху", {
                from: fmtNumber(offset + 1),
                to: fmtNumber(Math.min(offset + LIMIT, total)),
                total: fmtNumber(total),
              })}
            </span>
            <span style={{ display: "flex", gap: 6 }}>
              <Button small disabled={offset === 0} onClick={() => set({ offset: String(Math.max(0, offset - LIMIT)) }, true)}>
                {t("‹ назад")}
              </Button>
              <Button small disabled={offset + LIMIT >= total} onClick={() => set({ offset: String(offset + LIMIT) }, true)}>
                {t("дальше ›")}
              </Button>
            </span>
          </div>
        )}
      </section>
      <span className="small muted">
        {t("Старые прогоны (до ragglet-21) без nDCG показывают «—»: колонка берётся из summary как есть, ничего не пересчитывается.")}
      </span>
    </Page>
  );
}

function RunRow({ r, live }: { r: S["RunItem"]; live?: S["TaskItem"] }) {
  const t = useT();
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
        <input type="checkbox" disabled aria-label={t("Выбрать {id}", { id: r.id })} title={t("Сравнение — ragglet-32")} />
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
          {r.tags.slice(0, 3).map((tag) => (
            <Tag key={tag}>{tag}</Tag>
          ))}
          {r.tags.length > 3 && <span className="small muted">+{r.tags.length - 3}</span>}
        </span>
      </td>
    </tr>
  );
}
