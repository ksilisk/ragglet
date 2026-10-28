import { Fragment, useMemo, useRef, useState } from "react";
import { downloadCsv, fileName } from "../../analysis/csv";
import { compareDiffTable, compareMetricsTable } from "../../analysis/tables";
import { CsvButton, ExportMenu } from "../../charts/ExportMenu";
import { Link, useSearchParams } from "react-router";
import {
  LATENCY_GROUP,
  commonK,
  delta,
  diffGroups,
  isNoisyEnv,
  metricRows,
  qualityAxes,
  qualityByRun,
  runLabels,
  showValue,
  stageRowsByRun,
  type Compare,
  type CompareRun,
  type Kind,
  type MetricRow,
} from "../../analysis/compare";
import { useQualityMode } from "../../analysis/qualityMode";
import { METRIC_LABEL, type Metric } from "../../analysis/summary";
import { useCompare } from "../../api/queries";
import { Chart, type ChartExport } from "../../charts/Chart";
import { qualityByKOption } from "../../charts/options/qualityByK";
import { stageBarHeight, stageBarOption } from "../../charts/options/stageBar";
import { LIGHT } from "../../charts/theme";
import { msg, useT } from "../../i18n/i18n";
import { fmtDataset, fmtMetric, fmtMs, fmtNumber, fmtPercent, fmtQps, fmtWhen } from "../../lib/format";
import { cx } from "../../ui/cx";
import k from "../../ui/kit.module.css";
import { Card, Crumbs, Empty, Page, PageHead } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { Segmented } from "../../ui/Segmented";
import { StateChip } from "../../ui/states";
import { QualityModeToggle } from "../run/resultCards";
import r from "../run/results.module.css";
import c from "./compare.module.css";

const KIND_TITLE: Record<Kind, string> = {
  config: msg("Настройка RAG (config)"),
  data: msg("Данные (data)"),
  env: msg("Окружение (env)"),
};

/**
 * Сравнение прогонов (ragglet-32 §9): /compare?ids=a,b[,…] из чекбоксов списка прогонов. Плашка
 * сопоставимости, различия конфига по классам, метрики бок о бок с Δ к базовому (первому), общие
 * графики «качество по k» и «задержки по стадиям».
 */
export default function ComparePage() {
  const t = useT();
  const [params, setParams] = useSearchParams();
  const ids = useMemo(() => (params.get("ids") ?? "").split(",").map((x) => x.trim()).filter(Boolean), [params]);
  const compare = useCompare(ids);
  const setIds = (next: string[]) =>
    setParams(
      (prev) => {
        const out = new URLSearchParams(prev);
        out.set("ids", next.join(","));
        return out;
      },
      { replace: true },
    );

  return (
    <Page>
      <Crumbs items={[{ label: t("Прогоны"), to: "/runs" }, { label: t("Сравнение") }]} />
      <PageHead title={t("Сравнение прогонов")} count={ids.length || undefined} />
      {ids.length < 2 ? (
        <Empty>
          {t("Нужно хотя бы два прогона: отметьте их в списке прогонов и нажмите «Сравнить».")} <Link to="/runs">{t("К прогонам")}</Link>
        </Empty>
      ) : compare.error ? (
        <ErrorBox error={compare.error} />
      ) : !compare.data ? (
        <span className="muted">{t("Загрузка сравнения…")}</span>
      ) : (
        <CompareView data={compare.data} onIds={setIds} />
      )}
    </Page>
  );
}

function CompareView({ data, onIds }: { data: Compare; onIds: (ids: string[]) => void }) {
  const runs = data.runs;
  const labels = useMemo(() => runLabels(runs), [runs]);
  const ids = runs.map((run) => run.id);
  return (
    <>
      <RunChips runs={runs} labels={labels} onRemove={(id) => onIds(ids.filter((x) => x !== id))} onBase={(id) => onIds([id, ...ids.filter((x) => x !== id)])} />
      <Verdict data={data} />
      <MetricsTable runs={runs} labels={labels} />
      <div className={r.grid2}>
        <QualityCompare runs={runs} labels={labels} />
        <StagesCompare runs={runs} labels={labels} />
      </div>
      <Differences data={data} labels={labels} />
    </>
  );
}

function Swatch({ slot }: { slot: number }) {
  // цвет слота — тот же, что у линии прогона на графике (токены --series-N)
  return <span className={c.swatch} style={{ background: `var(--series-${(slot % LIGHT.series.length) + 1})` }} aria-hidden />;
}

function RunChips({ runs, labels, onRemove, onBase }: { runs: CompareRun[]; labels: string[]; onRemove: (id: string) => void; onBase: (id: string) => void }) {
  const t = useT();
  return (
    <div className={c.runs}>
      {runs.map((run, i) => (
        <div key={run.id} className={c.run}>
          <div className={c.runHead}>
            <Swatch slot={i} />
            <Link className={`${k.rowLink} mono`} to={`/runs/${encodeURIComponent(run.id)}`} title={run.id}>
              {labels[i]}
            </Link>
            <StateChip state={run.state} />
            {i === 0 && <span className={c.base}>{t("базовый")}</span>}
          </div>
          <div className="small muted">
            {fmtDataset(run.dataset)}
            {" "}
            <span className="mono">{run.collection}</span>
            {" "}
            {fmtWhen(run.enqueued_at)}
            {!run.has_manifest && <span>{" "}{t("без manifest")}</span>}
          </div>
          <div className={c.runActions}>
            {i > 0 && (
              <button type="button" className={r.more} onClick={() => onBase(run.id)}>
                {t("сделать базовым")}
              </button>
            )}
            {runs.length > 2 && (
              <button type="button" className={r.more} onClick={() => onRemove(run.id)}>
                {t("убрать")}
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Плашка сопоставимости по вердикту сервера (правило ragglet-23 живёт в bench/compare.py). */
function Verdict({ data }: { data: Compare }) {
  const t = useT();
  const paths = (kind: string, pred: (p: string) => boolean = () => true) =>
    data.differences.filter((d) => d.kind === kind && pred(d.path)).map((d) => d.path);
  const latencyPaths = paths("env", (p) => !data.confounders.includes(p) && !isNoisyEnv(p));
  const notes: Array<{ tone: "ok" | "warn" | "bad"; title: string; list?: string[] }> = [];
  if (data.missing_manifest.length > 0) {
    notes.push({ tone: "warn", title: t("У прогонов нет manifest: данные и окружение не проверить, сравнение только по конфигу."), list: data.missing_manifest });
  } else {
    if (data.quality_comparable === false) notes.push({ tone: "bad", title: t("Датасеты разные — качество несравнимо."), list: paths("data") });
    if (data.latency_comparable === false)
      notes.push({ tone: "bad", title: t("Окружение разное — задержки несравнимы."), list: latencyPaths.length ? latencyPaths : paths("env", (p) => !data.confounders.includes(p)) });
    if (data.confounders.length > 0) notes.push({ tone: "warn", title: t("Разный код или окружение сборки индекса — часть разницы может быть от них."), list: data.confounders });
    if (notes.length === 0) notes.push({ tone: "ok", title: t("Различаются только настройки RAG: разница метрик относится к ним.") });
  }
  return (
    <div className={c.verdicts}>
      {notes.map((note, i) => (
        <div key={i} className={cx(c.verdict, c[note.tone])} role={note.tone === "ok" ? "status" : "alert"}>
          <b>{note.title}</b>
          {note.list && note.list.length > 0 && <span className="mono small">{note.list.join(", ")}</span>}
        </div>
      ))}
    </div>
  );
}

function fmt(row: MetricRow, v: number | null): string {
  if (row.format === "metric") return fmtMetric(v);
  if (row.format === "ms") return fmtMs(v);
  if (row.format === "qps") return fmtQps(v);
  if (row.format === "seconds") return v === null ? "—" : `${fmtNumber(v / 1000, 1)}`;
  return fmtPercent(v);
}

function MetricsTable({ runs, labels }: { runs: CompareRun[]; labels: string[] }) {
  const t = useT();
  const ks = commonK(runs);
  const [kSel, setK] = useState<number | null>(null);
  const kv = kSel && ks.includes(kSel) ? kSel : ks.includes(10) ? 10 : (ks[ks.length - 1] ?? 10);
  const [stagesOpen, setStagesOpen] = useState(false);
  const rows = metricRows(runs, kv);
  const groups = [...new Set(rows.map((x) => x.group))];
  return (
    <Card
      title={t("Метрики бок о бок")}
      aside={
        <span className={r.toolbar}>
          {ks.length > 0 && (
            <span className={r.toolbar} style={{ gap: 6 }}>
              <label htmlFor="compare-k" className="small muted">
                k
              </label>
              <select id="compare-k" className={`${k.inp} ${k.sm} num`} style={{ width: "auto" }} value={kv} onChange={(e) => setK(Number(e.target.value))}>
                {ks.map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            </span>
          )}
          <CsvButton
            onClick={() => {
              const table = compareMetricsTable(runs, rows);
              downloadCsv(`${fileName("compare", `${runs.length}runs`, "metrics", `k${kv}`)}.csv`, table.header, table.rows);
            }}
          />
        </span>
      }
    >
      <div className={r.scroll}>
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Метрики бок о бок")}>
          <thead>
            <tr>
              <th>{t("Метрика")}</th>
              {runs.map((run, i) => (
                <th key={run.id} className="r">
                  <span className={c.thRun}>
                    <Swatch slot={i} />
                    {labels[i]}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => (
              <Fragment key={group}>
                <tr className={c.groupRow}>
                  <td colSpan={runs.length + 1}>{t(group)}</td>
                </tr>
                {rows
                  .filter((row) => row.group === group && (!row.stage || stagesOpen))
                  .map((row) => (
                    <tr key={row.key}>
                      <td className="nowrap">{row.label}</td>
                      {row.values.map((v, i) => {
                        const raw = i > 0 ? delta(row.values[0] ?? null, v) : null;
                        const d = raw && raw.abs !== 0 ? raw : null; // равные значения — без Δ
                        const better = d && !d.noise && d.abs !== 0 && row.better ? (row.better === "higher") === d.abs > 0 : null;
                        return (
                          <td key={i} className="r num nowrap">
                            {fmt(row, v?.mean ?? null)}
                            {v?.std !== null && v?.std !== undefined && <span className={r.std}> ± {fmt(row, v.std)}</span>}
                            {d && (
                              <div className={cx(c.delta, d.noise && c.noise, better === true && c.better, better === false && c.worse)} title={d.noise ? t("в пределах разброса повторов: |Δ| не больше суммы std") : undefined}>
                                {d.abs > 0 ? "+" : d.abs < 0 ? "−" : ""}
                                {fmt(row, Math.abs(d.abs))}
                                {d.rel !== null && ` (${d.rel > 0 ? "+" : d.rel < 0 ? "−" : ""}${fmtPercent(Math.abs(d.rel))})`}
                              </div>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                {group === LATENCY_GROUP && rows.some((row) => row.stage) && (
                  <tr>
                    <td colSpan={runs.length + 1}>
                      <button type="button" className={r.more} onClick={() => setStagesOpen((x) => !x)}>
                        {stagesOpen ? t("скрыть стадии") : t("показать p50 по стадиям")}
                      </button>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <span className={r.caption}>
        {t("Под значением — Δ к базовому прогону. Серым — в пределах разброса повторов: |Δ| не больше суммы std; это не тест значимости.")}
      </span>
    </Card>
  );
}

function QualityCompare({ runs, labels }: { runs: CompareRun[]; labels: string[] }) {
  const t = useT();
  const exp = useRef<ChartExport>(null);
  const [mode, setMode] = useQualityMode();
  const { ks, metrics } = useMemo(() => qualityAxes(runs), [runs]);
  const [metricSel, setMetric] = useState<Metric>("ndcg");
  const metric = metrics.includes(metricSel) ? metricSel : (metrics[0] ?? "ndcg");
  const data = useMemo(() => qualityByRun(runs, metric, ks, mode, labels), [runs, metric, ks, mode, labels]);
  return (
    <Card
      title={t("Качество по k")}
      aside={
        <span className={r.toolbar}>
          <select className={`${k.inp} ${k.sm}`} style={{ width: "auto" }} aria-label={t("Метрика")} value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
            {metrics.map((m) => (
              <option key={m} value={m}>
                {METRIC_LABEL[m]}
              </option>
            ))}
          </select>
          <QualityModeToggle mode={mode} onChange={setMode} />
          {ks.length > 0 && <ExportMenu target={exp} />}
        </span>
      }
    >
      {ks.length === 0 ? (
        <Empty>{t("Метрик doc-уровня у этих прогонов нет.")}</Empty>
      ) : (
        <Chart build={qualityByKOption} data={data} height={300} label={t("Качество по k по прогонам")} exportRef={exp} exportName={fileName("compare", `${runs.length}runs`, "quality-k", metric)} />
      )}
      <span className={r.caption}>{t("Линия — прогон, цвет — как у прогона в шапке. Усы — std по повторам.")}</span>
    </Card>
  );
}

function StagesCompare({ runs, labels }: { runs: CompareRun[]; labels: string[] }) {
  const t = useT();
  const exp = useRef<ChartExport>(null);
  const [p, setP] = useState<"50" | "95" | "99">("50");
  const data = useMemo(() => ({ ...stageRowsByRun(runs, Number(p), labels), percentile: Number(p) }), [runs, p, labels]);
  return (
    <Card
      title={t("Задержки по стадиям")}
      aside={
        <span className={r.toolbar}>
          <Segmented
            label={t("Перцентиль")}
            value={p}
            onChange={setP}
            items={[
              { id: "50", label: "p50" },
              { id: "95", label: "p95" },
              { id: "99", label: "p99" },
            ]}
          />
          <ExportMenu target={exp} />
        </span>
      }
    >
      <Chart
        build={stageBarOption}
        data={data}
        height={stageBarHeight(runs.length)}
        label={t("Задержки по стадиям по прогонам")}
        exportRef={exp}
        exportName={fileName("compare", `${runs.length}runs`, "latency-stages", `p${p}`)}
      />
      <span className={r.caption}>{t("Строка — прогон, цвета стадий те же, что в «Обзоре». Отметка — total; сумма медиан стадий не равна медиане total.")}</span>
    </Card>
  );
}

function Differences({ data, labels }: { data: Compare; labels: string[] }) {
  const t = useT();
  const groups = diffGroups(data.differences);
  const [showNoisy, setShowNoisy] = useState(false);
  if (groups.length === 0) return <Card title={t("Различия")}>{<span className="small muted">{t("Snapshot и manifest прогонов совпадают (кроме имени, описания и тегов).")}</span>}</Card>;
  return (
    <Card
      title={t("Различия")}
      aside={
        <span className={r.toolbar}>
          <span className="small muted">{t("Базовый — первый прогон: отличающиеся от него значения выделены.")}</span>
          <CsvButton
            onClick={() => {
              const table = compareDiffTable(data);
              downloadCsv(`${fileName("compare", `${data.runs.length}runs`, "differences")}.csv`, table.header, table.rows);
            }}
          />
        </span>
      }
    >
      <div className={r.scroll}>
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Различия")}>
          <thead>
            <tr>
              <th>{t("Путь")}</th>
              {data.runs.map((run, i) => (
                <th key={run.id}>
                  <span className={c.thRun}>
                    <Swatch slot={i} />
                    {labels[i]}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => {
              const noisy = group.kind === "env" ? group.items.filter((x) => isNoisyEnv(x.path)) : [];
              const items = showNoisy ? group.items : group.items.filter((x) => !noisy.includes(x));
              return (
                <Fragment key={group.kind}>
                  <tr className={c.groupRow}>
                    <td colSpan={data.runs.length + 1}>
                      {t(KIND_TITLE[group.kind])} <span className={r.count}>{group.items.length}</span>
                    </td>
                  </tr>
                  {items.map((item) => (
                    <tr key={item.path}>
                      <td className="mono small">{item.display}</td>
                      {item.values.map((v, i) => (
                        <td key={i} className={cx("mono small", c.value, i > 0 && showValue(v) !== showValue(item.values[0]) && c.changed)} title={showValue(v)}>
                          {showValue(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {noisy.length > 0 && (
                    <tr>
                      <td colSpan={data.runs.length + 1}>
                        <button type="button" className={r.more} onClick={() => setShowNoisy((x) => !x)}>
                          {showNoisy ? t("скрыть points, indexed_vectors и segments") : t("ещё {n}: points, indexed_vectors и segments коллекций", { n: noisy.length })}
                        </button>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <span className={r.caption}>{t("Классы — по секции (ragglet-23): config — настройка RAG, data — датасет, env — окружение. Имя, описание, теги и каталог артефактов не сравниваются.")}</span>
    </Card>
  );
}
