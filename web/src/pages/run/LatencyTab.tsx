import { Fragment, useMemo, useState } from "react";
import { ApiProblem, type S } from "../../api/client";
import { useRunColumns, useRunInclude } from "../../api/queries";
import { CHART_FIELDS, cdf, column, cumulativeHitRate, stageCount, timelinePoints, type ColorBy, type RepeatRows } from "../../analysis/perQuery";
import { snapshotFacts, type SnapshotFacts } from "../../analysis/snapshot";
import { CACHE_LAYERS, latencyGrid, num, repeatCount, timeoutStages, type Summary } from "../../analysis/summary";
import { Chart } from "../../charts/Chart";
import { cacheCurveOption } from "../../charts/options/cacheCurve";
import { latencyCdfOption } from "../../charts/options/latencyCdf";
import { timelineOption } from "../../charts/options/timeline";
import { useT } from "../../i18n/i18n";
import { fmtMs, fmtNumber, fmtPercent, fmtQps } from "../../lib/format";
import k from "../../ui/kit.module.css";
import { Card, Empty } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { Segmented } from "../../ui/Segmented";
import r from "./results.module.css";

type Pick = "all" | number;

/**
 * «Задержки и нагрузка» (ragglet-32 §6): перцентили по стадиям, распределение lat_total_ms, линия
 * времени, пропускная способность и исходы по повторам, кэш. Всё, кроме перцентилей и таблиц из
 * summary, строится по per-query.
 */
export function LatencyTab({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const query = useRunInclude(run.id, "summary");
  const snapshot = useRunInclude(run.id, "scenario");
  const summary = query.data?.summary ?? null;
  const facts = useMemo(() => snapshotFacts(snapshot.data?.scenario_yaml), [snapshot.data?.scenario_yaml]);
  const columns = useRunColumns(run.id, CHART_FIELDS, Boolean(summary));
  const [pick, setPick] = useState<Pick>("all");
  if (query.error) return <ErrorBox error={query.error} />;
  if (query.isPending) return <span className="muted">{t("Загрузка summary…")}</span>;
  if (!summary) return <Empty>{t("У прогона нет summary: ядро пишет его в конце успешного прогона.")}</Empty>;

  const repeats = repeatCount(summary);
  const rows = columns.data ?? null;
  const noPerQuery = columns.error instanceof ApiProblem && columns.error.has("run.no_per_query");
  const perQuery = { rows, loading: columns.isPending, error: noPerQuery ? null : columns.error, none: noPerQuery };

  return (
    <>
      {repeats > 1 && (
        <div className={r.toolbar}>
          <span className="small muted">{t("Повтор")}</span>
          <Segmented<string>
            label={t("Повтор")}
            value={String(pick)}
            onChange={(v) => setPick(v === "all" ? "all" : Number(v))}
            items={[{ id: "all", label: t("все") }, ...summary.repeats.map((_, i) => ({ id: String(i), label: String(i) }))]}
          />
        </div>
      )}
      <PercentileCard summary={summary} pick={pick} rows={rows} />
      <div className={r.grid2}>
        <CdfCard pick={pick} perQuery={perQuery} budget={facts.totalBudgetMs} />
        <TimelineCard pick={pick} perQuery={perQuery} facts={facts} />
      </div>
      <ThroughputCard summary={summary} facts={facts} concurrency={typeof run.plan?.load?.concurrency === "number" ? run.plan.load.concurrency : facts.load.concurrency} />
      {facts.cache.enabled && <CacheCard summary={summary} pick={pick} perQuery={perQuery} facts={facts} />}
    </>
  );
}

interface PerQuery {
  rows: RepeatRows | null;
  loading: boolean;
  error: unknown;
  none: boolean; // save_per_query: false
}

const picked = (rows: RepeatRows, pick: Pick) => (pick === "all" ? rows.repeats : rows.repeats.filter((x) => x === pick));

/** Подпись вместо графика, когда per-query нет, не загрузились или ещё грузятся. */
function PerQueryState({ perQuery }: { perQuery: PerQuery }) {
  const t = useT();
  if (perQuery.none) return <Empty>{t("per-query не сохранялись (run.save_per_query: false): доступны только итоги summary.")}</Empty>;
  if (perQuery.error) return <ErrorBox error={perQuery.error} />;
  return <span className="small muted">{t("Загрузка per-query…")}</span>;
}

function PercentileCard({ summary, pick, rows }: { summary: Summary; pick: Pick; rows: RepeatRows | null }) {
  const t = useT();
  const grid = latencyGrid(summary);
  const many = repeatCount(summary) > 1;
  if (!grid.hasTotal) return <Card title={t("Перцентили по стадиям")}><Empty>{t("В summary нет перцентилей задержек.")}</Empty></Card>;
  const count = (stage: (typeof grid.stages)[number] | "total") =>
    rows ? picked(rows, pick).reduce((acc, rep) => acc + stageCount(rows.rows.get(rep) ?? [], stage), 0) : null;
  return (
    <Card title={t("Перцентили по стадиям, мс")}>
      <table className={`${k.tbl} ${k.compact}`} aria-label={t("Перцентили по стадиям")}>
        <thead>
          <tr>
            <th>{t("Стадия")}</th>
            {grid.percentiles.map((p) => (
              <th key={p} className="r">
                p{p}
              </th>
            ))}
            <th className="r" title={t("сколько запросов завершили стадию")}>
              n
            </th>
          </tr>
        </thead>
        <tbody>
          {[...grid.stages, "total" as const].map((stage) => (
            <tr key={stage}>
              <td>{stage === "total" ? <b>total</b> : stage}</td>
              {grid.percentiles.map((p) => {
                const s = grid.get(stage, p);
                const value = pick === "all" ? (s?.mean ?? null) : (s?.perRepeat[pick] ?? null);
                return (
                  <td key={p} className="r num">
                    {fmtMs(value)}
                    {pick === "all" && many && s?.std !== null && s?.std !== undefined && <span className={r.std}> ± {fmtMs(s.std)}</span>}
                  </td>
                );
              })}
              <td className="r num muted">{fmtNumber(count(stage))}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <span className={r.caption}>
        {pick === "all" && many ? t("Среднее ± std перцентиля по повторам.") : t("Значения повтора.")}{" "}
        {t("Задержка стадии — только по запросам, где стадия завершилась; total — по всем, с таймаутами.")}
      </span>
    </Card>
  );
}

function CdfCard({ pick, perQuery, budget }: { pick: Pick; perQuery: PerQuery; budget: number | null }) {
  const t = useT();
  const [scale, setScale] = useState<"linear" | "log">("log");
  const rows = perQuery.rows;
  const data = useMemo(
    () =>
      rows
        ? {
            log: scale === "log",
            budgetMs: budget,
            series: picked(rows, pick).map((rep) => ({ name: t("повтор {index}", { index: rep }), slot: rep, points: cdf(column(rows.rows.get(rep) ?? [], "lat_total_ms")) })),
          }
        : null,
    [rows, pick, scale, budget, t],
  );
  return (
    <Card
      title={t("Распределение lat_total_ms")}
      aside={
        <Segmented
          label={t("Шкала")}
          value={scale}
          onChange={setScale}
          items={[
            { id: "linear", label: t("линейная") },
            { id: "log", label: t("лог") },
          ]}
        />
      }
    >
      {data ? <Chart build={latencyCdfOption} data={data} height={300} label={t("Распределение lat_total_ms")} /> : <PerQueryState perQuery={perQuery} />}
      <span className={r.caption}>
        {t("Доля запросов, ответивших не медленнее x. Ступенька у пунктира — запросы, упёршиеся в бюджет timeouts.total_ms.")}
        {budget !== null ? " " + t("Бюджет прогона — {ms} мс.", { ms: fmtMs(budget) }) : ""}
      </span>
    </Card>
  );
}

const OUTCOME_GROUPS = ["ok", "timeout", "error"];
const CACHE_GROUPS = ["hit", "miss", "none"];

function TimelineCard({ pick, perQuery, facts }: { pick: Pick; perQuery: PerQuery; facts: SnapshotFacts }) {
  const t = useT();
  const [colorBy, setColorBy] = useState<ColorBy>("outcome");
  const rows = perQuery.rows;
  const repeat = pick === "all" ? (rows?.repeats[0] ?? 0) : pick;
  const layers = CACHE_LAYERS.filter((layer) => facts.cache[layer]);
  const data = useMemo(() => {
    if (!rows) return null;
    const points = timelinePoints(rows.rows.get(repeat) ?? [], colorBy);
    const order = colorBy === "outcome" ? OUTCOME_GROUPS : CACHE_GROUPS;
    const seen = new Set(points.map((p) => p.group));
    return { points, groups: [...order.filter((g) => seen.has(g)), ...[...seen].filter((g) => !order.includes(g))], budgetMs: facts.totalBudgetMs };
  }, [rows, repeat, colorBy, facts.totalBudgetMs]);
  return (
    <Card
      title={t("Линия времени, повтор {index}", { index: repeat })}
      aside={
        layers.length > 0 && (
          <Segmented<ColorBy>
            label={t("Раскраска")}
            value={colorBy}
            onChange={setColorBy}
            items={[{ id: "outcome", label: t("исход") }, ...layers.map((layer) => ({ id: layer, label: t("кэш {layer}", { layer }) }))]}
          />
        )
      }
    >
      {!data ? (
        <PerQueryState perQuery={perQuery} />
      ) : data.points.length === 0 ? (
        <Empty>{t("В per-query нет t_start_ms: прогон старого формата (до ragglet-19).")}</Empty>
      ) : (
        <Chart build={timelineOption} data={data} height={300} label={t("Линия времени повтора")} />
      )}
      <span className={r.caption}>
        {t("Точка — запрос: старт от начала повтора и общая задержка. Колесо мыши и ползунок под осью приближают участок.")}
        {pick === "all" && rows && rows.repeats.length > 1 ? " " + t("Показан повтор 0; другой — переключателем повторов вверху.") : ""}
      </span>
    </Card>
  );
}

function ThroughputCard({ summary, facts, concurrency }: { summary: Summary; facts: SnapshotFacts; concurrency: number | null | undefined }) {
  const t = useT();
  const stages = timeoutStages(summary);
  const share = (count: number | null, n: number | null) => (count === null ? "—" : `${fmtNumber(count)}${n ? ` (${fmtPercent(count / n)})` : ""}`);
  return (
    <Card title={t("Пропускная способность и исходы по повторам")}>
      <table className={`${k.tbl} ${k.compact}`} aria-label={t("Пропускная способность и исходы по повторам")}>
        <thead>
          <tr>
            <th>#</th>
            <th className="r">n</th>
            <th className="r">{t("wall, с")}</th>
            <th className="r">QPS</th>
            <th className="r">ok</th>
            <th className="r">timeout</th>
            <th className="r">error</th>
            <th>{t("Таймауты по стадиям")}</th>
          </tr>
        </thead>
        <tbody>
          {summary.repeats.map((row, i) => {
            const n = num(row, "n");
            const list = stages.filter((s) => s.perRepeat[i]).map((s) => `${s.stage} ${fmtNumber(s.perRepeat[i])}`);
            return (
              <tr key={i}>
                <td className="num">{String(row.repeat_idx ?? i)}</td>
                <td className="r num">{fmtNumber(n)}</td>
                <td className="r num">{fmtNumber((num(row, "wall_ms") ?? Number.NaN) / 1000, 1)}</td>
                <td className="r num">{fmtQps(num(row, "throughput_qps"))}</td>
                <td className="r num">{share(num(row, "ok_count"), n)}</td>
                <td className="r num">{share(num(row, "timeout_count"), n)}</td>
                <td className="r num">{share(num(row, "error_count"), n)}</td>
                <td className="mono small">{list.length ? list.join(", ") : <span className="muted">—</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <span className={r.caption}>
        {t("Нагрузка: {type}, concurrency {concurrency}.", { type: facts.load.type ?? "—", concurrency: concurrency ?? "—" })}{" "}
        {t("QPS — только ответы ok в секунду времени повтора: таймаут и ошибка пользователю ничего не дали.")}
      </span>
    </Card>
  );
}

const LAYER_SLOT = { emb: 0, ret: 2 } as const;

function CacheCard({ summary, pick, perQuery, facts }: { summary: Summary; pick: Pick; perQuery: PerQuery; facts: SnapshotFacts }) {
  const t = useT();
  const layers = useMemo(() => CACHE_LAYERS.filter((layer) => facts.cache[layer]), [facts.cache]);
  const rows = perQuery.rows;
  const repeat = pick === "all" ? (rows?.repeats[0] ?? 0) : pick;
  const data = useMemo(
    () =>
      rows
        ? { series: layers.map((layer) => ({ name: layer, slot: LAYER_SLOT[layer], points: cumulativeHitRate(rows.rows.get(repeat) ?? [], layer) })).filter((s) => s.points.length > 0) }
        : null,
    [rows, repeat, layers],
  );
  if (layers.length === 0) return null;
  return (
    <div className={r.grid2}>
      <Card title={t("Кэш по повторам")}>
        <table className={`${k.tbl} ${k.compact}`} aria-label={t("Кэш по повторам")}>
          <thead>
            <tr>
              <th>#</th>
              <th>{t("Слой")}</th>
              <th className="r">hits</th>
              <th className="r">misses</th>
              <th className="r">hit rate</th>
              <th className="r">{t("p50 попаданий")}</th>
              <th className="r">{t("p50 промахов")}</th>
            </tr>
          </thead>
          <tbody>
            {summary.repeats.map((row, i) => (
              <Fragment key={i}>
                {layers.map((layer, j) => (
                  <tr key={layer}>
                    <td className="num">{j === 0 ? String(row.repeat_idx ?? i) : ""}</td>
                    <td className="mono">{layer}</td>
                    <td className="r num">{fmtNumber(num(row, `${layer}_hits`))}</td>
                    <td className="r num">{fmtNumber(num(row, `${layer}_misses`))}</td>
                    <td className="r num">{fmtPercent(num(row, `${layer}_hit_rate`))}</td>
                    <td className="r num">{fmtMs(num(row, `lat_total_${layer}_hit_p50`))}</td>
                    <td className="r num">{fmtMs(num(row, `lat_total_${layer}_miss_p50`))}</td>
                  </tr>
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
        <span className={r.caption}>
          {t("Доля накопительная: hits / (hits + misses) за весь повтор, по счётчикам.")}{" "}
          {facts.cache.ret ? t("Кэш retrieval на уникальных запросах не попадает: LRU очищается перед каждым повтором.") : ""}
        </span>
      </Card>
      <Card title={t("Накопительная доля попаданий, повтор {index}", { index: repeat })}>
        {data ? (
          data.series.length > 0 ? (
            <Chart build={cacheCurveOption} data={data} height={240} label={t("Накопительная доля попаданий кэша")} />
          ) : (
            <Empty>{t("В per-query нет отметок кэша: к кэшу не обращались.")}</Empty>
          )
        ) : (
          <PerQueryState perQuery={perQuery} />
        )}
        <span className={r.caption}>{t("После каждого обращения к слою — доля попаданий с начала повтора; последняя точка равна hit rate повтора. На ней виден прогрев.")}</span>
      </Card>
    </div>
  );
}
