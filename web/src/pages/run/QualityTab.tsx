import { Fragment, useState } from "react";
import { Link } from "react-router";
import type { S } from "../../api/client";
import { useRunInclude } from "../../api/queries";
import { COMMON_NOTE, LEVEL_NOTE, METRIC_INFO, SHORT_NOTE } from "../../analysis/metricInfo";
import { queriesSearch } from "../../analysis/queryParams";
import { METRIC_LABEL, legacyQualityKeys, levels, misses, qualityGrid, repeatCount, stat, type Level, type Stat, type Summary } from "../../analysis/summary";
import { useT } from "../../i18n/i18n";
import { fmtMetric, fmtNumber, fmtPercent } from "../../lib/format";
import k from "../../ui/kit.module.css";
import { Card, Empty } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { Segmented } from "../../ui/Segmented";
import { InfoTip } from "../../ui/Tip";
import { QualityCard } from "./resultCards";
import { downloadCsv, fileName } from "../../analysis/csv";
import { missesTable, qualityTable, type Table } from "../../analysis/tables";
import { CsvButton } from "../../charts/ExportMenu";

const save = (name: string, table: Table) => downloadCsv(`${name}.csv`, table.header, table.rows);
import r from "./results.module.css";

/** «Качество» (ragglet-32 §5): метрика × k по уровням, раскрытие по повторам, промахи со ссылками. */
export function QualityTab({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const query = useRunInclude(run.id, "summary");
  const summary = query.data?.summary ?? null;
  const [levelSel, setLevel] = useState<Level | null>(null);
  if (query.error) return <ErrorBox error={query.error} />;
  if (query.isPending) return <span className="muted">{t("Загрузка summary…")}</span>;
  if (!summary) return <Empty>{t("У прогона нет summary: ядро пишет его в конце успешного прогона.")}</Empty>;

  const available = levels(summary);
  const level = levelSel && available.includes(levelSel) ? levelSel : (available[0] ?? "doc");
  const legacy = legacyQualityKeys(summary);

  return (
    <>
      {available.length > 1 && (
        <div className={r.toolbar}>
          <span className="small muted">{t("Уровень")}</span>
          <Segmented label={t("Уровень метрик")} value={level} onChange={setLevel} items={available.map((x) => ({ id: x, label: x }))} />
        </div>
      )}
      {available.length === 0 ? (
        <Empty>{t("Метрик doc- и chunk-уровня в summary нет: прогон старого формата.")}</Empty>
      ) : (
        <div className={r.grid2}>
          <QualityTable summary={summary} level={level} name={run.id} />
          <QualityCard summary={summary} level={level} name={run.id} title={t("Метрика по k")} />
        </div>
      )}
      {level === "doc" && available.includes("doc") && <MissesCard summary={summary} name={run.id} />}
      {legacy.length > 0 && <LegacyCard summary={summary} keys={legacy} name={run.id} />}
    </>
  );
}

function Cell({ s, many }: { s: Stat | null; many: boolean }) {
  if (!s || s.mean === null) return <td className="r num muted">—</td>;
  return (
    <td className="r num nowrap">
      {fmtMetric(s.mean)}
      {many && s.std !== null && <span className={r.std}> ± {fmtMetric(s.std)}</span>}
    </td>
  );
}

function QualityTable({ summary, level, name }: { summary: Summary; level: Level; name: string }) {
  const t = useT();
  const grid = qualityGrid(summary, level);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const many = repeatCount(summary) > 1;
  const toggle = (metric: string) => setOpen((prev) => {
    const next = new Set(prev);
    if (next.has(metric)) next.delete(metric);
    else next.add(metric);
    return next;
  });
  const short = level === "doc" ? grid.ks.map((x) => stat(summary, `doc_short@${x}_rate`)) : [];

  return (
    <Card
      title={t("Метрика × k")}
      aside={
        <div className={r.toolbar}>
          <InfoTip label={t("Как читать")} content={<>{t(LEVEL_NOTE[level])} {t(COMMON_NOTE)}</>} />
          <CsvButton onClick={() => save(fileName(name, "quality", level), qualityTable(summary, level))} />
        </div>
      }
    >
      <table className={`${k.tbl} ${k.compact}`} aria-label={t("Метрика × k")}>
        <thead>
          <tr>
            <th>{t("Метрика")}</th>
            {grid.ks.map((x) => (
              <th key={x} className="r">
                @{x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.metrics.map((metric) => (
            <Fragment key={metric}>
              <tr>
                <td className="nowrap">
                  {many ? (
                    <button type="button" className={r.expander} aria-expanded={open.has(metric)} onClick={() => toggle(metric)}>
                      <span className={r.caret} aria-hidden>{open.has(metric) ? "▾" : "▸"}</span>
                      {METRIC_LABEL[metric]}
                    </button>
                  ) : (
                    METRIC_LABEL[metric]
                  )}{" "}
                  <InfoTip
                    label={t("Что такое {metric}", { metric: METRIC_LABEL[metric] })}
                    content={
                      <>
                        {t(METRIC_INFO[metric].question)}
                        <br />
                        <span className="mono">{t(METRIC_INFO[metric].formula)}</span>
                      </>
                    }
                  />
                </td>
                {grid.ks.map((x) => (
                  <Cell key={x} s={grid.get(metric, x)} many={many} />
                ))}
              </tr>
              {open.has(metric) &&
                summary.repeats.map((_, i) => (
                  <tr key={i} className={r.subRow}>
                    <td>{t("повтор {index}", { index: i })}</td>
                    {grid.ks.map((x) => (
                      <td key={x} className="r num">
                        {fmtMetric(grid.get(metric, x)?.perRepeat[i] ?? null)}
                      </td>
                    ))}
                  </tr>
                ))}
            </Fragment>
          ))}
          {short.some((s) => s.mean !== null) && (
            <tr>
              <td className="nowrap">
                {t("список короче k")} <InfoTip label={t("Что такое список короче k")} content={t(SHORT_NOTE)} />
              </td>
              {short.map((s, i) => (
                <td key={i} className="r num">
                  {fmtPercent(s.mean)}
                </td>
              ))}
            </tr>
          )}
        </tbody>
      </table>
      <span className={r.caption}>
        {many
          ? t("Ячейка: среднее ± std по {n} повторам; стрелка у метрики раскрывает значения повторов.", { n: repeatCount(summary) })
          : t("Один повтор: разброса нет.")}
      </span>
    </Card>
  );
}

/** Промахи при k по повторам: n − round(n · hit@k); число — ссылка в «Запросы» с фильтром miss_at. */
function MissesCard({ summary, name }: { summary: Summary; name: string }) {
  const t = useT();
  const grid = qualityGrid(summary, "doc");
  if (!grid.metrics.includes("hit")) return null;
  const rows = grid.ks.map((x) => ({ k: x, perRepeat: misses(summary, x) }));
  return (
    <Card
      title={t("Промахи при k")}
      aside={
        <div className={r.toolbar}>
          <InfoTip label={t("Что такое промах")} content={t("Промах при k — среди первых k нет ни одного нужного документа (hit@k = 0). Запрос без ответа — тоже промах.")} />
          <CsvButton onClick={() => save(fileName(name, "misses"), missesTable(summary))} />
        </div>
      }
    >
      <table className={`${k.tbl} ${k.compact}`} aria-label={t("Промахи при k")}>
        <thead>
          <tr>
            <th>k</th>
            {summary.repeats.map((_, i) => (
              <th key={i} className="r">
                {t("повтор {index}", { index: i })}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.k}>
              <td className="num">{row.k}</td>
              {row.perRepeat.map((n, i) => (
                <td key={i} className="r num">
                  {n === null ? "—" : <Link to={{ search: queriesSearch({ repeat: i, miss_at: row.k }) }}>{fmtNumber(n)}</Link>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <span className={r.caption}>{t("Число открывает эти запросы во вкладке «Запросы».")}</span>
    </Card>
  );
}

/** Метрики до ragglet-16/21: ключи как есть, без переименования и графиков. */
function LegacyCard({ summary, keys, name }: { summary: Summary; keys: string[]; name: string }) {
  const t = useT();
  const many = repeatCount(summary) > 1;
  return (
    <Card
      title={t("Метрики старого формата")}
      aside={<CsvButton onClick={() => save(fileName(name, "legacy-metrics"), { header: ["key", "mean", "std"], rows: keys.map((key) => [key, stat(summary, key).mean, stat(summary, key).std]) })} />}
    >
      <span className={r.caption}>{t("Ключи summary как есть: до ragglet-16 recall@k означал нынешний hit@k. Ничего не пересчитывается.")}</span>
      <table className={`${k.tbl} ${k.compact}`} aria-label={t("Метрики старого формата")}>
        <thead>
          <tr>
            <th>{t("Ключ")}</th>
            <th className="r">{t("Значение")}</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((key) => (
            <tr key={key}>
              <td className="mono">{key}</td>
              <Cell s={stat(summary, key)} many={many} />
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
