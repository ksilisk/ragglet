import { useMemo, type ReactNode } from "react";
import { Chart } from "../../charts/Chart";
import { qualityByKOption, type QualityMode } from "../../charts/options/qualityByK";
import { stageBarHeight, stageBarOption } from "../../charts/options/stageBar";
import { useT } from "../../i18n/i18n";
import { fmtMs, fmtPercent } from "../../lib/format";
import { useQualityMode } from "../../analysis/qualityMode";
import { qualityByMetric, stageRow } from "../../analysis/series";
import type { SnapshotFacts } from "../../analysis/snapshot";
import { latencyGrid, repeatCount, timeoutStages, totalQueries, type Level, type Summary } from "../../analysis/summary";
import k from "../../ui/kit.module.css";
import { Card, Empty } from "../../ui/Layout";
import { Segmented } from "../../ui/Segmented";
import r from "./results.module.css";

/** Карточки графиков, общие для «Обзора» и вкладок «Качество» и «Задержки». */

export function QualityModeToggle({ mode, onChange }: { mode: QualityMode; onChange: (mode: QualityMode) => void }) {
  const t = useT();
  return (
    <Segmented
      label={t("Вид графика (временно, на время ревью)")}
      value={mode}
      onChange={onChange}
      items={[
        { id: "lines", label: t("линии") },
        { id: "bars", label: t("столбики") },
      ]}
    />
  );
}

/** «Качество по k»: серия на метрику, усы — std по повторам. */
export function QualityCard({ summary, level, title, aside }: { summary: Summary; level: Level; title?: string; aside?: ReactNode }) {
  const t = useT();
  const [mode, setMode] = useQualityMode();
  const data = useMemo(() => qualityByMetric(summary, level, mode), [summary, level, mode]);
  const repeats = repeatCount(summary);
  return (
    <Card
      title={title ?? t("Качество по k")}
      aside={
        <div className={r.toolbar}>
          {aside}
          <QualityModeToggle mode={mode} onChange={setMode} />
        </div>
      }
    >
      {data.series.length === 0 ? (
        <Empty>{t("Метрик уровня {level} в summary нет: прогон старого формата или уровень выключен.", { level })}</Empty>
      ) : (
        <>
          <Chart build={qualityByKOption} data={data} height={280} label={t("Качество по k, уровень {level}", { level })} />
          <span className={r.caption}>
            {repeats > 1
              ? t("Шкала 0–1. Усы — std по {n} повторам.", { n: repeats })
              : t("Шкала 0–1. Один повтор: разброса нет, для усов нужны repeats ≥ 3.")}
          </span>
        </>
      )}
    </Card>
  );
}

/** «Задержки по стадиям»: полоса p50, таблица перцентилей, таймауты по стадиям. */
export function StageCard({ summary, facts }: { summary: Summary; facts: SnapshotFacts }) {
  const t = useT();
  const grid = latencyGrid(summary);
  const p = grid.percentiles.includes(50) ? 50 : (grid.percentiles[0] ?? 50);
  const data = useMemo(() => ({ stages: grid.stages, rows: [stageRow(summary, "", p)], percentile: p }), [summary, grid.stages, p]);
  const timeouts = timeoutStages(summary);
  const n = totalQueries(summary);
  const off = facts.stagesOff.filter((stage) => !grid.stages.includes(stage));

  if (!grid.hasTotal) {
    return (
      <Card title={t("Задержки по стадиям")}>
        <Empty>{t("В summary нет перцентилей задержек.")}</Empty>
      </Card>
    );
  }
  return (
    <Card title={t("Задержки по стадиям, p{p}", { p })}>
      <Chart build={stageBarOption} data={data} height={stageBarHeight(1)} label={t("Задержки по стадиям, p{p}", { p })} />
      {off.length > 0 && <span className={r.caption}>{t("Выключены в сценарии: {stages}.", { stages: off.join(", ") })}</span>}
      <table className={`${k.tbl} ${k.compact}`} aria-label={t("Перцентили по стадиям")}>
        <thead>
          <tr>
            <th>{t("Стадия")}</th>
            {grid.percentiles.map((x) => (
              <th key={x} className="r">
                p{x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...grid.stages, "total" as const].map((stage) => (
            <tr key={stage}>
              <td>{stage === "total" ? <b>total</b> : stage}</td>
              {grid.percentiles.map((x) => {
                const v = fmtMs(grid.get(stage, x)?.mean ?? null);
                return (
                  <td key={x} className="r num">
                    {stage === "total" ? <b>{v}</b> : v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <span className={r.caption}>
        {timeouts.length > 0
          ? t("Таймауты по стадиям: {list}.", {
              list: timeouts.map((x) => `${x.stage} ${x.count}${n ? ` (${fmtPercent(x.count / n)})` : ""}`).join(", "),
            }) + " "
          : ""}
        {t("Стадия, не завершившая запрос, в свои перцентили не входит. Сумма медиан стадий не равна медиане total: стадии fanout идут параллельно, есть накладные расходы.")}
      </span>
    </Card>
  );
}
