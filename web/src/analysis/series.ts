import { METRIC_SLOT } from "../charts/palette";
import type { QualityByKData, QualityMode } from "../charts/options/qualityByK";
import type { StageBarRow } from "../charts/options/stageBar";
import { METRIC_LABEL, latencyGrid, qualityGrid, type Level, type Summary } from "./summary";

/** Ряды графиков из summary: тонкий слой между разбором ключей и построителями опций. */

/** «Качество по k» одного прогона: серия на метрику уровня, цвет закреплён за метрикой. */
export function qualityByMetric(summary: Summary, level: Level, mode: QualityMode): QualityByKData {
  const grid = qualityGrid(summary, level);
  return {
    ks: grid.ks,
    mode,
    series: grid.metrics.map((metric) => ({
      name: METRIC_LABEL[metric],
      slot: METRIC_SLOT[metric],
      points: grid.ks.map((k) => {
        const s = grid.get(metric, k);
        return { k, mean: s?.mean ?? null, std: s?.std ?? null, min: s?.min ?? null, max: s?.max ?? null };
      }),
    })),
  };
}

/** Строка полосы стадий: перцентиль каждой стадии и total, по среднему за повторы. */
export function stageRow(summary: Summary, label: string, percentile: number): StageBarRow {
  const grid = latencyGrid(summary);
  return {
    label,
    values: Object.fromEntries(grid.stages.map((stage) => [stage, grid.get(stage, percentile)?.mean ?? null])),
    total: grid.get("total", percentile)?.mean ?? null,
  };
}
