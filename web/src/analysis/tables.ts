import type { Compare, MetricRow } from "./compare";
import type { Cell } from "./csv";
import { CACHE_LAYERS, latencyGrid, misses, num, qualityGrid, type Level, type Summary } from "./summary";

/**
 * Таблицы для CSV (ragglet-32 §10): длинный формат — строка на значение, ключи summary как есть,
 * значения повторов отдельными колонками. Так их читает pandas без переименований.
 */
export interface Table {
  header: string[];
  rows: Cell[][];
}

const repeatHeader = (summary: Summary) => summary.repeats.map((_, i) => `repeat_${i}`);
const cellOf = (v: unknown): Cell => (v === null || v === undefined ? null : typeof v === "object" ? JSON.stringify(v) : (v as Cell));

/** Повторы: все колонки summary.json как есть (словари — JSON в ячейке). */
export function repeatsTable(summary: Summary): Table {
  const header = [...new Set(summary.repeats.flatMap((row) => Object.keys(row)))];
  return { header, rows: summary.repeats.map((row) => header.map((key) => cellOf(row[key]))) };
}

export function qualityTable(summary: Summary, level: Level): Table {
  const grid = qualityGrid(summary, level);
  const rows: Cell[][] = [];
  for (const metric of grid.metrics) {
    for (const k of grid.ks) {
      const s = grid.get(metric, k);
      if (s) rows.push([level, metric, k, s.mean, s.std, s.min, s.max, ...s.perRepeat]);
    }
  }
  return { header: ["level", "metric", "k", "mean", "std", "min", "max", ...repeatHeader(summary)], rows };
}

export function percentilesTable(summary: Summary): Table {
  const grid = latencyGrid(summary);
  const rows: Cell[][] = [];
  for (const stage of [...grid.stages, "total" as const]) {
    for (const p of grid.percentiles) {
      const s = grid.get(stage, p);
      if (s) rows.push([stage, p, s.mean, s.std, s.min, s.max, ...s.perRepeat]);
    }
  }
  return { header: ["stage", "percentile", "mean_ms", "std_ms", "min_ms", "max_ms", ...repeatHeader(summary)], rows };
}

export function missesTable(summary: Summary): Table {
  const rows: Cell[][] = [];
  for (const k of qualityGrid(summary, "doc").ks) misses(summary, k).forEach((n, repeat) => rows.push([k, repeat, num(summary.repeats[repeat], "n"), n]));
  return { header: ["k", "repeat", "n", "misses"], rows };
}

export function throughputTable(summary: Summary): Table {
  const keys = ["n", "wall_ms", "throughput_qps", "ok_count", "timeout_count", "error_count", "ok_rate", "timeout_rate", "error_rate"];
  return {
    header: ["repeat", ...keys, "timeout_stage_counts"],
    rows: summary.repeats.map((row, i) => [i, ...keys.map((key) => num(row, key)), cellOf(row.timeout_stage_counts)]),
  };
}

export function cacheTable(summary: Summary, layers: string[]): Table {
  const keys = (layer: string) => [`${layer}_hits`, `${layer}_misses`, `${layer}_hit_rate`, `lat_total_${layer}_hit_p50`, `lat_total_${layer}_miss_p50`];
  const rows: Cell[][] = [];
  summary.repeats.forEach((row, i) => {
    for (const layer of CACHE_LAYERS.filter((x) => layers.includes(x))) rows.push([i, layer, ...keys(layer).map((key) => num(row, key))]);
  });
  return { header: ["repeat", "layer", "hits", "misses", "hit_rate", "lat_total_hit_p50", "lat_total_miss_p50"], rows };
}

/** Метрики бок о бок: строка — метрика, колонки — <id>_mean, <id>_std. */
export function compareMetricsTable(runs: Compare["runs"], rows: MetricRow[]): Table {
  return {
    header: ["metric", ...runs.flatMap((run) => [`${run.id}_mean`, `${run.id}_std`])],
    rows: rows.map((row) => [row.key, ...row.values.flatMap((v) => [v?.mean ?? null, v?.std ?? null])]),
  };
}

export function compareDiffTable(data: Compare): Table {
  return {
    header: ["kind", "path", ...data.runs.map((run) => run.id)],
    rows: data.differences.map((d) => [d.kind, d.path, ...d.values.map(cellOf)]),
  };
}
