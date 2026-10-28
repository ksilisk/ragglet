import type { S } from "../api/client";
import type { QualityByKData, QualityMode } from "../charts/options/qualityByK";
import type { StageBarRow } from "../charts/options/stageBar";
import { msg } from "../i18n/i18n";
import { METRICS, METRIC_LABEL, latencyGrid, qualityGrid, stat, type Metric, type Stage, type Stat, type Summary } from "./summary";

/**
 * Сравнение прогонов (ragglet-32 §9): метрики бок о бок с Δ к базовому, ряды общих графиков и группы
 * различий. Вердикт сопоставимости считает сервер (bench/compare.py), здесь он только читается.
 */
export type Compare = S["RunCompare"];
export type CompareRun = S["CompareRun"];

export const MAX_COMPARE = 6;

/** Общие k: метрика бок о бок осмысленна только при k, которое есть у всех прогонов. */
export function commonK(runs: CompareRun[]): number[] {
  const sets = runs.map((run) => new Set(run.summary ? qualityGrid(run.summary, "doc").ks : []));
  const first = sets[0] ?? new Set<number>();
  return [...first].filter((k) => sets.every((s) => s.has(k))).sort((a, b) => a - b);
}

export const QUALITY_GROUP = msg("Качество, doc");
export const LATENCY_GROUP = msg("Задержки, мс");
export const LOAD_GROUP = msg("Нагрузка");
export const CACHE_GROUP = msg("Кэш");

export interface MetricRow {
  key: string; // колонка summary: doc_ndcg@10_mean, lat_total_p50, …
  label: string;
  group: string; // msg(): перевод при выводе
  better: "higher" | "lower" | null;
  format: "metric" | "ms" | "qps" | "percent" | "seconds";
  values: Array<Stat | null>;
  stage?: boolean; // строки стадий свёрнуты по умолчанию
}

const has = (summary: Summary | null | undefined, key: string): Stat | null => {
  if (!summary) return null;
  const s = stat(summary, key);
  return s.mean === null ? null : s;
};

export function metricRows(runs: CompareRun[], k: number): MetricRow[] {
  const rows: MetricRow[] = [];
  const add = (key: string, label: string, group: string, better: MetricRow["better"], format: MetricRow["format"], stage = false) => {
    const values = runs.map((run) => has(run.summary, key));
    // доли кэша есть в старых summary и при выключенном кэше (нули): строка из одних нулей ничего не говорит
    const empty = group === CACHE_GROUP ? values.every((v) => !v?.mean) : values.every((v) => v === null);
    if (!empty) rows.push({ key, label, group, better, format, values, stage });
  };
  for (const metric of METRICS) add(`doc_${metric}@${k}_mean`, `${METRIC_LABEL[metric]}@${k}`, QUALITY_GROUP, "higher", "metric");
  for (const p of [50, 95, 99]) add(`lat_total_p${p}`, `total p${p}`, LATENCY_GROUP, "lower", "ms");
  const stages = new Set<Stage>(runs.flatMap((run) => (run.summary ? latencyGrid(run.summary).stages : [])));
  for (const stage of stages) add(`lat_${stage}_p50`, `${stage} p50`, LATENCY_GROUP, "lower", "ms", true);
  add("throughput_qps", "QPS", LOAD_GROUP, "higher", "qps");
  add("wall_ms", "wall", LOAD_GROUP, "lower", "seconds");
  add("timeout_rate", "timeout rate", LOAD_GROUP, "lower", "percent");
  add("error_rate", "error rate", LOAD_GROUP, "lower", "percent");
  add("emb_hit_rate", "emb hit rate", CACHE_GROUP, "higher", "percent");
  add("ret_hit_rate", "ret hit rate", CACHE_GROUP, "higher", "percent");
  return rows;
}

export interface Delta {
  abs: number;
  rel: number | null; // доля от базового; null — базовый 0
  noise: boolean; // |Δ| не больше суммы std: в пределах разброса повторов
}

export function delta(base: Stat | null, other: Stat | null): Delta | null {
  if (!base || !other || base.mean === null || other.mean === null) return null;
  const abs = other.mean - base.mean;
  const spread = (base.std ?? 0) + (other.std ?? 0);
  return { abs, rel: base.mean !== 0 ? abs / Math.abs(base.mean) : null, noise: spread > 0 && Math.abs(abs) <= spread };
}

/** k и метрики doc-уровня, которые есть хотя бы у одного прогона: оси общего графика. */
export function qualityAxes(runs: CompareRun[]): { ks: number[]; metrics: Metric[] } {
  const grids = runs.flatMap((run) => (run.summary ? [qualityGrid(run.summary, "doc")] : []));
  const ks = [...new Set(grids.flatMap((g) => g.ks))].sort((a, b) => a - b);
  const metrics = METRICS.filter((m) => grids.some((g) => g.metrics.includes(m)));
  return { ks, metrics };
}

/** «Качество по k» для нескольких прогонов: линия на прогон, цвет — позиция прогона в ids. */
export function qualityByRun(runs: CompareRun[], metric: Metric, ks: number[], mode: QualityMode, labels: string[]): QualityByKData {
  return {
    ks,
    mode,
    series: runs.map((run, i) => ({
      name: labels[i] ?? run.id,
      slot: i,
      points: ks.map((k) => {
        const s = has(run.summary, `doc_${metric}@${k}_mean`);
        return { k, mean: s?.mean ?? null, std: s?.std ?? null, min: s?.min ?? null, max: s?.max ?? null };
      }),
    })),
  };
}

/** Полоса стадий на прогон; стадии — объединение по прогонам, в порядке конвейера. */
export function stageRowsByRun(runs: CompareRun[], percentile: number, labels: string[]): { stages: Stage[]; rows: StageBarRow[] } {
  const grids = runs.map((run) => (run.summary ? latencyGrid(run.summary) : null));
  const present = new Set(grids.flatMap((g) => g?.stages ?? []));
  const order: Stage[] = ["embed", "retrieve", "merge", "rerank", "context", "generation"];
  return {
    stages: order.filter((s) => present.has(s)),
    rows: runs.map((run, i) => ({
      label: labels[i] ?? run.id,
      values: Object.fromEntries((grids[i]?.stages ?? []).map((stage) => [stage, grids[i]!.get(stage, percentile)?.mean ?? null])),
      total: grids[i]?.get("total", percentile)?.mean ?? null,
    })),
  };
}

/**
 * Подписи прогонов на графиках: имя сценария, а если оно повторяется — с временем постановки.
 * Время берётся из id (`<name>_<UTC>`), он уникален.
 */
export function runLabels(runs: CompareRun[]): string[] {
  const counts = new Map<string, number>();
  for (const run of runs) counts.set(run.scenario, (counts.get(run.scenario) ?? 0) + 1);
  return runs.map((run) => {
    if ((counts.get(run.scenario) ?? 0) < 2) return run.scenario;
    const stamp = /_(\d{8}T\d{6}Z(?:-\d+)?)$/.exec(run.id)?.[1];
    // «20261024T180917Z-2» → «24.10 18:09-2» (UTC, как в id)
    return stamp ? `${run.scenario} ${stamp.slice(6, 8)}.${stamp.slice(4, 6)} ${stamp.slice(9, 11)}:${stamp.slice(11, 13)}${stamp.includes("-") ? stamp.slice(stamp.indexOf("-")) : ""}` : run.id;
  });
}

export type Kind = "config" | "data" | "env";
export const KIND_ORDER: Kind[] = ["config", "data", "env"];

/** Различия по классам; путь без «scenario.» — секция видна по группе. */
export function diffGroups(differences: Compare["differences"]): Array<{ kind: Kind; items: Array<{ path: string; display: string; values: unknown[] }> }> {
  return KIND_ORDER.map((kind) => ({
    kind,
    items: differences.filter((d) => d.kind === kind).map((d) => ({ path: d.path, display: d.path.replace(/^scenario\./, ""), values: d.values })),
  })).filter((g) => g.items.length > 0);
}

/** Различия env, которые есть всегда и мало что значат: размер и сегменты коллекции. */
const NOISY_ENV = /^env\.services\.storage\.[^.]+\.(points|indexed_vectors|segments)$/;
export const isNoisyEnv = (path: string) => NOISY_ENV.test(path);

/** Значение различия коротко: строки как есть, остальное — компактный JSON. */
export function showValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}
