import type { S } from "../api/client";

/**
 * Разбор summary прогона (ragglet-32 §3): summary.json — строка на повтор, summary.aggregate.json —
 * `<колонка>_{mean,std,min,max}_over_repeats`. Одно правило для всех вкладок и сравнения. Ключа нет —
 * null («—» в UI), а не 0: у старых поколений артефактов колонок меньше, ничего не пересчитывается.
 */
export type Summary = S["RunSummaryOut"];
export type Row = Record<string, unknown>;

export const LEVELS = ["doc", "chunk"] as const;
export type Level = (typeof LEVELS)[number];

/** Порядок показа — как в docs/metrics.md. */
export const METRICS = ["ndcg", "recall", "r_cap", "hit", "mrr", "precision"] as const;
export type Metric = (typeof METRICS)[number];
export const METRIC_LABEL: Record<Metric, string> = { ndcg: "nDCG", recall: "Recall", r_cap: "R_cap", hit: "Hit", mrr: "MRR", precision: "Precision" };

/** Стадии в порядке конвейера; total — не стадия, а весь запрос. */
export const STAGES = ["embed", "retrieve", "merge", "rerank", "context", "generation"] as const;
export type Stage = (typeof STAGES)[number];

export const OUTCOMES = ["ok", "timeout", "error"] as const;
export type Outcome = (typeof OUTCOMES)[number];

export const CACHE_LAYERS = ["emb", "ret"] as const;
export type CacheLayer = (typeof CACHE_LAYERS)[number];

const STAT_SUFFIX = /^(.*)_(mean|std|min|max)_over_repeats$/;
const QUALITY_KEY = /^(doc|chunk)_(ndcg|recall|r_cap|hit|mrr|precision)@(\d+)_mean$/;
const LATENCY_KEY = /^lat_(embed|retrieve|merge|rerank|context|generation|total)_p(\d+(?:\.\d+)?)$/;
const LEGACY_QUALITY = /^(?:ndcg|recall|r_cap|hit|mrr|precision)(?:@\d+)?_mean$/;

export function num(row: Row | undefined | null, key: string): number | null {
  const v = row?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export interface Stat {
  mean: number | null;
  std: number | null; // null при одном повторе: разброса нет
  min: number | null;
  max: number | null;
  perRepeat: Array<number | null>;
}

export function repeatCount(summary: Summary): number {
  return summary.repeats.length || num(summary.aggregate, "repeats") || 1;
}

const present = (xs: Array<number | null>) => xs.filter((x): x is number => x !== null);

function meanOf(xs: Array<number | null>): number | null {
  const v = present(xs);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** Выборочное std (ddof=1), как в aggregate_over_repeats. */
function stdOf(xs: Array<number | null>): number | null {
  const v = present(xs);
  if (v.length < 2) return null;
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
}

/** Колонка summary по повторам: агрегат, если он есть, иначе считается из строк повторов. */
export function stat(summary: Summary, key: string): Stat {
  const perRepeat = summary.repeats.map((row) => num(row, key));
  const agg = summary.aggregate;
  const many = repeatCount(summary) > 1;
  const values = present(perRepeat);
  return {
    mean: num(agg, `${key}_mean_over_repeats`) ?? meanOf(perRepeat),
    std: many ? (num(agg, `${key}_std_over_repeats`) ?? stdOf(perRepeat)) : null,
    min: num(agg, `${key}_min_over_repeats`) ?? (values.length ? Math.min(...values) : null),
    max: num(agg, `${key}_max_over_repeats`) ?? (values.length ? Math.max(...values) : null),
    perRepeat,
  };
}

/** Все колонки summary: ключи повторов и агрегата без суффикса _over_repeats. */
export function summaryKeys(summary: Summary): Set<string> {
  const keys = new Set<string>();
  for (const row of summary.repeats) for (const key of Object.keys(row)) keys.add(key);
  for (const key of Object.keys(summary.aggregate)) {
    const m = STAT_SUFFIX.exec(key);
    if (m) keys.add(m[1]!);
  }
  return keys;
}

export interface QualityGrid {
  level: Level;
  metrics: Metric[]; // какие есть, в порядке METRICS
  ks: number[];
  get(metric: Metric, k: number): Stat | null;
}

export const qualityKey = (level: Level, metric: Metric, k: number) => `${level}_${metric}@${k}_mean`;

export function qualityGrid(summary: Summary, level: Level): QualityGrid {
  const metrics = new Set<Metric>();
  const ks = new Set<number>();
  const keys = summaryKeys(summary);
  for (const key of keys) {
    const m = QUALITY_KEY.exec(key);
    if (!m || m[1] !== level) continue;
    metrics.add(m[2] as Metric);
    ks.add(Number(m[3]));
  }
  return {
    level,
    metrics: METRICS.filter((x) => metrics.has(x)),
    ks: [...ks].sort((a, b) => a - b),
    get: (metric, k) => (keys.has(qualityKey(level, metric, k)) ? stat(summary, qualityKey(level, metric, k)) : null),
  };
}

export function levels(summary: Summary): Level[] {
  return LEVELS.filter((level) => qualityGrid(summary, level).metrics.length > 0);
}

/** k, для которых есть хоть одна метрика doc-уровня. */
export function availableK(summary: Summary): number[] {
  return qualityGrid(summary, "doc").ks;
}

/** Метрики качества до ragglet-16/21 (recall@10_mean, mrr_mean): показываются как есть. */
export function legacyQualityKeys(summary: Summary): string[] {
  return [...summaryKeys(summary)].filter((key) => LEGACY_QUALITY.test(key)).sort();
}

export interface LatencyGrid {
  stages: Stage[]; // со значениями, в порядке конвейера
  percentiles: number[];
  hasTotal: boolean;
  get(stage: Stage | "total", p: number): Stat | null;
}

export const latencyKey = (stage: Stage | "total", p: number) => `lat_${stage}_p${p}`;

export function latencyGrid(summary: Summary): LatencyGrid {
  const stages = new Set<string>();
  const ps = new Set<number>();
  const keys = summaryKeys(summary);
  for (const key of keys) {
    const m = LATENCY_KEY.exec(key);
    if (!m) continue;
    // выключенная стадия даёт null во всех повторах: такую не показываем
    if (stat(summary, key).mean === null) continue;
    stages.add(m[1]!);
    ps.add(Number(m[2]));
  }
  return {
    stages: STAGES.filter((x) => stages.has(x)),
    percentiles: [...ps].sort((a, b) => a - b),
    hasTotal: stages.has("total"),
    get: (stage, p) => {
      const key = latencyKey(stage, p);
      return keys.has(key) ? stat(summary, key) : null;
    },
  };
}

/** Исходы за все повторы; null — в этом формате артефактов счётчиков нет (до ragglet-19). */
export function outcomeTotals(summary: Summary): Record<Outcome, number> | null {
  if (!summary.repeats.some((row) => num(row, "ok_count") !== null)) return null;
  const sum = (key: string) => summary.repeats.reduce((acc, row) => acc + (num(row, key) ?? 0), 0);
  return { ok: sum("ok_count"), timeout: sum("timeout_count"), error: sum("error_count") };
}

export interface TimeoutStage {
  stage: string;
  count: number; // за все повторы
  perRepeat: number[];
}

/** timeout_stage_counts — словарь в каждом повторе; в агрегат он не попадает. */
export function timeoutStages(summary: Summary): TimeoutStage[] {
  const out = new Map<string, number[]>();
  summary.repeats.forEach((row, i) => {
    const counts = row.timeout_stage_counts;
    if (!counts || typeof counts !== "object") return;
    for (const [stage, value] of Object.entries(counts as Record<string, unknown>)) {
      if (typeof value !== "number") continue;
      const per = out.get(stage) ?? summary.repeats.map(() => 0);
      per[i] = value;
      out.set(stage, per);
    }
  });
  return [...out.entries()]
    .map(([stage, perRepeat]) => ({ stage, count: perRepeat.reduce((a, b) => a + b, 0), perRepeat }))
    .sort((a, b) => b.count - a.count || a.stage.localeCompare(b.stage));
}

/** Всего запросов за все повторы. */
export function totalQueries(summary: Summary): number | null {
  const ns = summary.repeats.map((row) => num(row, "n"));
  return ns.some((n) => n !== null) ? ns.reduce<number>((a, b) => a + (b ?? 0), 0) : null;
}

/**
 * Промахи при k по повторам: n − round(n · hit@k). Точно, потому что hit у запроса — 0 или 1, а
 * запрос без ответа — промах (docs/metrics.md). Сервер фильтрует miss_at по тому же условию.
 */
export function misses(summary: Summary, k: number, level: Level = "doc"): Array<number | null> {
  return summary.repeats.map((row) => {
    const n = num(row, "n");
    const hit = num(row, qualityKey(level, "hit", k));
    return n === null || hit === null ? null : n - Math.round(n * hit);
  });
}
