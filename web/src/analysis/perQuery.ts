import type { CacheLayer, Stage } from "./summary";

/**
 * Per-query колонками для графиков задержек (ragglet-32 §3): только нужные поля всех запросов
 * повтора (`fields`), без текстов и списков. Чистые функции над строками — CDF, счётчики стадий,
 * группы линии времени, накопительная доля попаданий кэша.
 */
export type QueryRow = Record<string, unknown>;

export const CHART_FIELDS = [
  "qid",
  "t_start_ms",
  "lat_embed_ms",
  "lat_retrieve_ms",
  "lat_merge_ms",
  "lat_rerank_ms",
  "lat_context_ms",
  "lat_generation_ms",
  "lat_total_ms",
  "outcome",
  "timeout_stage",
  "emb_cache",
  "ret_cache",
];

export interface RepeatRows {
  repeats: number[]; // какие повторы записаны
  columns: string[];
  rows: Map<number, QueryRow[]>;
}

const numeric = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Эмпирическая функция распределения: точки (x, доля запросов не медленнее x), x по возрастанию. */
export function cdf(values: Array<number | null>): Array<[number, number]> {
  const xs = values.filter((v): v is number => v !== null).sort((a, b) => a - b);
  const n = xs.length;
  const out: Array<[number, number]> = [];
  xs.forEach((x, i) => {
    // одинаковые значения — одна точка с верхней долей
    if (i + 1 < n && xs[i + 1] === x) return;
    out.push([x, (i + 1) / n]);
  });
  return out;
}

export const latencyColumn = (stage: Stage | "total") => `lat_${stage}_ms`;

export function column(rows: QueryRow[], name: string): Array<number | null> {
  return rows.map((row) => numeric(row[name]));
}

/** Сколько запросов завершили стадию: непустые lat_<стадия>_ms (docs/metrics.md). */
export function stageCount(rows: QueryRow[], stage: Stage | "total"): number {
  return column(rows, latencyColumn(stage)).filter((v) => v !== null).length;
}

export type ColorBy = "outcome" | CacheLayer;

export interface TimelinePoint {
  t: number; // с от начала повтора
  ms: number;
  qid: string;
  group: string; // исход или hit/miss слоя кэша
}

export function timelinePoints(rows: QueryRow[], colorBy: ColorBy): TimelinePoint[] {
  const key = colorBy === "outcome" ? "outcome" : `${colorBy}_cache`;
  return rows.flatMap((row) => {
    const t = numeric(row.t_start_ms);
    const ms = numeric(row.lat_total_ms);
    if (t === null || ms === null) return [];
    const group = typeof row[key] === "string" && row[key] ? String(row[key]) : colorBy === "outcome" ? "ok" : "none";
    return [{ t: t / 1000, ms, qid: String(row.qid ?? ""), group }];
  });
}

/**
 * Накопительная доля попаданий по ходу повтора: запросы по t_start_ms, после каждого обращения к
 * слою — hits / (hits + misses) с начала повтора. Последняя точка равна <слой>_hit_rate из summary.
 */
export function cumulativeHitRate(rows: QueryRow[], layer: CacheLayer): Array<[number, number]> {
  const key = `${layer}_cache`;
  const ordered = rows
    .map((row) => ({ t: numeric(row.t_start_ms), mark: row[key] }))
    .filter((x): x is { t: number; mark: unknown } => x.t !== null)
    .sort((a, b) => a.t - b.t);
  let hits = 0;
  let total = 0;
  const out: Array<[number, number]> = [];
  for (const { t, mark } of ordered) {
    if (mark !== "hit" && mark !== "miss") continue;
    total += 1;
    if (mark === "hit") hits += 1;
    out.push([t / 1000, hits / total]);
  }
  return out;
}
