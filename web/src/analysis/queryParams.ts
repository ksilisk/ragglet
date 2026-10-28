/**
 * Фильтры вкладки «Запросы» в URL (ragglet-32 §7) под теми же именами, что параметры
 * GET /v1/runs/{id}/queries: ссылку на «вот эти таймауты» можно вставить в заметки. Умолчания в URL не
 * пишутся; смена фильтра сбрасывает страницу и открытый запрос.
 */
export interface QueryFilters {
  repeat: number;
  outcome: string | null;
  miss_at: number | null;
  min_total_ms: number | null;
  q: string;
  sort: string | null;
  offset: number;
  k: number | null; // k колонок метрик; null — по умолчанию
  qid: string | null; // открытый в панели запрос
}

export const DEFAULT_FILTERS: QueryFilters = { repeat: 0, outcome: null, miss_at: null, min_total_ms: null, q: "", sort: null, offset: 0, k: null, qid: null };

const KEYS = Object.keys(DEFAULT_FILTERS) as Array<keyof QueryFilters>;
/** Это не фильтры: их смена не сбрасывает страницу и открытый запрос. */
const VIEW_KEYS = new Set<keyof QueryFilters>(["offset", "qid", "k"]);

const int = (v: string | null) => (v !== null && /^\d+$/.test(v) ? Number(v) : null);
const float = (v: string | null) => (v !== null && v !== "" && Number.isFinite(Number(v)) ? Number(v) : null);

export function readFilters(params: URLSearchParams): QueryFilters {
  const outcome = params.get("outcome");
  return {
    repeat: int(params.get("repeat")) ?? 0,
    outcome: outcome === "ok" || outcome === "timeout" || outcome === "error" ? outcome : null,
    miss_at: int(params.get("miss_at")) || null,
    min_total_ms: float(params.get("min_total_ms")),
    q: params.get("q") ?? "",
    sort: params.get("sort") || null,
    offset: int(params.get("offset")) ?? 0,
    k: int(params.get("k")) || null,
    qid: params.get("qid") || null,
  };
}

/** Новые параметры: прочие ключи URL (tab) остаются, умолчания удаляются. */
export function writeFilters(params: URLSearchParams, patch: Partial<QueryFilters>): URLSearchParams {
  const next = new URLSearchParams(params);
  const changesFilter = Object.keys(patch).some((key) => !VIEW_KEYS.has(key as keyof QueryFilters));
  const merged: Partial<QueryFilters> = { ...patch };
  if (changesFilter) {
    if (!("offset" in patch)) merged.offset = 0;
    if (!("qid" in patch)) merged.qid = null;
  }
  for (const key of KEYS) {
    if (!(key in merged)) continue;
    const value = merged[key];
    if (value === null || value === undefined || value === "" || value === DEFAULT_FILTERS[key]) next.delete(key);
    else next.set(key, String(value));
  }
  return next;
}

/** «?tab=queries&…» — ссылка из «Обзора» и «Качества» на вкладку с фильтром. */
export function queriesSearch(patch: Partial<QueryFilters>): string {
  return `?${writeFilters(new URLSearchParams({ tab: "queries" }), patch).toString()}`;
}
