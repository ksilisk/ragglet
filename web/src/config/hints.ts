import { msg, type Vars } from "../i18n/i18n";
import { getAt } from "./paths";

/**
 * Подсказка формы: только показывается, ничего не блокирует; валидность решает сервер. message —
 * шаблон msg("… {depth} …"), числа — в vars; при выводе t(hint.message, hint.vars).
 */
export interface Hint {
  path: string;
  severity: "warning" | "info";
  message: string;
  vars?: Vars;
}

type Data = Record<string, unknown> | null | undefined;

const num = (data: Data, path: string, fallback: number): number => {
  const v = getAt(data, path.split("."));
  return typeof v === "number" ? v : fallback;
};
const bool = (data: Data, path: string, fallback = false): boolean => {
  const v = getAt(data, path.split("."));
  return typeof v === "boolean" ? v : fallback;
};

export interface FinalDepth {
  depth: number;
  source: string;
  parts: Array<[string, number]>;
}

/** Глубина финального списка — так же, как её считает валидатор схемы. */
export function finalDepth(data: Data, defaults: Data): FinalDepth {
  const topK = num(data, "retrieval.top_k", num(defaults, "retrieval.top_k", 30));
  const parts: Array<[string, number]> = [["retrieval.top_k", topK]];
  let depth = topK;
  let source = "retrieval.top_k";
  if (bool(data, "rerank.enabled")) {
    for (const name of ["input_top_k", "output_top_n"] as const) {
      const value = num(data, `rerank.${name}`, num(defaults, `rerank.${name}`, name === "input_top_k" ? 30 : 10));
      parts.push([name, value]);
      if (value < depth) {
        depth = value;
        source = `rerank.${name}`;
      }
    }
  }
  return { depth, source, parts };
}

export interface CollectionMeta {
  dataset_source?: unknown;
  dataset_id?: unknown;
  embedding_model?: unknown;
  embedding_normalize?: unknown;
}

export function scenarioHints(data: Data, defaults: Data, collection?: CollectionMeta | null): Hint[] {
  if (!data) return [];
  const hints: Hint[] = [];

  const { depth, source, parts } = finalDepth(data, defaults);
  const ks = getAt(data, ["metrics", "quality", "k"]) ?? getAt(defaults, ["metrics", "quality", "k"]);
  const formula = parts.length > 1 ? ` = min(${parts.map(([n, v]) => `${n} ${v}`).join(", ")})` : "";
  const depthHint = { severity: "info", message: msg("Глубина финального списка: {depth}{formula}."), vars: { depth, formula } } as const;
  hints.push({ path: "metrics.quality.k", ...depthHint });
  if (Array.isArray(ks)) {
    const over = ks.filter((k): k is number => typeof k === "number" && k > depth);
    if (over.length) {
      hints.push({
        path: "metrics.quality.k",
        severity: "warning",
        message: msg("k = {ks} больше глубины финального списка {depth} ({source}): проверка не пройдёт."),
        vars: { ks: over.join(", "), depth, source },
      });
    }
  }

  const rerank = bool(data, "rerank.enabled");
  const topK = num(data, "retrieval.top_k", num(defaults, "retrieval.top_k", 30));
  if (rerank) {
    const inputTopK = num(data, "rerank.input_top_k", num(defaults, "rerank.input_top_k", 30));
    hints.push({ path: "rerank.output_top_n", ...depthHint });
    if (inputTopK > topK) {
      hints.push({
        path: "rerank.input_top_k",
        severity: "warning",
        message: msg("input_top_k больше retrieval.top_k ({topK}): реранк получит не больше {topK} кандидатов."),
        vars: { topK },
      });
    }
  }

  const total = num(data, "timeouts.total_ms", num(defaults, "timeouts.total_ms", 6000));
  const budgets: Array<[string, number]> = [["retrieval_ms", num(data, "timeouts.retrieval_ms", num(defaults, "timeouts.retrieval_ms", 1500))]];
  if (rerank) budgets.push(["rerank_ms", num(data, "timeouts.rerank_ms", num(defaults, "timeouts.rerank_ms", 800))]);
  if (bool(data, "generation.enabled")) budgets.push(["llm_ms", num(data, "timeouts.llm_ms", num(defaults, "timeouts.llm_ms", 3500))]);
  const sum = budgets.reduce((acc, [, v]) => acc + v, 0);
  hints.push({
    path: "timeouts.total_ms",
    severity: sum > total ? "warning" : "info",
    message:
      sum > total
        ? msg("Сумма бюджетов стадий {sum} мс ({budgets}) больше total_ms {total}: запрос упрётся в total раньше, чем в стадию.")
        : msg("Сумма бюджетов включённых стадий: {sum} мс из {total}."),
    vars: { sum, total, budgets: budgets.map(([n, v]) => `${n} ${v}`).join(" + ") },
  });

  const strategy = getAt(data, ["retrieval", "strategy"]) ?? "single";
  const sources = getAt(data, ["retrieval", "sources"]);
  const list = Array.isArray(sources) ? (sources as Array<Record<string, unknown>>) : [];
  const keyword = list.filter((s) => s?.kind === "keyword").length;
  if (strategy === "single" && list.length > 1) {
    hints.push({
      path: "retrieval.sources",
      severity: "warning",
      message: msg("single берёт только первый источник: остальные {rest} не используются."),
      vars: { rest: list.length - 1 },
    });
  }
  if (strategy === "single" && keyword > 0) {
    hints.push({ path: "retrieval.sources", severity: "warning", message: msg("keyword-источник при single не реализован: нужен fanout_merge.") });
  }
  if (keyword > 1) {
    hints.push({ path: "retrieval.sources", severity: "warning", message: msg("Больше одного keyword-источника не реализовано.") });
  }

  if (bool(data, "generation.enabled") && !bool(data, "context.enabled")) {
    hints.push({ path: "generation.enabled", severity: "warning", message: msg("Generation требует context.enabled: включите Context.") });
  }

  if (collection) {
    const src = getAt(data, ["dataset", "source"]) ?? "local";
    const id = getAt(data, ["dataset", "id"]);
    if (collection.dataset_id && id && (collection.dataset_id !== id || collection.dataset_source !== src)) {
      hints.push({
        path: "dataset.id",
        severity: "warning",
        message: msg("Коллекция собрана по {collection}, а сценарий — {scenario}."),
        vars: { collection: `${String(collection.dataset_source)}:${String(collection.dataset_id)}`, scenario: `${String(src)}:${String(id)}` },
      });
    }
    const model = getAt(data, ["embedding", "model"]) ?? getAt(defaults, ["embedding", "model"]);
    if (collection.embedding_model && model !== collection.embedding_model) {
      hints.push({
        path: "embedding.model",
        severity: "warning",
        message: msg("Модель коллекции — {model}: preflight не пропустит другую."),
        vars: { model: String(collection.embedding_model) },
      });
    }
  }
  return hints;
}

export function profileHints(data: Data): Hint[] {
  if (!data) return [];
  const method = getAt(data, ["chunking", "method"]) ?? "recursive";
  const size = getAt(data, ["chunking", "chunk_size"]);
  const overlap = getAt(data, ["chunking", "chunk_overlap"]);
  if (method === "recursive" && typeof size === "number" && typeof overlap === "number" && overlap >= size) {
    return [
      {
        path: "chunking.chunk_overlap",
        severity: "warning",
        message: msg("chunk_overlap ({overlap}) должен быть меньше chunk_size ({size})."),
        vars: { overlap, size },
      },
    ];
  }
  return [];
}
