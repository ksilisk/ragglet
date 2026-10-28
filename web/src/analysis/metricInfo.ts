import { msg } from "../i18n/i18n";
import type { Level, Metric } from "./summary";

/**
 * Определения метрик для подсказок — сокращённо по docs/metrics.md. Строки — msg(), перевод при
 * выводе через t().
 */
export const METRIC_INFO: Record<Metric, { question: string; formula: string }> = {
  hit: { question: msg("Нашли ли хоть один нужный документ среди первых k."), formula: msg("1, если среди первых k есть нужный, иначе 0") },
  recall: { question: msg("Какую долю нужных документов нашли."), formula: msg("(нужных среди первых k) / |R|") },
  r_cap: {
    question: msg("Как recall, но делим не больше чем на k: на датасетах с десятками нужных документов идеальная выдача даёт 1."),
    formula: msg("(нужных среди первых k) / min(k, |R|)"),
  },
  precision: { question: msg("Какая доля выдачи полезна."), formula: msg("(нужных среди первых k) / k") },
  mrr: { question: msg("Как высоко стоит первый нужный документ."), formula: msg("1 / (место первого нужного); 0, если его нет среди первых k") },
  ndcg: {
    question: msg("Насколько хорош порядок с учётом оценок: очки равны оценке и уменьшаются с местом, делятся на очки идеальной выдачи."),
    formula: "DCG@k / IDCG@k",
  },
};

export const COMMON_NOTE = msg("Нужный — оценка в qrels больше 0. Запрос без ответа (таймаут, ошибка) — промах: все метрики 0.");

export const LEVEL_NOTE: Record<Level, string> = {
  doc: msg("Уровень doc: чанки переводятся в документы, документ занимает место лучшего чанка. Сравнимо с BEIR."),
  chunk: msg("Уровень chunk: каждый уникальный чанк — отдельное место, нужность наследуется от документа."),
};

export const SHORT_NOTE = msg("Доля запросов, где doc-метрики @k посчитаны по списку короче k документов (doc_short@k_rate).");
