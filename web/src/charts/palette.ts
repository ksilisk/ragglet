import type { Metric, Outcome, Stage } from "../analysis/summary";
import type { ChartTheme } from "./theme";

/**
 * Закреплённые цвета (ragglet-32 §2): цвет следует за сущностью, а не за местом в списке — скрытая
 * серия или убранный прогон не перекрашивают остальные. Слоты 0–5 = --series-1..6.
 */
export const STAGE_SLOT: Record<Stage, number> = { embed: 0, retrieve: 1, merge: 2, rerank: 3, context: 4, generation: 5 };
export const METRIC_SLOT: Record<Metric, number> = { ndcg: 0, recall: 1, hit: 2, mrr: 3, precision: 4, r_cap: 5 };

/** Второй канал после цвета: маркер и штрих у линий — различимо и в ч/б печати. */
export const SYMBOLS = ["circle", "triangle", "rect", "diamond", "roundRect", "pin"] as const;
export const DASHES: Array<"solid" | number[]> = ["solid", [6, 4], [2, 3], [10, 3, 2, 3], [4, 2], [1, 3]];

/** Текст поверх заливки слота: на светлых слотах (оранжевый, янтарный, розовый) — тёмный. */
const LABEL_ON = ["#ffffff", "#15181d", "#ffffff", "#ffffff", "#15181d", "#15181d"];

export const slotColor = (theme: ChartTheme, slot: number) => theme.series[slot % theme.series.length]!;
export const slotSymbol = (slot: number) => SYMBOLS[slot % SYMBOLS.length]!;
export const slotDash = (slot: number) => DASHES[slot % DASHES.length]!;
export const labelOn = (slot: number) => LABEL_ON[slot % LABEL_ON.length]!;

/** Исходы — цвета состояний, а не серий; форма дублирует цвет. */
export function outcomeStyle(theme: ChartTheme, outcome: Outcome | string): { color: string; symbol: string } {
  if (outcome === "timeout") return { color: theme.warning, symbol: "triangle" };
  if (outcome === "error") return { color: theme.danger, symbol: "diamond" };
  return { color: theme.faint, symbol: "circle" };
}
