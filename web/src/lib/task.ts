import type { S } from "../api/client";
import { msg } from "../i18n/i18n";

export type Phase = "queued" | "preparing" | "progress" | "finishing" | "done";

/** Фаза так же, как её считает API (ragglet-28 §9), — для TaskItem, у которого нет поля phase. */
export function phaseOf(state: S["TaskState"], progress: S["ProgressOut"] | null | undefined): Phase {
  if (state === "queued") return "queued";
  if (state !== "running") return "done";
  if (!progress) return "preparing";
  return progress.done < progress.total ? "progress" : "finishing";
}

/** Подписи фаз, подсказки и единицы — msg(): переводятся при выводе через t(). */
export const PHASE_LABEL: Record<S["TaskKind"], Record<Exclude<Phase, "done">, string>> = {
  run: { queued: msg("В очереди"), preparing: msg("Подготовка"), progress: msg("Запросы"), finishing: msg("Завершение") },
  build: { queued: msg("В очереди"), preparing: msg("Подготовка"), progress: msg("Документы"), finishing: msg("Завершение") },
};

export const PHASE_HINT: Record<S["TaskKind"], Record<"preparing" | "finishing", string>> = {
  run: { preparing: msg("preflight, загрузка моделей, индекс BM25"), finishing: msg("запись summary") },
  build: { preparing: msg("датасет и модель эмбеддингов"), finishing: msg("последний батч и метаданные") },
};

export const UNIT: Record<S["TaskKind"], string> = { run: msg("запросов"), build: msg("документов") };
