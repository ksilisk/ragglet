import type { S } from "../api/client";

export type Phase = "queued" | "preparing" | "progress" | "finishing" | "done";

/** Фаза так же, как её считает API (ragglet-28 §9), — для TaskItem, у которого нет поля phase. */
export function phaseOf(state: S["TaskState"], progress: S["ProgressOut"] | null | undefined): Phase {
  if (state === "queued") return "queued";
  if (state !== "running") return "done";
  if (!progress) return "preparing";
  return progress.done < progress.total ? "progress" : "finishing";
}

export const PHASE_LABEL: Record<S["TaskKind"], Record<Exclude<Phase, "done">, string>> = {
  run: { queued: "В очереди", preparing: "Подготовка", progress: "Запросы", finishing: "Завершение" },
  build: { queued: "В очереди", preparing: "Подготовка", progress: "Документы", finishing: "Завершение" },
};

export const PHASE_HINT: Record<S["TaskKind"], Record<"preparing" | "finishing", string>> = {
  run: { preparing: "preflight, загрузка моделей, индекс BM25", finishing: "запись summary" },
  build: { preparing: "датасет и модель эмбеддингов", finishing: "последний батч и метаданные" },
};

export const UNIT: Record<S["TaskKind"], string> = { run: "запросов", build: "документов" };
