import type { S } from "../api/client";

export interface ProgressView {
  done: number;
  total: number;
  fraction: number;
  /** Номер повтора с 1 и всего повторов; null — плана нет. */
  repeat: { index: number; of: number } | null;
  elapsedS: number | null;
  /** Осталось ≈, секунд; null — оценивать рано. */
  remainingS: number | null;
  /** Скорость ≈, единиц в секунду (запросов или документов). */
  rate: number | null;
}

/** Меньше этого от первого события — оценка «осталось» ещё шумная. */
export const ETA_MIN_WINDOW_S = 3;

const seconds = (a: string, b: string | number) => (new Date(b).getTime() - new Date(a).getTime()) / 1000;

/**
 * Прогресс задачи для полосы: линейная оценка по средней скорости между первым и последним
 * событием прогресса (ragglet-15, решение 5). Первое событие приходит после первой единицы
 * работы, поэтому скорость считается по done − 1 за окно first_at…updated_at.
 */
export function progressView(
  progress: S["ProgressOut"] | null | undefined,
  plan: Pick<S["RunPlanOut"], "repeats" | "queries_per_repeat"> | null | undefined,
  startedAt: string | null | undefined,
  now: number,
): ProgressView | null {
  if (!progress) return null;
  const { done, total } = progress;
  const fraction = total > 0 ? Math.min(1, done / total) : 0;
  const elapsedS = startedAt ? Math.max(0, seconds(startedAt, now)) : null;

  let repeat: ProgressView["repeat"] = null;
  if (plan && plan.queries_per_repeat > 0 && plan.repeats > 0) {
    const index = done > 0 ? Math.floor((done - 1) / plan.queries_per_repeat) + 1 : 1;
    repeat = { index: Math.min(index, plan.repeats), of: plan.repeats };
  }

  let rate: number | null = null;
  let remainingS: number | null = null;
  const window = seconds(progress.first_at, progress.updated_at);
  if (done >= total) {
    remainingS = 0;
  } else if (done >= 2 && window >= ETA_MIN_WINDOW_S) {
    rate = (done - 1) / window;
    const sinceUpdate = Math.max(0, seconds(progress.updated_at, now));
    remainingS = Math.max(0, (total - done) / rate - sinceUpdate);
  }
  return { done, total, fraction, repeat, elapsedS, remainingS, rate };
}
