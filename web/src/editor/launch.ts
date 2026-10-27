import type { Problem, ProblemIssue } from "../api/client";

/**
 * Запуск (ragglet-15, решение 4; ragglet-31 §7): сохранить → dry-run → подтвердить предупреждения →
 * поставить в очередь (202). Шаги внедряются: тот же автомат работает в редакторе (с сохранением),
 * в «Повторить» и «Запустить выбранные» (без него).
 */
export type LaunchResult =
  | { kind: "started"; id: string; position: number | null }
  | { kind: "conflict"; problem: Problem } // 409 при сохранении
  | { kind: "invalid" } // сохранён черновиком или 422 при сохранении: проблемы уже в панели
  | { kind: "problems"; stage: "check" | "submit" } // 422 dry-run или постановки
  | { kind: "cancelled" } // пользователь не стал запускать с предупреждениями
  | { kind: "blocked" }; // текст не разбирается

export type SaveStep = { kind: "saved"; valid: boolean } | { kind: "conflict"; problem: Problem } | { kind: "rejected" } | { kind: "blocked" };
export type CheckStep<R> = { kind: "ok"; warnings: ProblemIssue[]; report: R } | { kind: "problems" } | { kind: "blocked" };
export type SubmitStep = { kind: "accepted"; id: string; position: number | null } | { kind: "problems" };

export interface LaunchSteps<R> {
  save?: () => Promise<SaveStep>;
  check: () => Promise<CheckStep<R>>;
  confirm: (warnings: ProblemIssue[], report: R) => Promise<boolean>;
  submit: () => Promise<SubmitStep>;
}

export async function launch<R>(steps: LaunchSteps<R>): Promise<LaunchResult> {
  if (steps.save) {
    const saved = await steps.save();
    if (saved.kind === "conflict") return { kind: "conflict", problem: saved.problem };
    if (saved.kind === "blocked") return { kind: "blocked" };
    // черновик: схема не прошла, dry-run ответил бы той же 422
    if (saved.kind === "rejected" || !saved.valid) return { kind: "invalid" };
  }
  const checked = await steps.check();
  if (checked.kind === "blocked") return { kind: "blocked" };
  if (checked.kind === "problems") return { kind: "problems", stage: "check" };
  if (checked.warnings.length > 0 && !(await steps.confirm(checked.warnings, checked.report))) return { kind: "cancelled" };
  // постановка делает preflight ещё раз: индекс могли пересобрать между проверкой и запуском
  const submitted = await steps.submit();
  if (submitted.kind === "problems") return { kind: "problems", stage: "submit" };
  return { kind: "started", id: submitted.id, position: submitted.position };
}
