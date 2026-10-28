import type { ReactNode } from "react";
import { ApiProblem, api, settle, type Problem, type ProblemIssue, type S } from "../../api/client";
import type { CheckStep } from "../../editor/launch";
import { translate, useT } from "../../i18n/i18n";
import { fmtDataset, fmtNumber } from "../../lib/format";
import { IssueList } from "../../ui/ProblemView";

export type DryRun = { ok: true; check: S["RunCheck"] } | { ok: false; problem: Problem };
export type Enqueue = { ok: true; accepted: S["RunAccepted"] } | { ok: false; problem: Problem };

/** dry-run сценария, переданного содержимым (ragglet-28 §8): 200 — отчёт preflight, 422 — проблемы. */
export async function dryRunScenario(scenario: string, scenarioId: string | null): Promise<DryRun> {
  try {
    const { data } = await settle<S["RunCheck"]>(api.POST("/v1/runs", { body: { scenario, scenario_id: scenarioId, dry_run: true, tags: [] } }));
    return { ok: true, check: data };
  } catch (e) {
    if (e instanceof ApiProblem && e.status === 422) return { ok: false, problem: e.problem };
    throw e;
  }
}

/** Постановка в очередь: 202 и id прогона; preflight ещё раз на сервере. */
export async function enqueueScenario(scenario: string, scenarioId: string | null): Promise<Enqueue> {
  try {
    const { data } = await settle<S["RunAccepted"]>(api.POST("/v1/runs", { body: { scenario, scenario_id: scenarioId, dry_run: false, tags: [] } }));
    return { ok: true, accepted: data };
  } catch (e) {
    if (e instanceof ApiProblem && e.status === 422) return { ok: false, problem: e.problem };
    throw e;
  }
}

export function checkStep(result: DryRun): CheckStep<S["RunCheck"]> {
  return result.ok ? { kind: "ok", warnings: result.check.warnings, report: result.check } : { kind: "problems" };
}

/** Короткая строка отчёта preflight: датасет, запросы × повторы, нагрузка. */
export function planLine(check: S["RunCheck"]): string {
  const load = Object.entries(check.plan.load)
    .map(([key, value]) => (key === "type" ? String(value) : `${key} ${String(value)}`))
    .join(" ");
  const plan = translate("{queries} запросов × {repeats} = {total}", { queries: fmtNumber(check.dataset.queries), repeats: check.plan.repeats, total: fmtNumber(check.plan.total) });
  return `${fmtDataset(check.dataset.label)} · ${plan} · ${load}`;
}

/** Тело диалога «Запустить всё равно?» (ragglet-15, пример, шаг 4). */
export function WarningsBody({ warnings, check }: { warnings: ProblemIssue[]; check: S["RunCheck"] }): ReactNode {
  const t = useT();
  return (
    <>
      <span>{t("Dry-run прошёл: {plan}.", { plan: planLine(check) })}</span>
      <span>
        <b>{t("Предупреждения preflight:")}</b>
      </span>
      <IssueList issues={warnings} />
    </>
  );
}
