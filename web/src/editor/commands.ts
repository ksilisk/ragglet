import { ApiProblem, api, settle, type Problem, type S } from "../api/client";
import { translate } from "../i18n/i18n";
import { writeDocument, type LibraryKind } from "../pages/library/library";
import { checkFromProblem, type CheckResult, type EditorState } from "./store";
import type { StoreApi } from "zustand";

export type SaveOutcome =
  | { kind: "saved"; doc: S["ConfigDocument"] }
  | { kind: "conflict"; problem: Problem }
  | { kind: "rejected"; problem: Problem } // 422: битый YAML, name ≠ id
  | { kind: "blocked" }; // текст не разбирается: на сервер не шлём

/** Сохранить текущий текст: PUT с etag версии, с которой начата правка. */
export async function save(store: StoreApi<EditorState>): Promise<SaveOutcome> {
  const st = store.getState();
  if (st.parsed.error) return { kind: "blocked" };
  const text = st.text;
  try {
    const doc = await writeDocument(st.kind, st.id, text, st.saved.etag);
    store.getState().loadSaved(doc, false);
    return { kind: "saved", doc };
  } catch (e) {
    if (e instanceof ApiProblem && e.status === 409) return { kind: "conflict", problem: e.problem };
    if (e instanceof ApiProblem && e.status === 422) {
      store.getState().setCheck(checkFromProblem("save", e.problem), text);
      return { kind: "rejected", problem: e.problem };
    }
    throw e;
  }
}

export type DryRunOutcome = { kind: "ok"; check: CheckResult } | { kind: "problems"; check: CheckResult } | { kind: "blocked" };

/** dry-run текста как есть: POST /v1/runs или /v1/builds с dry_run: true, без сохранения. */
export async function dryRun(store: StoreApi<EditorState>, kind: LibraryKind = store.getState().kind): Promise<DryRunOutcome> {
  const st = store.getState();
  if (st.parsed.error) return { kind: "blocked" };
  const text = st.text;
  const started = performance.now();
  try {
    const { data } =
      kind === "scenario"
        ? await settle<S["RunCheck"]>(api.POST("/v1/runs", { body: { scenario: text, scenario_id: st.id, dry_run: true, tags: [] } }))
        : await settle<S["BuildCheck"]>(api.POST("/v1/builds", { body: { profile: text, profile_id: st.id, dry_run: true } }));
    const check: CheckResult = {
      source: "check",
      at: Date.now(),
      durationMs: performance.now() - started,
      ok: true,
      title: data.warnings.length ? translate("Проверка пройдена с предупреждениями") : null,
      hint: null,
      issues: data.warnings,
      report: data,
    };
    store.getState().setCheck(check, text);
    return { kind: "ok", check };
  } catch (e) {
    if (e instanceof ApiProblem && e.status === 422) {
      const check = checkFromProblem("check", e.problem, performance.now() - started);
      store.getState().setCheck(check, text);
      return { kind: "problems", check };
    }
    throw e;
  }
}

/** Открыть секции, где после проверки есть ошибки. */
export function openSectionsWithErrors(store: StoreApi<EditorState>, sections: Iterable<string>) {
  for (const id of sections) store.getState().setOpen(id, true);
}
