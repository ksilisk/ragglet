import { createContext, useContext } from "react";
import { createStore, useStore, type StoreApi } from "zustand";
import type { Problem, ProblemIssue, S } from "../api/client";
import { formatPath, type Path } from "../config/paths";
import type { Hint } from "../config/hints";
import type { Registry } from "../config/registry";
import { appendItem, deleteIn, parseYaml, removeItem, setIn, type Parsed } from "../config/yamlDoc";
import type { LibraryKind } from "../pages/library/library";

export type CheckSource = "save" | "check" | "launch" | "task";

/** Последний ответ сервера о тексте: черновик после PUT, dry-run, постановка, упавшая задача. */
export interface CheckResult {
  source: CheckSource;
  at: number;
  durationMs: number | null;
  ok: boolean;
  title: string | null;
  hint: string | null;
  issues: ProblemIssue[];
  report: S["RunCheck"] | S["BuildCheck"] | null;
}

export interface Saved {
  text: string;
  etag: string;
  updatedAt: string;
  valid: boolean;
}

export interface EditorState {
  kind: LibraryKind;
  id: string;
  text: string;
  parsed: Parsed;
  saved: Saved;
  check: CheckResult | null;
  checkedText: string | null;
  /** Пути, изменённые формой после последней проверки: их проблемы устарели. */
  touched: ReadonlySet<string>;
  tab: "form" | "yaml";
  open: Readonly<Record<string, boolean>>;
  /** Выключенные стадии (rerank.enabled: false и т.п.) — для оглавления. */
  off: Readonly<Record<string, boolean>>;
  /** Запрос «прокрутить к полю» из панели «Проверка»; seq — чтобы повторный клик сработал. */
  focus: { path: string | null; line: number | null; seq: number } | null;

  setText(text: string): void;
  edit(path: Path, value: unknown): void;
  remove(path: Path): void;
  append(path: Path, item: unknown): void;
  removeAt(path: Path, index: number): void;
  /** Ответ сервера с документом: обновить сохранённую версию; replace — заменить и текст. */
  loadSaved(doc: S["ConfigDocument"], replace: boolean): void;
  setCheck(check: CheckResult | null, text?: string): void;
  setTab(tab: "form" | "yaml"): void;
  setOpen(section: string, open: boolean): void;
  setOff(section: string, off: boolean): void;
  focusOn(path: string | null, line?: number | null): void;
}

export function checkFromProblem(source: CheckSource, problem: Problem, durationMs: number | null = null): CheckResult {
  return {
    source,
    at: Date.now(),
    durationMs,
    ok: !problem.issues.some((i) => i.severity === "error"),
    title: problem.title,
    hint: problem.hint ?? null,
    issues: problem.issues,
    report: null,
  };
}

function savedOf(doc: S["ConfigDocument"]): Saved {
  return { text: doc.yaml, etag: doc.etag, updatedAt: doc.updated_at, valid: doc.valid };
}

export function createEditorStore(kind: LibraryKind, doc: S["ConfigDocument"], open: Record<string, boolean>): StoreApi<EditorState> {
  return createStore<EditorState>()((set, get) => {
    const write = (path: Path, next: (text: string) => string) => {
      const { text, parsed, touched } = get();
      if (!parsed.doc) return; // битый YAML правится только во вкладке YAML
      const out = next(text);
      if (out === text) return;
      set({ text: out, parsed: parseYaml(out), touched: new Set([...touched, formatPath(path)]) });
    };
    return {
      kind,
      id: doc.id,
      text: doc.yaml,
      parsed: parseYaml(doc.yaml),
      saved: savedOf(doc),
      check: doc.problem ? checkFromProblem("save", doc.problem) : null,
      checkedText: doc.problem ? doc.yaml : null,
      touched: new Set(),
      tab: "form",
      open,
      off: {},
      focus: null,

      setText: (text) => {
        if (text === get().text) return;
        set({ text, parsed: parseYaml(text) });
      },
      edit: (path, value) => write(path, (t) => setIn(t, path, value)),
      remove: (path) => write(path, (t) => deleteIn(t, path)),
      append: (path, item) => write(path, (t) => appendItem(t, path, item)),
      removeAt: (path, index) => write(path, (t) => removeItem(t, path, index)),
      loadSaved: (next, replace) => {
        const patch: Partial<EditorState> = { saved: savedOf(next), id: next.id };
        if (replace) Object.assign(patch, { text: next.yaml, parsed: parseYaml(next.yaml), touched: new Set<string>() });
        if (next.problem) Object.assign(patch, { check: checkFromProblem("save", next.problem), checkedText: next.yaml, touched: new Set<string>() });
        else if (get().check?.source === "save") Object.assign(patch, { check: null, checkedText: null });
        set(patch);
      },
      setCheck: (check, text) => set({ check, checkedText: check ? (text ?? get().text) : null, touched: new Set() }),
      setTab: (tab) => set({ tab }),
      setOpen: (section, isOpen) => set({ open: { ...get().open, [section]: isOpen } }),
      setOff: (section, isOff) => {
        if (Boolean(get().off[section]) !== isOff) set({ off: { ...get().off, [section]: isOff } });
      },
      focusOn: (path, line = null) => set({ focus: { path, line, seq: (get().focus?.seq ?? 0) + 1 } }),
    };
  });
}

export interface EditorContextValue {
  store: StoreApi<EditorState>;
  registry: Registry;
  defaults: Record<string, unknown>;
  /** Клиентские подсказки по разобранному тексту (config/hints.ts). */
  hints: (data: Record<string, unknown> | null) => Hint[];
}

export const EditorContext = createContext<EditorContextValue | null>(null);

export function useEditorContext(): EditorContextValue {
  const ctx = useContext(EditorContext);
  if (!ctx) throw new Error("editor context is missing");
  return ctx;
}

export function useEditor<T>(selector: (state: EditorState) => T): T {
  return useStore(useEditorContext().store, selector);
}

export const isDirty = (s: EditorState) => s.text !== s.saved.text;
