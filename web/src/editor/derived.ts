import { useMemo } from "react";
import type { Hint } from "../config/hints";
import { getAt, parsePath } from "../config/paths";
import { bindIssues, isStale, sectionMarks, type BoundIssue } from "../config/registry";
import { useEditor, useEditorContext } from "./store";

/** Проблемы последней проверки, привязанные к полям; у правленых после проверки полей — скрыты. */
export function useVisibleIssues(): BoundIssue[] {
  const { registry } = useEditorContext();
  const check = useEditor((s) => s.check);
  const touched = useEditor((s) => s.touched);
  return useMemo(() => bindIssues(check?.issues ?? [], registry).filter((i) => !isStale(i.anchor, touched)), [check, registry, touched]);
}

export function useIssuesAt(anchor: string): BoundIssue[] {
  const issues = useVisibleIssues();
  return useMemo(() => issues.filter((i) => i.anchor === anchor), [issues, anchor]);
}

export function useSectionMarks() {
  const issues = useVisibleIssues();
  return useMemo(() => sectionMarks(issues), [issues]);
}

export function useHints(): Hint[] {
  const { hints } = useEditorContext();
  const data = useEditor((s) => s.parsed.data);
  return useMemo(() => hints(data), [hints, data]);
}

export function useHintsAt(path: string): Hint[] {
  const hints = useHints();
  return useMemo(() => hints.filter((h) => h.path === path), [hints, path]);
}

/** Значение поля по пути (из разобранного текста) и умолчание схемы для плейсхолдера. */
export function useValue<T = unknown>(path: string): T | undefined {
  const segs = useMemo(() => parsePath(path), [path]);
  return useEditor((s) => getAt(s.parsed.data, segs)) as T | undefined;
}

export function useDefault(path: string): unknown {
  const { defaults } = useEditorContext();
  return getAt(defaults, parsePath(path.replace(/\[\d+\]/g, "")).filter((s) => typeof s === "string"));
}

export function useEditorActions() {
  const { store } = useEditorContext();
  return store.getState();
}
