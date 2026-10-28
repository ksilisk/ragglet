import type { ProblemIssue } from "../api/client";
import { formatPath, parsePath, patternOf } from "./paths";

/**
 * Раскладка формы: какие пути конфига где показаны. Привязка проблемы (ragglet-31 §5): поле с этим
 * путём → ближайшая группа (укорачиваем путь с конца) → секция по корню → секция по умолчанию.
 */
export interface Registry<Section extends string = string> {
  sections: ReadonlyArray<{ id: Section; title: string }>;
  /** Шаблон пути ("retrieval.sources[].params.ef_search") → секция. Корни секций тоже здесь. */
  anchors: Readonly<Record<string, Section>>;
  /** Куда идут проблемы без path и с неизвестным корнем: «Датасет и индекс» (ragglet-15, решение 3). */
  fallback: Section;
  /** Секция шапки (name, description, tags): без точки в навигации. */
  header?: Section;
}

export interface BoundIssue extends ProblemIssue {
  section: string;
  /** Конкретный путь поля или группы, у которого показать проблему; null — только секция и панель. */
  anchor: string | null;
}

export function bindIssue(issue: ProblemIssue, registry: Registry): BoundIssue {
  if (!issue.path) return { ...issue, section: registry.fallback, anchor: null };
  const segs = parsePath(issue.path);
  for (let len = segs.length; len >= 1; len -= 1) {
    const prefix = segs.slice(0, len);
    const section = registry.anchors[patternOf(prefix)];
    if (section) return { ...issue, section, anchor: formatPath(prefix) };
  }
  return { ...issue, section: registry.fallback, anchor: null };
}

export function bindIssues(issues: readonly ProblemIssue[], registry: Registry): BoundIssue[] {
  return issues.map((issue) => bindIssue(issue, registry));
}

/** Поле правили после проверки: проблема у него была про старое значение. */
export function isStale(anchor: string | null, touched: ReadonlySet<string>): boolean {
  if (anchor === null) return false;
  for (const path of touched) {
    if (anchor === path || anchor.startsWith(`${path}.`) || anchor.startsWith(`${path}[`)) return true;
  }
  return false;
}

export type SectionMark = "error" | "warning" | null;

export function sectionMarks(issues: readonly BoundIssue[]): Map<string, SectionMark> {
  const marks = new Map<string, SectionMark>();
  for (const issue of issues) {
    if (issue.severity === "error") marks.set(issue.section, "error");
    else if (!marks.get(issue.section)) marks.set(issue.section, "warning");
  }
  return marks;
}

/** DOM-id поля по пути: для прокрутки и фокуса из панели «Проверка». */
export function fieldDomId(path: string): string {
  return `f-${path.replace(/[^A-Za-z0-9_-]/g, "_")}`;
}
