import type { ProblemIssue } from "../api/client";
import { parsePath } from "../config/paths";
import { rangeOf, type Parsed } from "../config/yamlDoc";
import { translate } from "../i18n/i18n";
import type { Diagnostic } from "./YamlEditor";

const LINE_COL = /line (\d+)(?:, column (\d+))?/;

/** Смещение в тексте по «line 3, column 2» (с единицы), как в where у config.yaml. */
export function offsetOf(text: string, line: number, col = 1): number {
  let offset = 0;
  for (let i = 1; i < line; i += 1) {
    const next = text.indexOf("\n", offset);
    if (next < 0) return text.length;
    offset = next + 1;
  }
  return Math.min(offset + Math.max(0, col - 1), text.length);
}

/** Где в тексте проблема: диапазон узла по path или позиция из where; null — места нет. */
export function locate(issue: Pick<ProblemIssue, "path" | "where">, text: string, parsed: Parsed): [number, number] | null {
  if (issue.path && parsed.doc) {
    const range = rangeOf(parsed.doc, parsePath(issue.path));
    if (range) return range;
  }
  const m = issue.where ? LINE_COL.exec(issue.where) : null;
  if (m) {
    const from = offsetOf(text, Number(m[1]), m[2] ? Number(m[2]) : 1);
    const lineEnd = text.indexOf("\n", from);
    return [from, lineEnd < 0 ? text.length : Math.max(lineEnd, from + 1)];
  }
  return null;
}

export function yamlDiagnostics(text: string, parsed: Parsed, issues: readonly ProblemIssue[]): Diagnostic[] {
  const out: Diagnostic[] = [];
  if (parsed.error) {
    const from = parsed.error.line ? offsetOf(text, parsed.error.line, parsed.error.col ?? 1) : parsed.error.offset;
    out.push({ from, to: Math.min(from + 1, text.length), severity: "error", message: translate("YAML не разбирается: {message}", { message: translate(parsed.error.message) }) });
  }
  for (const issue of issues) {
    const at = locate(issue, text, parsed);
    if (at) out.push({ from: at[0], to: at[1], severity: issue.severity, message: `${issue.message} (${issue.code})` });
  }
  return out;
}
