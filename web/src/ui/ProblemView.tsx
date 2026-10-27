import type { ReactNode } from "react";
import { ApiProblem, type Problem, type ProblemIssue } from "../api/client";
import { cx } from "./cx";
import s from "./kit.module.css";

export function IssueList({ issues, onPick }: { issues: ProblemIssue[]; onPick?: (issue: ProblemIssue) => void }) {
  return (
    <ul className={s.issues}>
      {issues.map((issue, i) => (
        <li key={i} className={s.issue}>
          <span className={issue.severity === "error" ? s.x : s.w} aria-label={issue.severity === "error" ? "ошибка" : "предупреждение"}>
            {issue.severity === "error" ? "✕" : "!"}
          </span>
          <span>
            {issue.message}{" "}
            <span className={s.issueMeta}>
              {onPick && issue.path ? (
                <button type="button" className="mono" style={{ all: "unset", cursor: "pointer", color: "var(--accent)" }} onClick={() => onPick(issue)}>
                  {issue.path}
                </button>
              ) : (
                issue.path
              )}
              {issue.path && (issue.where || issue.code) ? " · " : ""}
              {issue.where ? `${issue.where} · ` : ""}
              {issue.code}
            </span>
          </span>
        </li>
      ))}
    </ul>
  );
}

export function ProblemBox({ problem, warning, children }: { problem: Problem; warning?: boolean; children?: ReactNode }) {
  return (
    <div className={cx(s.problem, warning && s.warnBox)} role={warning ? "status" : "alert"}>
      <div className={s.problemTitle}>{problem.title}</div>
      {problem.hint && <div>{problem.hint}</div>}
      <IssueList issues={problem.issues} />
      {children}
    </div>
  );
}

/** Ошибка запроса: ApiProblem — как Problem, остальное — текстом. */
export function ErrorBox({ error }: { error: unknown }) {
  if (error instanceof ApiProblem) return <ProblemBox problem={error.problem} />;
  return <ProblemBox problem={{ title: "Ошибка", hint: null, issues: [{ code: "client.error", severity: "error", path: null, where: null, message: String(error) }] }} />;
}
