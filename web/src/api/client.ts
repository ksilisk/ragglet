import createClient from "openapi-fetch";
import { translate } from "../i18n/i18n";
import type { components, paths } from "./schema";

export type S = components["schemas"];
export type Problem = S["Problem"];
export type ProblemIssue = S["ProblemIssue"];

// fetch берётся в момент запроса, а не при создании клиента: так его можно подменить в тестах.
export const api = createClient<paths>({
  baseUrl: typeof window !== "undefined" ? window.location.origin : "",
  fetch: (request) => globalThis.fetch(request),
});

/** Любой ответ не 2xx: тело всегда Problem (ragglet-28 §3); status 0 — API недоступен. */
export class ApiProblem extends Error {
  readonly status: number;
  readonly problem: Problem;

  constructor(status: number, problem: Problem) {
    super(problem.title);
    this.name = "ApiProblem";
    this.status = status;
    this.problem = problem;
  }

  get issues(): ProblemIssue[] {
    return this.problem.issues;
  }

  has(code: string): boolean {
    return this.problem.issues.some((issue) => issue.code === code);
  }
}

export function isProblem(body: unknown): body is Problem {
  return typeof body === "object" && body !== null && "title" in body && Array.isArray((body as Problem).issues);
}

export function problemOf(title: string, message: string, code = "http.error"): Problem {
  return { title, hint: null, issues: [{ code, severity: "error", path: null, where: null, message }] };
}

interface FetchResult {
  data?: unknown;
  error?: unknown;
  response: Response;
}

/**
 * Ответ openapi-fetch → данные или ApiProblem. Тело ошибки разбирается как Problem всегда, а не по
 * схеме OpenAPI: у части ручек 422 объявлен как HTTPValidationError, хотя сервер отдаёт Problem.
 */
export async function settle<T>(request: Promise<FetchResult>): Promise<{ data: T; status: number }> {
  let result: FetchResult;
  try {
    result = await request;
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    throw new ApiProblem(0, problemOf(translate("API недоступен"), translate("Запущен ли ragglet serve? {reason}", { reason }), "network.error"));
  }
  const { response } = result;
  if (!response.ok) {
    const body = result.error;
    throw new ApiProblem(
      response.status,
      isProblem(body) ? body : problemOf(`HTTP ${response.status}`, typeof body === "string" && body ? body : response.statusText),
    );
  }
  return { data: result.data as T, status: response.status };
}

export async function unwrap<T>(request: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  return (await settle<T>(request)).data;
}
