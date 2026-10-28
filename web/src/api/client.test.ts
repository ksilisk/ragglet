import { ApiProblem, settle } from "./client";

const response = (status: number) => new Response(null, { status, statusText: status === 422 ? "Unprocessable Content" : "OK" });

describe("settle", () => {
  it("2xx — данные и статус", async () => {
    await expect(settle(Promise.resolve({ data: { id: "x" }, response: response(202) }))).resolves.toEqual({ data: { id: "x" }, status: 202 });
  });

  it("не 2xx с Problem — ApiProblem с этим Problem", async () => {
    const problem = { title: "Scenario config is invalid", hint: null, issues: [{ code: "config.value_error", severity: "error" as const, path: "metrics.quality.k", where: null, message: "k" }] };
    const error = await settle(Promise.resolve({ error: problem, response: response(422) })).catch((e) => e);
    expect(error).toBeInstanceOf(ApiProblem);
    expect(error.status).toBe(422);
    expect(error.has("config.value_error")).toBe(true);
    expect(error.issues[0].path).toBe("metrics.quality.k");
  });

  it("тело не Problem (HTTPValidationError, текст) — Problem из статуса", async () => {
    const error = await settle(Promise.resolve({ error: { detail: [] }, response: response(422) })).catch((e) => e);
    expect(error.problem.title).toBe("HTTP 422");
    expect(error.problem.issues).toHaveLength(1);
  });

  it("сеть недоступна — status 0", async () => {
    const error = await settle(Promise.reject(new TypeError("Failed to fetch"))).catch((e) => e);
    expect(error).toBeInstanceOf(ApiProblem);
    expect(error.status).toBe(0);
    expect(error.has("network.error")).toBe(true);
  });
});
