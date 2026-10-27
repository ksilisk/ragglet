import type { ProblemIssue } from "../api/client";
import { launch, type LaunchSteps } from "./launch";

const warning: ProblemIssue = { code: "index.not_green", severity: "warning", path: null, where: "target default (nfcorpus_v1)", message: "collection status is yellow" };

function steps(overrides: Partial<LaunchSteps<string>> = {}) {
  const calls: string[] = [];
  const base: LaunchSteps<string> = {
    save: async () => (calls.push("save"), { kind: "saved", valid: true }),
    check: async () => (calls.push("check"), { kind: "ok", warnings: [], report: "r" }),
    confirm: async () => (calls.push("confirm"), true),
    submit: async () => (calls.push("submit"), { kind: "accepted", id: "s_20261127T101540Z", position: 1 }),
  };
  const wrapped = Object.fromEntries(
    Object.entries({ ...base, ...overrides }).map(([k, fn]) => [k, async (...args: unknown[]) => (k in overrides && calls.push(k), (fn as (...a: unknown[]) => unknown)(...args))]),
  ) as unknown as LaunchSteps<string>;
  return { calls, steps: wrapped };
}

describe("запуск: сохранить → dry-run → подтвердить → 202", () => {
  it("без предупреждений — сразу в очередь, без подтверждения", async () => {
    const { calls, steps: s } = steps();
    expect(await launch(s)).toEqual({ kind: "started", id: "s_20261127T101540Z", position: 1 });
    expect(calls).toEqual(["save", "check", "submit"]);
  });

  it("409 при сохранении — стоп, конфликт", async () => {
    const problem = { title: "Scenario was changed", hint: null, issues: [] };
    const { calls, steps: s } = steps({ save: async () => ({ kind: "conflict", problem }) });
    expect(await launch(s)).toEqual({ kind: "conflict", problem });
    expect(calls).toEqual(["save"]);
  });

  it("сохранён черновиком (valid: false) — dry-run не делается", async () => {
    const { calls, steps: s } = steps({ save: async () => ({ kind: "saved", valid: false }) });
    expect(await launch(s)).toEqual({ kind: "invalid" });
    expect(calls).toEqual(["save"]);
  });

  it("422 dry-run — стоп, проблемы", async () => {
    const { calls, steps: s } = steps({ check: async () => ({ kind: "problems" }) });
    expect(await launch(s)).toEqual({ kind: "problems", stage: "check" });
    expect(calls).toEqual(["save", "check"]);
  });

  it("предупреждения — подтверждение; отказ — ничего не ставится", async () => {
    const { calls, steps: s } = steps({
      check: async () => ({ kind: "ok", warnings: [warning], report: "r" }),
      confirm: async () => false,
    });
    expect(await launch(s)).toEqual({ kind: "cancelled" });
    expect(calls).toEqual(["save", "check", "confirm"]);
  });

  it("предупреждения подтверждены — в очередь", async () => {
    const seen: ProblemIssue[][] = [];
    const { steps: s } = steps({
      check: async () => ({ kind: "ok", warnings: [warning], report: "r" }),
      confirm: async (w) => (seen.push(w), true),
    });
    expect((await launch(s)).kind).toBe("started");
    expect(seen).toEqual([[warning]]);
  });

  it("422 при постановке (индекс пересобрали) — проблемы", async () => {
    const { steps: s } = steps({ submit: async () => ({ kind: "problems" }) });
    expect(await launch(s)).toEqual({ kind: "problems", stage: "submit" });
  });

  it("без шага сохранения (Повторить)", async () => {
    const { calls, steps: s } = steps();
    delete s.save;
    expect((await launch(s)).kind).toBe("started");
    expect(calls).toEqual(["check", "submit"]);
  });

  it("сборка: подтверждение всегда, даже без предупреждений", async () => {
    const { calls, steps: s } = steps();
    s.confirmAlways = true;
    expect((await launch(s)).kind).toBe("started");
    expect(calls).toEqual(["save", "check", "confirm", "submit"]);
  });

  it("битый YAML — стоп до сервера", async () => {
    const { steps: s } = steps({ save: async () => ({ kind: "blocked" }) });
    expect(await launch(s)).toEqual({ kind: "blocked" });
  });
});
