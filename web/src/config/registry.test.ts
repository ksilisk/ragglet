import type { ProblemIssue } from "../api/client";
import { PROFILE_REGISTRY } from "./profileFields";
import { bindIssue, fieldDomId, isStale, sectionMarks } from "./registry";
import { SCENARIO_REGISTRY } from "./scenarioFields";

const issue = (path: string | null, severity: "error" | "warning" = "error", where: string | null = null): ProblemIssue => ({
  code: "config.value_error", severity, path, where, message: "m",
});
const bind = (path: string | null) => {
  const b = bindIssue(issue(path), SCENARIO_REGISTRY);
  return [b.section, b.anchor];
};

describe("привязка проблем к полям (ragglet-31 §5)", () => {
  it.each([
    ["metrics.quality.k", "metrics", "metrics.quality.k"],
    ["retrieval.sources[1].params.backend_ref", "retrieval", "retrieval.sources[1].params.backend_ref"],
    ["retrieval.sources[1].params", "retrieval", "retrieval.sources[1].params"],
    ["retrieval.sources[1].params.unknown_param", "retrieval", "retrieval.sources[1].params"],
    ["retrieval.sources[2]", "retrieval", "retrieval.sources[2]"],
    ["storage.backends", "data", "storage.backends"],
    ["storage.backends[1].endpoint", "data", "storage.backends[1].endpoint"],
    ["rerank", "rerank", "rerank"],
    ["rerank.bogus", "rerank", "rerank"],
    ["generation.enabled", "generation", "generation.enabled"],
    ["retrieval.async", "retrieval", "retrieval.async"],
    ["run.load.concurrency", "load", "run.load.concurrency"],
    ["cache.embedding.namespace", "cache", "cache.embedding.namespace"],
    ["name", "general", "name"],
  ])("%s → %s / %s", (path, section, anchor) => {
    expect(bind(path)).toEqual([section, anchor]);
  });

  it("без path и с неизвестным корнем — «Датасет и индекс», без поля", () => {
    expect(bind(null)).toEqual(["data", null]);
    expect(bind("bogus")).toEqual(["data", null]);
    expect(bindIssue(issue(null, "error", "line 3, column 2"), SCENARIO_REGISTRY).where).toBe("line 3, column 2");
  });

  it("профиль: свои секции", () => {
    expect(bindIssue(issue("chunking.chunk_overlap"), PROFILE_REGISTRY)).toMatchObject({ section: "chunking", anchor: "chunking.chunk_overlap" });
    expect(bindIssue(issue(null), PROFILE_REGISTRY)).toMatchObject({ section: "dataset", anchor: null });
  });
});

describe("устаревание после правки", () => {
  const touched = new Set(["metrics.quality.k", "retrieval.sources"]);
  it("поле правили — его проблема устарела, остальные нет", () => {
    expect(isStale("metrics.quality.k", touched)).toBe(true);
    expect(isStale("retrieval.sources[1].params.backend_ref", touched)).toBe(true);
    expect(isStale("rerank.output_top_n", touched)).toBe(false);
    expect(isStale("metrics.quality", touched)).toBe(false);
    expect(isStale(null, touched)).toBe(false);
  });
});

it("точки секций: ошибка важнее предупреждения", () => {
  const bound = [issue("rerank.model", "warning"), issue("rerank.output_top_n"), issue("metrics.quality.k", "warning")].map((i) => bindIssue(i, SCENARIO_REGISTRY));
  const marks = sectionMarks(bound);
  expect(marks.get("rerank")).toBe("error");
  expect(marks.get("metrics")).toBe("warning");
  expect(marks.get("data")).toBeUndefined();
});

it("fieldDomId", () => {
  expect(fieldDomId("retrieval.sources[1].params")).toBe("f-retrieval_sources_1__params");
});
