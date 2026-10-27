import { finalDepth, profileHints, scenarioHints } from "./hints";

const defaults = { retrieval: { top_k: 30 }, rerank: { input_top_k: 30, output_top_n: 10 }, timeouts: { total_ms: 6000, retrieval_ms: 1500, rerank_ms: 800, llm_ms: 3500 }, metrics: { quality: { k: [1, 3, 5, 10] } } };
const warnings = (data: Record<string, unknown>, path: string) =>
  scenarioHints(data, defaults).filter((h) => h.path === path && h.severity === "warning").map((h) => h.message);

describe("глубина финального списка", () => {
  it("без rerank — top_k", () => {
    expect(finalDepth({ retrieval: { top_k: 50 } }, defaults)).toMatchObject({ depth: 50, source: "retrieval.top_k" });
  });
  it("с rerank — минимум из трёх", () => {
    expect(finalDepth({ retrieval: { top_k: 50 }, rerank: { enabled: true, input_top_k: 50, output_top_n: 10 } }, defaults)).toMatchObject({
      depth: 10,
      source: "rerank.output_top_n",
    });
  });
  it("k больше глубины — предупреждение у metrics.quality.k", () => {
    const data = { retrieval: { top_k: 50 }, rerank: { enabled: true, input_top_k: 50, output_top_n: 10 }, metrics: { quality: { k: [1, 10, 30] } } };
    expect(warnings(data, "metrics.quality.k")).toEqual(["k = 30 больше глубины финального списка 10 (rerank.output_top_n): проверка не пройдёт."]);
    expect(warnings({ ...data, metrics: { quality: { k: [1, 10] } } }, "metrics.quality.k")).toEqual([]);
  });
});

it("input_top_k > top_k", () => {
  expect(warnings({ retrieval: { top_k: 20 }, rerank: { enabled: true, input_top_k: 50 } }, "rerank.input_top_k")).toHaveLength(1);
  expect(warnings({ retrieval: { top_k: 50 }, rerank: { enabled: true, input_top_k: 50 } }, "rerank.input_top_k")).toHaveLength(0);
});

it("бюджеты таймаутов по включённым стадиям", () => {
  expect(warnings({}, "timeouts.total_ms")).toHaveLength(0); // 1500 ≤ 6000
  expect(warnings({ rerank: { enabled: true }, generation: { enabled: true }, timeouts: { total_ms: 5000 } }, "timeouts.total_ms")).toHaveLength(1); // 5800 > 5000
});

it("источники retrieval", () => {
  const two = { retrieval: { strategy: "single", sources: [{ kind: "vector" }, { kind: "vector" }] } };
  expect(warnings(two, "retrieval.sources")[0]).toMatch(/только первый/);
  const kw = { retrieval: { strategy: "fanout_merge", sources: [{ kind: "keyword" }, { kind: "keyword" }] } };
  expect(warnings(kw, "retrieval.sources")[0]).toMatch(/одного keyword/);
});

it("коллекция другого датасета и модели", () => {
  const data = { dataset: { source: "local", id: "datasets/demo" }, embedding: { model: "bge" } };
  const hints = scenarioHints(data, defaults, { dataset_source: "hf", dataset_id: "BeIR/nfcorpus", embedding_model: "all-MiniLM-L6-v2" });
  expect(hints.filter((h) => h.severity === "warning").map((h) => h.path).sort()).toEqual(["dataset.id", "embedding.model"]);
});

it("профиль: overlap < size", () => {
  expect(profileHints({ chunking: { method: "recursive", chunk_size: 100, chunk_overlap: 100 } })).toHaveLength(1);
  expect(profileHints({ chunking: { method: "recursive", chunk_size: 800, chunk_overlap: 120 } })).toHaveLength(0);
  expect(profileHints({ chunking: { method: "none" } })).toHaveLength(0);
});
