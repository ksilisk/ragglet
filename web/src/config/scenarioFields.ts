import type { Registry } from "./registry";

export type ScenarioSection = "general" | "data" | "retrieval" | "rerank" | "context" | "generation" | "cache" | "load" | "timeouts" | "metrics";

/** Секции в порядке стадий конвейера (ragglet-15, решение 3). */
export const SCENARIO_SECTIONS: ReadonlyArray<{ id: ScenarioSection; title: string }> = [
  { id: "data", title: "Датасет и индекс" },
  { id: "retrieval", title: "Retrieval" },
  { id: "rerank", title: "Rerank" },
  { id: "context", title: "Context" },
  { id: "generation", title: "Generation" },
  { id: "cache", title: "Кэш" },
  { id: "load", title: "Нагрузка и повторы" },
  { id: "timeouts", title: "Таймауты" },
  { id: "metrics", title: "Метрики и вывод" },
];

const fields: Record<ScenarioSection, string[]> = {
  general: ["name", "description", "tags", "version"],
  data: [
    "dataset", "dataset.source", "dataset.id", "dataset.revision", "dataset.split", "dataset.limit",
    "storage", "storage.mode", "storage.routing", "storage.routing.key",
    "storage.backend", "storage.backend.id", "storage.backend.type", "storage.backend.endpoint", "storage.backend.collection",
    "storage.backends", "storage.backends[]", "storage.backends[].id", "storage.backends[].type", "storage.backends[].endpoint", "storage.backends[].collection",
    "embedding", "embedding.model", "embedding.normalize",
  ],
  retrieval: [
    "retrieval", "retrieval.strategy", "retrieval.top_k", "retrieval.async",
    "retrieval.sources", "retrieval.sources[]", "retrieval.sources[].name", "retrieval.sources[].kind", "retrieval.sources[].weight",
    "retrieval.sources[].params", "retrieval.sources[].params.ef_search", "retrieval.sources[].params.backend_ref",
    "retrieval.sources[].params.k1", "retrieval.sources[].params.b",
    "retrieval.merge", "retrieval.merge.method", "retrieval.merge.rrf_k", "retrieval.merge.deduplicate",
  ],
  rerank: ["rerank", "rerank.enabled", "rerank.provider", "rerank.model", "rerank.revision", "rerank.input_top_k", "rerank.output_top_n"],
  context: ["context", "context.enabled", "context.policy", "context.max_tokens", "context.threshold", "context.include_metadata"],
  generation: [
    "generation", "generation.enabled", "generation.provider", "generation.model", "generation.base_url", "generation.api_key_env",
    "generation.temperature", "generation.max_output_tokens", "generation.save", "generation.save.prompt", "generation.save.answer", "generation.save.raw_response",
  ],
  cache: [
    "cache", "cache.enabled", "cache.embedding", "cache.embedding.type", "cache.embedding.endpoint", "cache.embedding.ttl_seconds", "cache.embedding.namespace",
    "cache.retrieval", "cache.retrieval.type", "cache.retrieval.size",
  ],
  load: ["seed", "run", "run.repeats", "run.shuffle", "run.save_per_query", "run.artifacts_dir", "run.load", "run.load.type", "run.load.concurrency"],
  timeouts: ["timeouts", "timeouts.total_ms", "timeouts.retrieval_ms", "timeouts.rerank_ms", "timeouts.llm_ms"],
  metrics: [
    "metrics", "metrics.quality", "metrics.quality.k", "metrics.quality.doc", "metrics.quality.chunk", "metrics.percentiles", "metrics.latency_steps",
    "output", "output.format", "output.include_debug_fields",
  ],
};

export const SCENARIO_REGISTRY: Registry<ScenarioSection> = {
  sections: SCENARIO_SECTIONS,
  anchors: Object.fromEntries(Object.entries(fields).flatMap(([section, paths]) => paths.map((p) => [p, section as ScenarioSection]))),
  fallback: "data",
  header: "general",
};
