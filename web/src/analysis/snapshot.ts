import { parseYaml } from "../config/yamlDoc";
import type { Stage } from "./summary";

/**
 * Что вкладкам результата нужно из snapshot сценария (?include=scenario). Snapshot — полный дамп
 * конфига, но у импортированных старых прогонов ключей может не быть: тогда null, а не умолчание.
 */
export interface SnapshotFacts {
  stagesOff: Stage[]; // выключенные в сценарии стадии
  cache: { enabled: boolean; emb: boolean; ret: boolean };
  totalBudgetMs: number | null; // timeouts.total_ms
  stageBudgetMs: Partial<Record<Stage, number>>;
  load: { type: string | null; concurrency: number | null };
  repeats: number | null;
  shuffle: boolean | null;
  seed: number | null;
  qualityK: number[];
  latencySteps: string[];
  savePerQuery: boolean | null;
  rerankModel: string | null;
  embeddingModel: string | null;
  datasetSplit: string | null;
  datasetLimit: number | null;
  contextEnabled: boolean;
}

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (v && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : {});
const numOr = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const strOr = (v: unknown) => (typeof v === "string" ? v : null);
const boolOr = (v: unknown) => (typeof v === "boolean" ? v : null);

export function snapshotFacts(yaml: string | null | undefined): SnapshotFacts {
  const cfg = obj(parseYaml(yaml ?? "").data);
  const run = obj(cfg.run);
  const load = obj(run.load);
  const cache = obj(cfg.cache);
  const timeouts = obj(cfg.timeouts);
  const metrics = obj(cfg.metrics);
  const dataset = obj(cfg.dataset);
  const enabled = (section: string) => boolOr(obj(cfg[section]).enabled);
  const stagesOff: Stage[] = [];
  if (enabled("rerank") === false) stagesOff.push("rerank");
  if (enabled("context") === false) stagesOff.push("context");
  if (enabled("generation") === false) stagesOff.push("generation");
  const cacheOn = boolOr(cache.enabled) === true;
  const budget: Partial<Record<Stage, number>> = {};
  const retrieval = numOr(timeouts.retrieval_ms);
  const rerank = numOr(timeouts.rerank_ms);
  const llm = numOr(timeouts.llm_ms);
  if (retrieval !== null) budget.retrieve = retrieval;
  if (rerank !== null && enabled("rerank")) budget.rerank = rerank;
  if (llm !== null && enabled("generation")) budget.generation = llm;
  return {
    stagesOff,
    cache: { enabled: cacheOn, emb: cacheOn && cache.embedding != null, ret: cacheOn && cache.retrieval != null },
    totalBudgetMs: numOr(timeouts.total_ms),
    stageBudgetMs: budget,
    load: { type: strOr(load.type), concurrency: numOr(load.concurrency) },
    repeats: numOr(run.repeats),
    shuffle: boolOr(run.shuffle),
    seed: numOr(cfg.seed),
    qualityK: Array.isArray(obj(metrics.quality).k) ? (obj(metrics.quality).k as unknown[]).filter((x): x is number => typeof x === "number") : [],
    latencySteps: Array.isArray(metrics.latency_steps) ? (metrics.latency_steps as unknown[]).map(String) : [],
    savePerQuery: boolOr(run.save_per_query),
    rerankModel: enabled("rerank") ? strOr(obj(cfg.rerank).model) : null,
    embeddingModel: strOr(obj(cfg.embedding).model),
    datasetSplit: strOr(dataset.split),
    datasetLimit: numOr(dataset.limit),
    contextEnabled: enabled("context") === true,
  };
}
