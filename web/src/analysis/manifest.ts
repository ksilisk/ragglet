import { msg } from "../i18n/i18n";

/**
 * manifest.json прогона (ragglet-17/23) в виде, удобном для «Условий»: что где лежит, без
 * интерпретации сверх ragglet-23. Пустые поля — null; у прогонов до ragglet-23 секций меньше.
 */
type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : null);
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
const numOr = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export interface EnvFacts {
  commit: string | null;
  python: string | null;
  platform: string | null;
  cpu: string | null;
  cpus: number | null;
  ram: number | null;
  device: string | null;
  accelerator: string | null;
  threads: number | null;
}

export interface ServiceFacts {
  role: string; // storage, cache
  name: string; // default, embedding
  type: string | null;
  version: string | null;
  status: string | null;
  points: number | null;
  indexed: number | null;
  segments: number | null;
}

export interface IndexFacts {
  profileId: string | null;
  builtAt: string | null;
  embeddingRevision: string | null;
  chunking: { method: string | null; size: number | null; overlap: number | null } | null;
  vectorSize: number | null;
  distance: string | null;
  hnsw: { m: number | null; efConstruct: number | null } | null;
}

export interface ManifestFacts {
  dataset: { revision: string | null; corpus: string | null; eval: string | null } | null;
  index: IndexFacts | null;
  buildEnv: EnvFacts | null;
  rerank: { revision: string | null } | null;
  env: EnvFacts | null;
  services: ServiceFacts[];
}

function envFacts(v: unknown): EnvFacts | null {
  const env = obj(v);
  if (!env) return null;
  const system = obj(env.system) ?? {};
  const compute = obj(env.compute) ?? {};
  return {
    commit: str(env.commit),
    python: str(env.python),
    platform: str(system.platform),
    cpu: str(system.cpu),
    cpus: numOr(system.cpus),
    ram: numOr(system.ram_bytes),
    device: str(compute.device),
    accelerator: str(compute.accelerator),
    threads: numOr(compute.torch_threads),
  };
}

export function manifestFacts(manifest: Obj | null | undefined): ManifestFacts | null {
  if (!manifest) return null;
  const dataset = obj(manifest.dataset);
  const index = obj(manifest.index);
  const params = obj(index?.params) ?? {};
  const chunking = obj(index?.chunking);
  const hnsw = obj(params.hnsw);
  const services: ServiceFacts[] = [];
  for (const [role, group] of Object.entries(obj(obj(manifest.env)?.services) ?? {})) {
    for (const [name, raw] of Object.entries(obj(group) ?? {})) {
      const s = obj(raw) ?? {};
      services.push({
        role,
        name,
        type: str(s.type),
        version: str(s.version),
        status: str(s.status),
        points: numOr(s.points),
        indexed: numOr(s.indexed_vectors),
        segments: numOr(s.segments),
      });
    }
  }
  return {
    dataset: dataset ? { revision: str(dataset.revision), corpus: str(dataset.corpus_sha256), eval: str(dataset.eval_sha256) } : null,
    index: index
      ? {
          profileId: str(index.profile_id),
          builtAt: str(index.built_at),
          embeddingRevision: str(index.embedding_revision),
          chunking: chunking ? { method: str(chunking.method), size: numOr(chunking.chunk_size), overlap: numOr(chunking.chunk_overlap) } : null,
          vectorSize: numOr(params.vector_size),
          distance: str(params.distance),
          hnsw: hnsw ? { m: numOr(hnsw.m), efConstruct: numOr(hnsw.ef_construct) } : null,
        }
      : null,
    buildEnv: envFacts(index?.build_env),
    rerank: obj(manifest.rerank) ? { revision: str(obj(manifest.rerank)!.revision) } : null,
    env: envFacts(manifest.env),
    services,
  };
}

/** Поля build_env, которые отличаются от окружения прогона: «≠ прогону» в «Условиях». */
export function envDiffers(build: EnvFacts | null, run: EnvFacts | null): Set<keyof EnvFacts> {
  const out = new Set<keyof EnvFacts>();
  if (!build || !run) return out;
  for (const key of Object.keys(build) as Array<keyof EnvFacts>) {
    if (build[key] !== null && run[key] !== null && build[key] !== run[key]) out.add(key);
  }
  return out;
}

export interface DerivedWarning {
  code: string;
  text: string; // msg(): перевод при выводе
  vars: Record<string, string | number>;
}

/**
 * Предупреждения, которые видны по manifest (ragglet-23): коллекция не green, HNSW не построен
 * (indexed_vectors = 0 при points > 0 — поиск полным перебором), commit неизвестен.
 */
export function derivedWarnings(facts: ManifestFacts | null): DerivedWarning[] {
  if (!facts) return [];
  const out: DerivedWarning[] = [];
  for (const s of facts.services) {
    if (s.role !== "storage") continue;
    if (s.status && s.status !== "green") out.push({ code: "index.not_green", text: msg("цель {name}: статус коллекции {status}"), vars: { name: s.name, status: s.status } });
    if (s.indexed === 0 && (s.points ?? 0) > 0) out.push({ code: "indexed_vectors = 0", text: msg("цель {name}: HNSW не построен, поиск полным перебором"), vars: { name: s.name } });
  }
  if (facts.env && facts.env.commit === null) out.push({ code: "env.commit", text: msg("commit неизвестен: сервер запущен не из git-checkout"), vars: {} });
  return out;
}

export const short = (hash: string | null, n = 8) => (hash ? hash.slice(0, n) : null);
