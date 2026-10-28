import { msg } from "../i18n/i18n";
import type { Registry } from "./registry";

export type ProfileSection = "general" | "dataset" | "chunking" | "embedding" | "storage" | "index";

export const PROFILE_SECTIONS: ReadonlyArray<{ id: ProfileSection; title: string }> = [
  // title — msg(): переводится при выводе (SectionNav)
  { id: "dataset", title: msg("Датасет") },
  { id: "chunking", title: msg("Чанкинг") },
  { id: "embedding", title: msg("Эмбеддинг") },
  { id: "storage", title: msg("Хранилище") },
  { id: "index", title: msg("Параметры индекса") },
];

const fields: Record<ProfileSection, string[]> = {
  general: ["id"],
  dataset: ["dataset", "dataset.source", "dataset.id", "dataset.revision"],
  chunking: ["chunking", "chunking.method", "chunking.chunk_size", "chunking.chunk_overlap"],
  embedding: ["embedding", "embedding.model", "embedding.revision", "embedding.normalize", "embedding.batch_size"],
  storage: [
    "storage", "storage.mode", "storage.type", "storage.endpoint", "storage.collection", "storage.routing", "storage.routing.key",
    "storage.backend", "storage.backend.id", "storage.backend.type", "storage.backend.endpoint", "storage.backend.collection",
    "storage.backends", "storage.backends[]", "storage.backends[].id", "storage.backends[].type", "storage.backends[].endpoint", "storage.backends[].collection",
  ],
  index: ["index_params", "index_params.distance", "index_params.hnsw", "index_params.hnsw.m", "index_params.hnsw.ef_construct"],
};

export const PROFILE_REGISTRY: Registry<ProfileSection> = {
  sections: PROFILE_SECTIONS,
  anchors: Object.fromEntries(Object.entries(fields).flatMap(([section, paths]) => paths.map((p) => [p, section as ProfileSection]))),
  fallback: "dataset",
  header: "general",
};
