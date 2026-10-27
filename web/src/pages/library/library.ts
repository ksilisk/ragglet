import type { QueryClient } from "@tanstack/react-query";
import { api, settle, unwrap, type S } from "../../api/client";
import { keys } from "../../api/queries";
import { setIn } from "../../config/yamlDoc";

/** Библиотека конфигов (ragglet-28 §4): сценарии и профили индексов — одна логика, два набора ручек. */
export type LibraryKind = "scenario" | "profile";

export const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

export const LIBRARY = {
  scenario: { idField: "name", noun: "сценарий", genitive: "сценария", route: (id: string) => `/scenarios/${encodeURIComponent(id)}` },
  profile: { idField: "id", noun: "профиль", genitive: "профиля", route: (id: string) => `/indexes/profiles/${encodeURIComponent(id)}` },
} as const;

export function getDocument(kind: LibraryKind, id: string): Promise<S["ConfigDocument"]> {
  return kind === "scenario"
    ? unwrap(api.GET("/v1/scenarios/{scenario_id}", { params: { path: { scenario_id: id } } }))
    : unwrap(api.GET("/v1/index-profiles/{profile_id}", { params: { path: { profile_id: id } } }));
}

export function createDocument(kind: LibraryKind, yaml: string): Promise<S["ConfigDocument"]> {
  return kind === "scenario"
    ? unwrap(api.POST("/v1/scenarios", { body: { yaml } }))
    : unwrap(api.POST("/v1/index-profiles", { body: { yaml } }));
}

/** PUT с etag версии, с которой начата правка; 201 — файл создан. */
export async function writeDocument(kind: LibraryKind, id: string, yaml: string, etag: string | null): Promise<S["ConfigDocument"]> {
  const body = { yaml, etag };
  const { data } =
    kind === "scenario"
      ? await settle<S["ConfigDocument"]>(api.PUT("/v1/scenarios/{scenario_id}", { params: { path: { scenario_id: id } }, body }))
      : await settle<S["ConfigDocument"]>(api.PUT("/v1/index-profiles/{profile_id}", { params: { path: { profile_id: id } }, body }));
  return data;
}

export async function deleteDocument(kind: LibraryKind, id: string, etag: string | null): Promise<void> {
  const query = etag ? { etag } : {};
  if (kind === "scenario") await settle(api.DELETE("/v1/scenarios/{scenario_id}", { params: { path: { scenario_id: id }, query } }));
  else await settle(api.DELETE("/v1/index-profiles/{profile_id}", { params: { path: { profile_id: id }, query } }));
}

export function fromRun(runId: string, name: string): Promise<S["ConfigDocument"]> {
  return unwrap(api.POST("/v1/scenarios", { body: { from_run: runId, name } }));
}

/** Копия под новым именем: в тексте меняется одно значение name / id, комментарии на месте. */
export async function duplicate(kind: LibraryKind, sourceId: string, newId: string): Promise<S["ConfigDocument"]> {
  const source = await getDocument(kind, sourceId);
  return createDocument(kind, setIn(source.yaml, [LIBRARY[kind].idField], newId));
}

export function invalidateLibrary(qc: QueryClient, kind: LibraryKind) {
  void qc.invalidateQueries({ queryKey: kind === "scenario" ? keys.scenarios : keys.profiles });
  void qc.invalidateQueries({ queryKey: keys.indexes }); // used_by / profiles у коллекций
}

/** Короткий скелет: остальное приходит умолчаниями схемы. Сервер сохранит его черновиком. */
export function skeleton(kind: LibraryKind, id: string, qdrant = "http://localhost:6333"): string {
  if (kind === "scenario") {
    return `version: 1
name: "${id}"
description: ""
tags: []

dataset:
  source: "local"
  id: ""

storage:
  mode: "single"
  backend:
    type: "qdrant"
    endpoint: "${qdrant}"
    collection: ""

retrieval:
  strategy: "single"
  sources:
    - name: "qdrant_vector"
      kind: "vector"
      weight: 1.0
`;
  }
  return `id: "${id}"
dataset:
  source: "local"
  id: ""
chunking:
  method: "recursive"
  chunk_size: 800
  chunk_overlap: 120
embedding:
  model: "all-MiniLM-L6-v2"
  normalize: true
storage:
  type: "qdrant"
  endpoint: "${qdrant}"
  collection: ""
`;
}

export function downloadText(filename: string, text: string, type = "application/yaml") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function exportYaml(kind: LibraryKind, id: string) {
  const doc = await getDocument(kind, id);
  downloadText(`${id}.yaml`, doc.yaml);
}
