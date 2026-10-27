import { useMemo, useState } from "react";
import type { S } from "../../api/client";
import { useDatasets, useIndexes, useScenarioMeta } from "../../api/queries";
import { fieldDomId } from "../../config/registry";
import { fmtNumber, fmtWhen } from "../../lib/format";
import { Button } from "../../ui/Button";
import { Chip } from "../../ui/Chip";
import { cx } from "../../ui/cx";
import k from "../../ui/kit.module.css";
import { useEditorActions, useValue } from "../../editor/derived";
import s from "../../editor/editor.module.css";
import { CheckField, FieldMessages, FieldShell, Grid, Note, NumberField, SelectField, TextField } from "../../editor/fields";
import { Section } from "../../editor/Section";
import { useEditor } from "../../editor/store";
import { COLLECTION_STATE, hostOf } from "../indexes/IndexesPage";

type Collection = S["CollectionItem"];
const keyOf = (endpoint: unknown, collection: unknown) => `${String(endpoint ?? "")}|${String(collection ?? "")}`;
const norm = (endpoint: unknown) => String(endpoint ?? "").toLowerCase().replace(/\/$/, "");

export function useSelectedCollection(): Collection | null {
  const indexes = useIndexes();
  const endpoint = useValue<string>("storage.backend.endpoint");
  const collection = useValue<string>("storage.backend.collection");
  return useMemo(
    () => indexes.data?.collections.find((c) => norm(c.endpoint) === norm(endpoint) && c.name === collection) ?? null,
    [indexes.data, endpoint, collection],
  );
}

/** Строка о коллекции под выбором, как на артборде: состояние · points · датасет · модель · chunking · собрана · профиль. */
export function CollectionLine({ c }: { c: Collection }) {
  const meta = c.ragglet ?? {};
  const state = COLLECTION_STATE[c.state];
  const parts = [
    `${fmtNumber(c.points)} points`,
    meta.dataset_id ? `${String(meta.dataset_source)}:${String(meta.dataset_id)}` : null,
    meta.embedding_model ? String(meta.embedding_model) : null,
    meta.chunking_method ? `${String(meta.chunking_method)}${meta.chunk_size ? ` ${String(meta.chunk_size)}/${String(meta.chunk_overlap)}` : ""}` : null,
    meta.built_at ? `собрана ${fmtWhen(String(meta.built_at))}` : null,
    meta.index_profile_id ? `профиль ${String(meta.index_profile_id)}` : null,
  ].filter(Boolean);
  return (
    <span className={s.hint}>
      <Chip tone={state.tone} title={`status ${c.status}`}>
        {state.label}
      </Chip>{" "}
      {parts.join(" · ")}
    </span>
  );
}

function CollectionSelect({ path, label, collections, onPick }: {
  path: string;
  label: string;
  collections: Collection[];
  onPick: (c: Collection) => void;
}) {
  const endpoint = useValue(`${path}.endpoint`);
  const name = useValue(`${path}.collection`);
  const readOnly = useEditor((st) => st.parsed.error !== null);
  const current = collections.find((c) => norm(c.endpoint) === norm(endpoint) && c.name === name);
  const value = current ? keyOf(current.endpoint, current.name) : name ? keyOf(endpoint, name) : "";
  return (
    <FieldShell path={`${path}.collection`} label={label} resettable={false}>
      <select
        className={cx(k.inp, "mono")}
        value={value}
        disabled={readOnly}
        onChange={(e) => {
          const picked = collections.find((c) => keyOf(c.endpoint, c.name) === e.target.value);
          if (picked) onPick(picked);
        }}
      >
        {!value && <option value="">— выберите коллекцию —</option>}
        {value && !current && (
          <option value={value}>
            {String(name)} — {hostOf(String(endpoint ?? ""))} · не найдена
          </option>
        )}
        {collections.map((c) => (
          <option key={keyOf(c.endpoint, c.name)} value={keyOf(c.endpoint, c.name)}>
            {c.name} — {hostOf(c.endpoint)}
            {c.state !== "ready" ? ` (${COLLECTION_STATE[c.state].label})` : ""}
          </option>
        ))}
      </select>
      <FieldMessages path={`${path}.endpoint`} />
    </FieldShell>
  );
}

export function DataSection() {
  const meta = useScenarioMeta().data!;
  const indexes = useIndexes();
  const datasets = useDatasets();
  const { edit, append, removeAt } = useEditorActions();
  const collections = indexes.data?.collections ?? [];
  const selected = useSelectedCollection();
  const mode = useValue<string>("storage.mode") ?? "single";
  const backends = useValue<Array<Record<string, unknown>>>("storage.backends") ?? [];
  const source = useValue<string>("dataset.source") ?? "local";
  const datasetId = useValue<string>("dataset.id");
  const embeddingModel = useValue<string>("embedding.model");
  const embeddingNormalize = useValue<boolean>("embedding.normalize");
  const [manual, setManual] = useState(false);
  const collectionName = useValueText("storage.backend.collection");

  const supportsRevision = meta.dataset_sources.find((d) => d.id === source)?.supports_revision ?? false;
  const local = datasets.data?.local ?? [];
  const splits = local.find((d) => d.id === datasetId)?.splits ?? [];
  const ragglet = selected?.ragglet ?? null;
  const indexModel = ragglet?.embedding_model as string | undefined;
  const indexNormalize = ragglet?.embedding_normalize as boolean | undefined;
  const effectiveModel = embeddingModel ?? "all-MiniLM-L6-v2";
  const effectiveNormalize = embeddingNormalize ?? true;
  const fromIndex = Boolean(indexModel) && effectiveModel === indexModel && (indexNormalize === undefined || effectiveNormalize === indexNormalize) && !manual;

  const pickPrimary = (c: Collection) => {
    edit(["storage", "backend", "endpoint"], c.endpoint);
    edit(["storage", "backend", "collection"], c.name);
    const m = c.ragglet;
    if (!m) return;
    if (m.embedding_model && m.embedding_model !== effectiveModel) edit(["embedding", "model"], m.embedding_model);
    if (typeof m.embedding_normalize === "boolean" && m.embedding_normalize !== effectiveNormalize) edit(["embedding", "normalize"], m.embedding_normalize);
    if (!datasetId && m.dataset_id) {
      edit(["dataset", "source"], m.dataset_source);
      edit(["dataset", "id"], m.dataset_id);
    }
  };

  const datasetDiffers = Boolean(ragglet?.dataset_id && datasetId && (ragglet.dataset_id !== datasetId || ragglet.dataset_source !== source));

  return (
    <Section id="data" title="Датасет и индекс" summary={`${source}:${datasetId ?? "—"} · ${collectionName}`}>
      <div className={cx(s.grid)} style={{ gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)" }}>
        <div className={s.field}>
          <CollectionSelect path="storage.backend" label={mode === "single" ? "Коллекция (индекс)" : "Основная коллекция (storage.backend)"} collections={collections} onPick={pickPrimary} />
          {selected && <CollectionLine c={selected} />}
          {indexes.data && indexes.data.endpoints.some((e) => !e.available) && (
            <span className={cx(s.hint, s.warn)}>Часть Qdrant недоступна: их коллекций нет в списке.</span>
          )}
        </div>
        <SelectField path="storage.mode" label="storage.mode" options={["single", "sharded", "replicated"]} />
      </div>

      {mode !== "single" && (
        <div className={s.cards} id={fieldDomId("storage.backends")}>
          <FieldMessages path="storage.backends" />
          {backends.map((b, i) => (
            <div key={i} className={s.itemCard} id={fieldDomId(`storage.backends[${i}]`)}>
              <div className={s.itemHead}>
                <span className="mono grow">{String(b.id ?? `цель ${i + 1}`)}</span>
                <Button small onClick={() => removeAt(["storage", "backends"], i)}>
                  Убрать
                </Button>
              </div>
              <Grid cols={3}>
                <TextField path={`storage.backends[${i}].id`} label="id" mono optional={false} />
                <div className={`${s.field} ${s.span2}`}>
                  <CollectionSelect
                    path={`storage.backends[${i}]`}
                    label="Коллекция"
                    collections={collections}
                    onPick={(c) => {
                      edit(["storage", "backends", i, "endpoint"], c.endpoint);
                      edit(["storage", "backends", i, "collection"], c.name);
                    }}
                  />
                </div>
              </Grid>
            </div>
          ))}
          <div>
            <Button small onClick={() => append(["storage", "backends"], { id: `shard_${backends.length}`, type: "qdrant", endpoint: selected?.endpoint ?? "", collection: "" })}>
              + цель storage.backends
            </Button>
          </div>
        </div>
      )}

      <Grid cols={5}>
        <SelectField path="dataset.source" label="dataset.source" options={meta.dataset_sources.map((d) => d.id)} />
        <TextField path="dataset.id" label="dataset.id" mono optional={false} span={2} list="dl-dataset-ids" placeholder={source === "hf" ? "BeIR/nfcorpus" : "datasets/demo"} />
        {supportsRevision ? <TextField path="dataset.revision" label="revision" mono placeholder="последняя" /> : <div />}
        <TextField path="dataset.split" label="split" mono list="dl-dataset-splits" />
      </Grid>
      <datalist id="dl-dataset-ids">
        {(source === "hf" ? (datasets.data?.hf ?? []).map((d) => d.id) : local.map((d) => d.id)).map((id) => (
          <option key={id} value={id} />
        ))}
      </datalist>
      <datalist id="dl-dataset-splits">
        {splits.map((sp) => (
          <option key={sp} value={sp} />
        ))}
      </datalist>
      {datasetDiffers && ragglet && (
        <div className={s.banner}>
          Коллекция собрана по {String(ragglet.dataset_source)}:{String(ragglet.dataset_id)}, а в сценарии — {source}:{datasetId}.
          <Button
            small
            onClick={() => {
              edit(["dataset", "source"], ragglet.dataset_source);
              edit(["dataset", "id"], ragglet.dataset_id);
            }}
          >
            Взять датасет коллекции
          </Button>
        </div>
      )}

      <Grid cols={5}>
        <NumberField path="dataset.limit" label="limit" min={1} />
        {fromIndex ? (
          <>
            <div className={`${s.field} ${s.span3}`}>
              <span className={s.label}>embedding — из индекса</span>
              <input className={cx(k.inp, k.ro, "mono")} value={`${effectiveModel} · normalize: ${String(effectiveNormalize)}`} readOnly aria-label="embedding из индекса" />
            </div>
            <div className={s.field} style={{ alignSelf: "end" }}>
              <Button onClick={() => setManual(true)}>Задать вручную</Button>
            </div>
          </>
        ) : (
          <>
            <TextField path="embedding.model" label="embedding.model" mono span={3} />
            <div style={{ alignSelf: "end" }}>
              <CheckField path="embedding.normalize" label="normalize" />
            </div>
          </>
        )}
      </Grid>
      {!fromIndex && indexModel && (
        <Note>
          В индексе: {indexModel}, normalize: {String(indexNormalize)}. Preflight сверяет модель с метаданными коллекции строго по строке.
        </Note>
      )}
    </Section>
  );
}

function useValueText(path: string): string {
  const v = useValue(path);
  return v === undefined || v === null || v === "" ? "—" : String(v);
}
