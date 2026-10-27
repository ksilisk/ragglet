import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import { ApiProblem, api, settle, type S } from "../../api/client";
import { keys, useDatasets, useIndexes, useProfileMeta, useTasks } from "../../api/queries";
import { profileHints } from "../../config/hints";
import { PROFILE_REGISTRY, PROFILE_SECTIONS } from "../../config/profileFields";
import { bindIssues, fieldDomId } from "../../config/registry";
import { enumAt } from "../../config/schema";
import { dryRun } from "../../editor/commands";
import { useEditorActions, useValue } from "../../editor/derived";
import s from "../../editor/editor.module.css";
import { EditorShell } from "../../editor/EditorShell";
import { CheckField, FieldMessages, Grid, Note, NumberField, SelectField, TextField } from "../../editor/fields";
import { launch } from "../../editor/launch";
import { Section } from "../../editor/Section";
import { EditorContext, checkFromProblem, createEditorStore, useEditor, type CheckResult } from "../../editor/store";
import { useEditorPage, useLibraryDocument } from "../../editor/useEditorPage";
import { fmtDataset, fmtNumber, fmtWhen } from "../../lib/format";
import { Button } from "../../ui/Button";
import { Chip } from "../../ui/Chip";
import { cx } from "../../ui/cx";
import { useConfirm } from "../../ui/Dialog";
import k from "../../ui/kit.module.css";
import { Crumbs, Empty, Page } from "../../ui/Layout";
import { Menu } from "../../ui/Menu";
import { ErrorBox, IssueList } from "../../ui/ProblemView";
import { hostOf } from "../indexes/IndexesPage";

export default function ProfileEditorPage() {
  const { id = "" } = useParams();
  const doc = useLibraryDocument("profile", id);
  const meta = useProfileMeta();
  const crumbs = <Crumbs items={[{ label: "Индексы", to: "/indexes" }, { label: "Профили", to: "/indexes?tab=profiles" }, { label: id }]} />;
  if (doc.error || meta.error) {
    const notFound = doc.error instanceof ApiProblem && doc.error.status === 404;
    return (
      <Page>
        {crumbs}
        {notFound ? <Empty>Профиля {id} нет в библиотеке.</Empty> : <ErrorBox error={doc.error ?? meta.error} />}
      </Page>
    );
  }
  if (!doc.data || !meta.data) {
    return (
      <Page>
        {crumbs}
        <span className="muted">Загрузка…</span>
      </Page>
    );
  }
  return <ProfileEditor key={id} doc={doc.data} meta={meta.data} />;
}

function ProfileEditor({ doc, meta }: { doc: S["ConfigDocument"]; meta: S["IndexProfileMeta"] }) {
  const open = { dataset: true, chunking: true, embedding: true, storage: true, index: false };
  for (const issue of bindIssues(doc.problem?.issues ?? [], PROFILE_REGISTRY)) (open as Record<string, boolean>)[issue.section] = true;
  const [store] = useState(() => createEditorStore("profile", doc, open));
  const server = useLibraryDocument("profile", doc.id).data;
  const page = useEditorPage(store, "profile", server);
  const ctx = useMemo(() => ({ store, registry: PROFILE_REGISTRY, defaults: meta.defaults, hints: profileHints }), [store, meta.defaults]);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const confirm = useConfirm();
  const tasks = useTasks();
  const [params, setParams] = useSearchParams();
  const autoBuild = useRef(params.get("build") === "1");

  const openErrorSections = (check: CheckResult) => {
    for (const issue of bindIssues(check.issues, PROFILE_REGISTRY)) if (issue.severity === "error") store.getState().setOpen(issue.section, true);
  };

  const onCheck = async () => {
    const out = await page.onCheck();
    if (out?.kind === "problems") openErrorSections(out.check);
  };

  // Собрать = сохранить → dry-run сборки → подтвердить «Будут пересозданы …» → 202 → страница сборки.
  const onBuild = () =>
    page.run("launch", async () => {
      const busy = (tasks.data ?? []).some((t) => t.state === "running");
      const result = await launch<S["BuildCheck"]>({
        confirmAlways: true,
        save: page.doSave,
        check: async () => {
          const out = await dryRun(store, "profile");
          if (out.kind === "problems") openErrorSections(out.check);
          return out.kind === "ok" ? { kind: "ok", warnings: out.check.issues, report: out.check.report as S["BuildCheck"] } : { kind: out.kind };
        },
        confirm: (_warnings, check) =>
          confirm({
            title: `Собрать индекс по профилю ${check.profile}?`,
            body: <BuildConfirmBody check={check} />,
            confirm: busy ? "Собрать · в очередь после текущей задачи" : "Собрать",
            wide: true,
          }),
        submit: async () => {
          const st = store.getState();
          try {
            const { data } = await settle<S["BuildAccepted"]>(api.POST("/v1/builds", { body: { profile: st.checkedText ?? st.text, profile_id: st.id, dry_run: false } }));
            return { kind: "accepted", id: data.id, position: data.position ?? null };
          } catch (e) {
            if (!(e instanceof ApiProblem && e.status === 422)) throw e;
            const check = checkFromProblem("launch", e.problem);
            st.setCheck(check, st.checkedText ?? st.text);
            openErrorSections(check);
            return { kind: "problems" };
          }
        },
      });
      if (result.kind === "started") {
        void qc.invalidateQueries({ queryKey: keys.tasks });
        void qc.invalidateQueries({ queryKey: keys.builds });
        void qc.invalidateQueries({ queryKey: keys.profiles });
        page.leaving.current = true;
        navigate(`/builds/${encodeURIComponent(result.id)}`);
      }
      return result;
    });

  // «Пересобрать» / «Собрать ▾» со страницы Индексы: сразу проверка и подтверждение.
  useEffect(() => {
    if (!autoBuild.current) return;
    autoBuild.current = false;
    setParams({}, { replace: true });
    void onBuild();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- один раз при открытии
  }, []);

  const actions = (
    <>
      <Button onClick={() => void page.onSave()} disabled={!page.dirty || page.broken || page.busy !== null}>
        {page.busy === "save" ? "Сохранение…" : "Сохранить"}
      </Button>
      <Button onClick={() => void onCheck()} disabled={page.broken || page.busy !== null}>
        {page.busy === "check" ? "Проверка…" : "Проверить"}
      </Button>
      <Button variant="primary" onClick={() => void onBuild()} disabled={page.broken || page.busy !== null} title="Сохранить, проверить и поставить сборку в очередь">
        {page.busy === "launch" ? "Сборка…" : "Собрать"}
      </Button>
      <Menu
        trigger={<Button aria-label="Ещё действия">⋯</Button>}
        items={[
          { label: "Сохранить как…", onSelect: page.onSaveAs },
          { label: "Сбросить к сохранённому", onSelect: () => void page.onReset(), disabled: !page.dirty },
          { label: "Экспорт YAML", onSelect: page.onExport },
          { label: "Удалить…", onSelect: () => void page.onDelete(), danger: true },
        ]}
      />
    </>
  );

  return (
    <EditorContext.Provider value={ctx}>
      <Page>
        <EditorShell head={<ProfileHead actions={actions} note={page.note} banners={page.banners} />} sections={PROFILE_SECTIONS} report={buildReport}>
          <DatasetSection meta={meta} />
          <ChunkingSection meta={meta} />
          <EmbeddingSection />
          <StorageSection meta={meta} />
          <IndexSection meta={meta} />
        </EditorShell>
        {page.dialogs}
      </Page>
    </EditorContext.Provider>
  );
}

function ProfileHead({ actions, note, banners }: { actions: ReactNode; note: string | null; banners: ReactNode }) {
  const id = useEditor((st) => st.id);
  const dirty = useEditor((st) => st.text !== st.saved.text);
  const valid = useEditor((st) => st.saved.valid);
  const value = useValue<string>("id");
  return (
    <>
      <Crumbs items={[{ label: "Индексы", to: "/indexes" }, { label: "Профили", to: "/indexes?tab=profiles" }, { label: id }]} />
      <div className={s.head}>
        <div className={s.headFields}>
          <div className={s.headRow}>
            <label htmlFor="f-id-input" className={s.headLabel}>
              id
            </label>
            <input id="f-id-input" className={cx(k.inp, k.ro, s.nameInput)} value={value ?? ""} readOnly title="id — это имя файла: переименование — «Сохранить как…»" />
            {dirty && <Chip tone="warn">не сохранено</Chip>}
            {!valid && <Chip tone="fail">черновик</Chip>}
          </div>
          <div id={fieldDomId("id")}>
            <FieldMessages path="id" />
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
          <div className={k.actions}>{actions}</div>
          {note && <span className={s.saveNote}>{note}</span>}
        </div>
      </div>
      {banners}
    </>
  );
}

function DatasetSection({ meta }: { meta: S["IndexProfileMeta"] }) {
  const datasets = useDatasets();
  const source = useValue<string>("dataset.source") ?? "local";
  const id = useValue<string>("dataset.id");
  const supportsRevision = meta.dataset_sources.find((d) => d.id === source)?.supports_revision ?? false;
  const ids = source === "hf" ? (datasets.data?.hf ?? []).map((d) => d.id) : (datasets.data?.local ?? []).map((d) => d.id);
  return (
    <Section id="dataset" title="Датасет" summary={`${source}:${id ?? "—"}`}>
      <Grid cols={4}>
        <SelectField path="dataset.source" label="dataset.source" options={meta.dataset_sources.map((d) => d.id)} />
        <TextField path="dataset.id" label="dataset.id" mono optional={false} span={2} list="dl-profile-datasets" placeholder={source === "hf" ? "BeIR/nfcorpus" : "datasets/demo"} />
        {supportsRevision && <TextField path="dataset.revision" label="revision" mono placeholder="последняя" />}
      </Grid>
      <datalist id="dl-profile-datasets">
        {ids.map((x) => (
          <option key={x} value={x} />
        ))}
      </datalist>
    </Section>
  );
}

function ChunkingSection({ meta }: { meta: S["IndexProfileMeta"] }) {
  const method = useValue<string>("chunking.method") ?? "recursive";
  const size = useValue<number>("chunking.chunk_size");
  const overlap = useValue<number>("chunking.chunk_overlap");
  return (
    <Section id="chunking" title="Чанкинг" summary={method === "recursive" ? `recursive ${size ?? 800}/${overlap ?? 120}` : method}>
      <Grid cols={4}>
        <SelectField path="chunking.method" label="method" options={enumAt(meta.json_schema, ["chunking", "method"])} />
        {method === "recursive" && (
          <>
            <NumberField path="chunking.chunk_size" label="chunk_size" min={1} placeholder="800" />
            <NumberField path="chunking.chunk_overlap" label="chunk_overlap" min={0} placeholder="120" />
          </>
        )}
      </Grid>
      {method === "none" && <Note>none — документ целиком одним чанком; размеры при этом запрещены схемой.</Note>}
    </Section>
  );
}

function EmbeddingSection() {
  const model = useValue<string>("embedding.model");
  return (
    <Section id="embedding" title="Эмбеддинг" summary={model ?? "—"}>
      <Grid cols={4}>
        <TextField path="embedding.model" label="model" mono span={2} optional={false} placeholder="all-MiniLM-L6-v2" />
        <TextField path="embedding.revision" label="revision" mono placeholder="main" />
        <NumberField path="embedding.batch_size" label="batch_size" min={1} />
      </Grid>
      <CheckField path="embedding.normalize" label="normalize" />
    </Section>
  );
}

function StorageSection({ meta }: { meta: S["IndexProfileMeta"] }) {
  const { append, removeAt } = useEditorActions();
  const indexes = useIndexes();
  const mode = useValue<string>("storage.mode") ?? "single";
  const endpoint = useValue<string>("storage.endpoint");
  const collection = useValue<string>("storage.collection");
  const backends = useValue<Array<Record<string, unknown>>>("storage.backends") ?? [];
  const known = [...new Set((indexes.data?.collections ?? []).map((c) => c.name))];
  return (
    <Section
      id="storage"
      title="Хранилище"
      summary={mode === "single" ? `${hostOf(endpoint ?? "")} / ${collection ?? "—"}` : `${mode} · ${backends.length} ${backends.length === 1 ? "цель" : "цели"}`}
    >
      <Grid cols={4}>
        <SelectField path="storage.mode" label="storage.mode" options={enumAt(meta.json_schema, ["storage", "mode"])} />
        <SelectField path="storage.type" label="storage.type" options={meta.storage_backends} />
        {mode !== "single" && <SelectField path="storage.routing.key" label="routing.key" options={enumAt(meta.json_schema, ["storage", "routing", "key"])} />}
      </Grid>
      <Grid cols={4}>
        <TextField path="storage.endpoint" label="endpoint" mono span={2} placeholder="http://localhost:6333" />
        <TextField path="storage.collection" label="collection" mono span={2} list="dl-profile-collections" />
      </Grid>
      <datalist id="dl-profile-collections">
        {known.map((c) => (
          <option key={c} value={c} />
        ))}
      </datalist>
      {mode !== "single" && (
        <div className={s.cards} id={fieldDomId("storage.backends")}>
          <h3 className={s.subhead}>Цели (storage.backends)</h3>
          <FieldMessages path="storage.backends" />
          {backends.map((b, i) => (
            <div key={i} className={s.itemCard} id={fieldDomId(`storage.backends[${i}]`)}>
              <div className={s.itemHead}>
                <span className="mono grow">{String(b.id ?? `цель ${i + 1}`)}</span>
                <Button small onClick={() => removeAt(["storage", "backends"], i)}>
                  Убрать
                </Button>
              </div>
              <Grid cols={4}>
                <TextField path={`storage.backends[${i}].id`} label="id" mono optional={false} />
                <TextField path={`storage.backends[${i}].endpoint`} label="endpoint" mono optional={false} />
                <TextField path={`storage.backends[${i}].collection`} label="collection" mono optional={false} span={2} list="dl-profile-collections" />
              </Grid>
            </div>
          ))}
          <div>
            <Button
              small
              onClick={() => append(["storage", "backends"], { id: `shard_${backends.length}`, type: "qdrant", endpoint: endpoint ?? "http://localhost:6333", collection: `${collection ?? "chunks"}_shard${backends.length}` })}
            >
              + цель
            </Button>
          </div>
        </div>
      )}
    </Section>
  );
}

function IndexSection({ meta }: { meta: S["IndexProfileMeta"] }) {
  const distance = useValue<string>("index_params.distance") ?? "cosine";
  const m = useValue<number>("index_params.hnsw.m") ?? 16;
  const ef = useValue<number>("index_params.hnsw.ef_construct") ?? 128;
  return (
    <Section id="index" title="Параметры индекса" summary={`${distance} · HNSW ${m}/${ef}`}>
      <Grid cols={4}>
        <SelectField path="index_params.distance" label="distance" options={enumAt(meta.json_schema, ["index_params", "distance"])} />
        <NumberField path="index_params.hnsw.m" label="hnsw.m" min={1} />
        <NumberField path="index_params.hnsw.ef_construct" label="hnsw.ef_construct" min={1} />
      </Grid>
    </Section>
  );
}

/** Тело подтверждения сборки (диалог на артборде Indexes): что будет пересоздано и кто этим пользуется. */
function BuildConfirmBody({ check }: { check: S["BuildCheck"] }) {
  return (
    <>
      <span>
        Dry-run прошёл: датасет {fmtDataset(check.dataset)}, {fmtNumber(check.documents)} документов, corpus sha256{" "}
        <span className="mono">{check.corpus_sha256.slice(0, 8)}…</span>
        {check.warnings.length ? `, предупреждений: ${check.warnings.length}` : ", предупреждений нет"}.
      </span>
      {check.warnings.length > 0 && <IssueList issues={check.warnings} />}
      <span>
        <b>{check.targets.some((t) => t.exists) ? "Будут пересозданы:" : "Будут созданы:"}</b>
      </span>
      <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: "4px 12px" }}>
        {check.targets.map((t) => (
          <div key={t.id} style={{ display: "contents" }}>
            <span className="mono">{t.collection}</span>
            <span className="muted">
              {hostOf(t.endpoint)} ·{" "}
              {!t.reachable
                ? `недоступен: ${t.error ?? "нет ответа"}`
                : t.exists
                  ? `существует, ${fmtNumber(t.points)} points${t.built_at ? `, собрана ${fmtWhen(t.built_at)}` : ", без метаданных"}`
                  : "новая коллекция"}
              {t.used_by.length > 0 && (
                <>
                  {" "}
                  · используют сценарии <span className="mono">{t.used_by.join(", ")}</span>
                </>
              )}
            </span>
          </div>
        ))}
      </div>
      {check.targets.some((t) => t.exists) && (
        <span className="muted">Старые прогоны на этих коллекциях станут несравнимыми с новым индексом по chunking и модели — их manifest это покажет.</span>
      )}
    </>
  );
}

function buildReport(check: CheckResult): ReactNode {
  const r = check.report as S["BuildCheck"] | null;
  if (!r || !("documents" in r)) return null;
  return (
    <>
      <span>
        <Chip tone="ok">dry-run пройден</Chip> {fmtDataset(r.dataset)} · {fmtNumber(r.documents)} документов · corpus sha256{" "}
        <span className="mono">{r.corpus_sha256.slice(0, 12)}</span>
      </span>
      <span className="muted">
        Цели:{" "}
        {r.targets.map((t, i) => (
          <span key={t.id}>
            {i > 0 && "; "}
            <span className="mono">{t.collection}</span> ({!t.reachable ? "недоступен" : t.exists ? `существует, ${fmtNumber(t.points)} points` : "новая"})
          </span>
        ))}
      </span>
    </>
  );
}
