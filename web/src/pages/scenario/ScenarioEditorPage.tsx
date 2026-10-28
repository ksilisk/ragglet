import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { useLocation, useNavigate, useParams } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { keys, useIndexes, useScenarioMeta } from "../../api/queries";
import { scenarioHints } from "../../config/hints";
import { bindIssues } from "../../config/registry";
import { SCENARIO_REGISTRY, SCENARIO_SECTIONS } from "../../config/scenarioFields";
import { parseYaml } from "../../config/yamlDoc";
import { useValue } from "../../editor/derived";
import { useT } from "../../i18n/i18n";
import s from "../../editor/editor.module.css";
import { EditorShell } from "../../editor/EditorShell";
import { FieldMessages, TagsField } from "../../editor/fields";
import { dryRun } from "../../editor/commands";
import { launch } from "../../editor/launch";
import { EditorContext, checkFromProblem, createEditorStore, useEditor, useEditorContext, type CheckResult } from "../../editor/store";
import { useEditorPage, useLibraryDocument } from "../../editor/useEditorPage";
import { fmtDataset, fmtNumber } from "../../lib/format";
import { Button } from "../../ui/Button";
import { Chip } from "../../ui/Chip";
import { cx } from "../../ui/cx";
import k from "../../ui/kit.module.css";
import { Crumbs, Empty, Page } from "../../ui/Layout";
import { Menu } from "../../ui/Menu";
import { useConfirm } from "../../ui/Dialog";
import { ErrorBox } from "../../ui/ProblemView";
import { WarningsBody, enqueueScenario } from "../runs/enqueue";
import { DataSection } from "./DataSection";
import { RetrievalSection } from "./RetrievalSection";
import { CacheSection, ContextSection, GenerationSection, LoadSection, MetricsSection, RerankSection, TimeoutsSection } from "./StageSections";

export default function ScenarioEditorPage() {
  const t = useT();
  const { id = "" } = useParams();
  const doc = useLibraryDocument("scenario", id);
  const meta = useScenarioMeta();
  if (doc.error || meta.error) {
    const notFound = doc.error instanceof ApiProblem && doc.error.status === 404;
    return (
      <Page>
        <Crumbs items={[{ label: t("Сценарии"), to: "/scenarios" }, { label: id }]} />
        {notFound ? <Empty>{t("Сценария {id} нет в библиотеке.", { id })}</Empty> : <ErrorBox error={doc.error ?? meta.error} />}
      </Page>
    );
  }
  if (!doc.data || !meta.data) {
    return (
      <Page>
        <Crumbs items={[{ label: t("Сценарии"), to: "/scenarios" }, { label: id }]} />
        <span className="muted">{t("Загрузка…")}</span>
      </Page>
    );
  }
  return <ScenarioEditor key={id} doc={doc.data} meta={meta.data} />;
}

/** Секции открыты, если стадия включена или в черновике есть ошибки; нагрузка и таймауты свёрнуты. */
function initialOpen(doc: S["ConfigDocument"]): Record<string, boolean> {
  const data = parseYaml(doc.yaml).data as Record<string, Record<string, unknown> | undefined> | null;
  const on = (section: string) => Boolean(data?.[section]?.enabled);
  const open: Record<string, boolean> = {
    data: true,
    retrieval: true,
    rerank: on("rerank"),
    context: on("context"),
    generation: on("generation"),
    cache: on("cache"),
    load: false,
    timeouts: false,
    metrics: true,
  };
  for (const issue of bindIssues(doc.problem?.issues ?? [], SCENARIO_REGISTRY)) open[issue.section] = true;
  return open;
}

function ScenarioEditor({ doc, meta }: { doc: S["ConfigDocument"]; meta: S["ScenarioMeta"] }) {
  const t = useT();
  const [store] = useState(() => createEditorStore("scenario", doc, initialOpen(doc)));
  const server = useLibraryDocument("scenario", doc.id).data;
  const indexes = useIndexes();
  const page = useEditorPage(store, "scenario", server);

  // Подсказки знают выбранную коллекцию: датасет и модель индекса против сценария.
  const hints = useCallback(
    (data: Record<string, unknown> | null) => {
      const storage = (data?.storage ?? {}) as { backend?: { endpoint?: string; collection?: string } };
      const norm = (e: unknown) => String(e ?? "").toLowerCase().replace(/\/$/, "");
      const collection = indexes.data?.collections.find((c) => norm(c.endpoint) === norm(storage.backend?.endpoint) && c.name === storage.backend?.collection);
      return scenarioHints(data, meta.defaults, collection?.ragglet ?? null);
    },
    [indexes.data, meta.defaults],
  );
  const ctx = useMemo(() => ({ store, registry: SCENARIO_REGISTRY, defaults: meta.defaults, hints }), [store, meta.defaults, hints]);

  const qc = useQueryClient();
  const navigate = useNavigate();
  const confirm = useConfirm();
  const location = useLocation();

  // «Открыть в редакторе» со страницы упавшего прогона: проблемы второго preflight — у полей.
  useEffect(() => {
    const problem = (location.state as { problem?: S["Problem"] } | null)?.problem;
    if (!problem) return;
    const check = checkFromProblem("task", problem);
    store.getState().setCheck(check);
    openErrorSections(check);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- один раз при открытии
  }, []);

  const onCheck = async () => {
    const out = await page.onCheck();
    if (out?.kind === "problems") openErrorSections(out.check);
  };

  // Запустить = сохранить → dry-run → подтвердить предупреждения → 202 → страница прогона.
  const onLaunch = () =>
    page.run("launch", async () => {
      const result = await launch<S["RunCheck"]>({
        save: page.doSave,
        check: async () => {
          const out = await dryRun(store);
          if (out.kind === "problems") openErrorSections(out.check);
          return out.kind === "ok" ? { kind: "ok", warnings: out.check.issues, report: out.check.report as S["RunCheck"] } : { kind: out.kind };
        },
        confirm: (warnings, check) =>
          confirm({ title: t("Запустить всё равно?"), body: <WarningsBody warnings={warnings} check={check} />, confirm: t("Запустить"), wide: true }),
        submit: async () => {
          const st = store.getState();
          const out = await enqueueScenario(st.checkedText ?? st.text, st.id);
          if (out.ok) return { kind: "accepted", id: out.accepted.id, position: out.accepted.position ?? null };
          const check = checkFromProblem("launch", out.problem);
          st.setCheck(check, st.checkedText ?? st.text);
          openErrorSections(check);
          return { kind: "problems" };
        },
      });
      if (result.kind === "started") {
        void qc.invalidateQueries({ queryKey: keys.tasks });
        void qc.invalidateQueries({ queryKey: keys.runs });
        void qc.invalidateQueries({ queryKey: keys.scenarios });
        page.leaving.current = true;
        navigate(`/runs/${encodeURIComponent(result.id)}`);
      }
      return result;
    });
  const openErrorSections = (check: CheckResult) => {
    for (const issue of bindIssues(check.issues, SCENARIO_REGISTRY)) if (issue.severity === "error") store.getState().setOpen(issue.section, true);
  };

  const actions = (
    <>
      <Button onClick={() => void page.onSave()} disabled={!page.dirty || page.broken || page.busy !== null} title={page.broken ? t("YAML не разбирается") : undefined}>
        {page.busy === "save" ? t("Сохранение…") : t("Сохранить")}
      </Button>
      <Button onClick={() => void onCheck()} disabled={page.broken || page.busy !== null} title={page.broken ? t("YAML не разбирается") : undefined}>
        {page.busy === "check" ? t("Проверка…") : t("Проверить")}
      </Button>
      <Button variant="primary" onClick={() => void onLaunch()} disabled={page.broken || page.busy !== null} title={page.broken ? t("YAML не разбирается") : t("Сохранить, проверить и поставить в очередь")}>
        {page.busy === "launch" ? t("Запуск…") : t("Запустить")}
      </Button>
      <Menu
        trigger={<Button aria-label={t("Ещё действия")}>⋯</Button>}
        items={[
          { label: t("Сохранить как…"), onSelect: page.onSaveAs },
          { label: t("Сбросить к сохранённому"), onSelect: () => void page.onReset(), disabled: !page.dirty },
          { label: t("Экспорт YAML"), onSelect: page.onExport },
          { label: t("Удалить…"), onSelect: () => void page.onDelete(), danger: true },
        ]}
      />
    </>
  );

  return (
    <EditorContext.Provider value={ctx}>
      <Page>
        <EditorShell head={<ScenarioHead actions={actions} note={page.note} banners={page.banners} />} sections={SCENARIO_SECTIONS} report={runReport}>
          <DataSection />
          <RetrievalSection />
          <RerankSection />
          <ContextSection />
          <GenerationSection />
          <CacheSection />
          <LoadSection />
          <TimeoutsSection />
          <MetricsSection />
        </EditorShell>
        {page.dialogs}
      </Page>
    </EditorContext.Provider>
  );
}

function ScenarioHead({ actions, note, banners }: { actions: ReactNode; note: string | null; banners: ReactNode }) {
  const t = useT();
  const id = useEditor((st) => st.id);
  const dirty = useEditor((st) => st.text !== st.saved.text);
  const valid = useEditor((st) => st.saved.valid);
  const name = useValue<string>("name");
  const description = useValue<string>("description");
  const readOnly = useEditor((st) => st.parsed.error !== null);
  const { store } = useEditorContext();
  return (
    <>
      <Crumbs items={[{ label: t("Сценарии"), to: "/scenarios" }, { label: id }]} />
      <div className={s.head}>
        <div className={s.headFields}>
          <div className={s.headRow}>
            <label htmlFor="f-name-input" className={s.headLabel}>
              name
            </label>
            <input
              id="f-name-input"
              className={cx(k.inp, k.ro, s.nameInput)}
              value={name ?? ""}
              readOnly
              title={t("Имя — это id файла: переименование — «Сохранить как…» под новым именем")}
            />
            {dirty && <Chip tone="warn">{t("не сохранено")}</Chip>}
            {!valid && <Chip tone="fail">{t("черновик")}</Chip>}
          </div>
          <div id="f-name">
            <FieldMessages path="name" />
          </div>
          <div className={s.headRow} id="f-description">
            <label htmlFor="f-description-input" className={s.headLabel}>
              description
            </label>
            <input
              id="f-description-input"
              className={cx(k.inp, s.descInput)}
              value={description ?? ""}
              readOnly={readOnly}
              placeholder={t("что проверяет этот сценарий")}
              onChange={(e) => store.getState().edit(["description"], e.target.value)}
              onBlur={(e) => e.target.value === "" && description !== undefined && store.getState().remove(["description"])}
            />
          </div>
          <div className={s.headRow}>
            <span className={s.headLabel}>tags</span>
            <TagsField path="tags" />
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


/** Отчёт preflight под панелью «Проверка» после успешного dry-run. */
function runReport(check: CheckResult): ReactNode {
  const r = check.report as S["RunCheck"] | null;
  if (!r || !("plan" in r)) return null;
  return <RunReport r={r} />;
}

function RunReport({ r }: { r: S["RunCheck"] }) {
  const t = useT();
  const load = Object.entries(r.plan.load).map(([key, v]) => (key === "type" ? String(v) : `${key} ${String(v)}`)).join(" · ");
  return (
    <>
      <span>
        <Chip tone="ok">{t("dry-run пройден")}</Chip> {fmtDataset(r.dataset.label)} <span className="mono muted">{r.dataset.revision?.slice(0, 12)}</span> · split{" "}
        {r.dataset.split} ·{" "}
        {t("{queries} запросов × {repeats} = {total} · нагрузка {load}", {
          queries: fmtNumber(r.dataset.queries),
          repeats: r.plan.repeats,
          total: fmtNumber(r.plan.total),
          load,
        })}
      </span>
      <span className="muted">
        {t("Цели:")}{" "}
        {r.targets.map((t, i) => (
          <span key={t.id}>
            {i > 0 && "; "}
            <span className="mono">{t.collection}</span> ({t.status}, {fmtNumber(t.points)} points)
          </span>
        ))}
      </span>
    </>
  );
}
