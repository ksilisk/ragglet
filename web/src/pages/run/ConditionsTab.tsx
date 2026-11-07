import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router";
import type { S } from "../../api/client";
import { useRunInclude, useRuns } from "../../api/queries";
import { derivedWarnings, envDiffers, manifestFacts, short, type EnvFacts, type ManifestFacts } from "../../analysis/manifest";
import { snapshotFacts, type SnapshotFacts } from "../../analysis/snapshot";
import { num } from "../../analysis/summary";
import { useT, type T } from "../../i18n/i18n";
import { fmtBytes, fmtNumber, fmtWhen, splitDataset } from "../../lib/format";
import { Chip } from "../../ui/Chip";
import { Card, Empty, KeyValue } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { Segmented } from "../../ui/Segmented";
import { Tip } from "../../ui/Tip";
import r from "./results.module.css";

/** Хэш коротко, полный — в подсказке; ревизия BeIR составная: «<data>+<qrels>». */
function Hash({ value }: { value: string | null }) {
  if (!value) return <span className="muted">—</span>;
  const text = value.split("+").map((part) => short(part)).join("+");
  return (
    <Tip content={<span className="mono">{value}</span>}>
      <span className="mono" tabIndex={0}>
        {text}
      </span>
    </Tip>
  );
}

const or = (v: ReactNode | null | undefined) => (v === null || v === undefined || v === "" ? <span className="muted">—</span> : v);

export interface ConditionsData {
  facts: ManifestFacts | null;
  snap: SnapshotFacts;
  manifest: Record<string, unknown> | null;
}

/** manifest и snapshot прогона — один раз на страницу (кэш TanStack Query общий с другими вкладками). */
export function useConditions(run: S["RunStatus"]): { data: ConditionsData | null; error: unknown } {
  const manifest = useRunInclude(run.id, "manifest", run.state !== "queued");
  const scenario = useRunInclude(run.id, "scenario");
  const data = useMemo(
    () =>
      manifest.data && scenario.data
        ? { facts: manifestFacts(manifest.data.manifest), snap: snapshotFacts(scenario.data.scenario_yaml), manifest: manifest.data.manifest ?? null }
        : null,
    [manifest.data, scenario.data],
  );
  return { data, error: manifest.error ?? scenario.error };
}

function envLine(t: T, env: EnvFacts) {
  return [
    env.commit && `commit ${short(env.commit)}`,
    env.python && `Python ${env.python}`,
    env.device,
    env.cpu,
  ].filter(Boolean).join(", ") || t("нет данных");
}

/**
 * «Условия» (ragglet-32 §8): manifest структурно — датасет, индекс со сборкой, rerank, окружение,
 * сервисы, нагрузка, предупреждения — и переключатель «JSON».
 */
export function ConditionsTab({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const { data, error } = useConditions(run);
  const [view, setView] = useState<"cards" | "json">("cards");
  if (error) return <ErrorBox error={error} />;
  if (!data) return <span className="muted">{t("Загрузка manifest…")}</span>;
  const { facts, snap } = data;

  return (
    <>
      <div className={r.toolbar}>
        <Segmented
          label={t("Вид")}
          value={view}
          onChange={setView}
          items={[
            { id: "cards", label: t("структурно") },
            { id: "json", label: "JSON" },
          ]}
        />
        <PreviousRunLink run={run} />
      </div>
      {!facts && <Empty>{t("manifest нет: прогон старого формата (до ragglet-17). Ниже — то, что есть в записи реестра и snapshot.")}</Empty>}
      {view === "json" ? (
        <Card
          title="manifest.json"
          aside={
            data.manifest && (
              <a className="small" href={`/v1/runs/${encodeURIComponent(run.id)}/artifacts/manifest.json`} download>
                {t("Скачать")}
              </a>
            )
          }
        >
          <pre className={r.json}>{data.manifest ? JSON.stringify(data.manifest, null, 2) : "null"}</pre>
        </Card>
      ) : (
        <div className={r.grid2}>
          <DatasetCard run={run} facts={facts} snap={snap} />
          <IndexCard run={run} facts={facts} snap={snap} />
          <EnvCard title={t("Окружение прогона")} env={facts?.env ?? null} differs={new Set()} />
          <EnvCard title={t("Окружение сборки индекса")} env={facts?.buildEnv ?? null} differs={envDiffers(facts?.buildEnv ?? null, facts?.env ?? null)} />
          <ServicesCard facts={facts} snap={snap} />
          <LoadCard run={run} snap={snap} />
          <WarningsCard run={run} facts={facts} />
        </div>
      )}
    </>
  );
}

function DatasetCard({ run, facts, snap }: { run: S["RunStatus"]; facts: ManifestFacts | null; snap: SnapshotFacts }) {
  const t = useT();
  const ds = splitDataset(run.dataset);
  // у импортированных прогонов плана нет: запросов на повтор — из summary
  const summary = useRunInclude(run.id, "summary", !run.plan && run.state === "completed").data?.summary;
  const queries = run.plan?.queries_per_repeat ?? num(summary?.repeats[0], "n");
  return (
    <Card title={t("Датасет")}>
      <KeyValue
        items={[
          [t("Источник"), <span className="mono">{ds.source ? `${ds.source}:${ds.id}` : ds.id}</span>],
          [t("Ревизия"), <Hash value={facts?.dataset?.revision ?? ds.revision} />],
          ["corpus_sha256", <Hash value={facts?.dataset?.corpus ?? null} />],
          ["eval_sha256", <Hash value={facts?.dataset?.eval ?? null} />],
          ["split", or(snap.datasetSplit)],
          ["limit", or(snap.datasetLimit !== null ? fmtNumber(snap.datasetLimit) : null)],
          [t("Запросов"), or(queries !== null ? fmtNumber(queries) : null)],
        ]}
      />
    </Card>
  );
}

function IndexCard({ run, facts, snap }: { run: S["RunStatus"]; facts: ManifestFacts | null; snap: SnapshotFacts }) {
  const t = useT();
  const ix = facts?.index ?? null;
  return (
    <Card title={t("Индекс")}>
      <KeyValue
        items={[
          [t("Коллекция"), <span className="mono">{run.collection}</span>],
          [t("Профиль"), or(ix?.profileId && <span className="mono">{ix.profileId}</span>)],
          [t("Собран"), or(ix?.builtAt && fmtWhen(ix.builtAt))],
          [
            t("Модель"),
            or(
              snap.embeddingModel && (
                <>
                  {snap.embeddingModel} {ix?.embeddingRevision && <Hash value={ix.embeddingRevision} />}
                </>
              ),
            ),
          ],
          ["chunking", or(ix?.chunking && [ix.chunking.method, ix.chunking.size !== null && `${ix.chunking.size}/${ix.chunking.overlap ?? 0}`].filter(Boolean).join(" "))],
          [t("Векторы"), or(ix && [ix.distance, ix.vectorSize !== null && String(ix.vectorSize)].filter(Boolean).join(" "))],
          ["HNSW", or(ix?.hnsw && `m ${ix.hnsw.m ?? "—"}, ef_construct ${ix.hnsw.efConstruct ?? "—"}`)],
          ["Rerank", snap.rerankModel ? <>{snap.rerankModel} {facts?.rerank?.revision && <Hash value={facts.rerank.revision} />}</> : <span className="muted">{t("выключен")}</span>],
        ]}
      />
    </Card>
  );
}

function EnvCard({ title, env, differs }: { title: string; env: EnvFacts | null; differs: Set<keyof EnvFacts> }) {
  const t = useT();
  const mark = (key: keyof EnvFacts, value: ReactNode) =>
    differs.has(key) ? (
      <>
        {value} <Chip tone="warn">{t("≠ прогону")}</Chip>
      </>
    ) : (
      value
    );
  if (!env) {
    return (
      <Card title={title}>
        <span className="small muted">{t("Нет в manifest этого прогона.")}</span>
      </Card>
    );
  }
  return (
    <Card title={title}>
      <KeyValue
        items={[
          ["commit", mark("commit", or(env.commit && <Hash value={env.commit} />))],
          ["Python", mark("python", or(env.python))],
          [t("Система"), mark("platform", or(env.platform && <span className="mono small">{env.platform}</span>))],
          ["CPU", mark("cpu", or(env.cpu && `${env.cpu}${env.cpus ? `, ${t("ядер: {n}", { n: env.cpus })}` : ""}`))],
          ["RAM", or(env.ram !== null ? fmtBytes(env.ram) : null)],
          [t("Устройство"), mark("device", or(env.device && `${env.device}${env.accelerator ? ` (${env.accelerator})` : ""}`))],
          [t("Потоки torch"), mark("threads", or(env.threads))],
        ]}
      />
    </Card>
  );
}

function ServicesCard({ facts, snap }: { facts: ManifestFacts | null; snap: SnapshotFacts }) {
  const t = useT();
  const services = facts?.services ?? [];
  return (
    <Card title={t("Сервисы")}>
      {services.length === 0 ? (
        <span className="small muted">{t("Нет в manifest этого прогона.")}</span>
      ) : (
        <KeyValue
          items={services.map((s) => [
            <span className="mono">
              {s.role}.{s.name}
            </span>,
            <span>
              {[s.type, s.version].filter(Boolean).join(" ")}
              {s.status && (
                <>
                  {" "}
                  <Chip tone={s.status === "green" ? "ok" : "warn"}>{s.status}</Chip>
                </>
              )}
              {s.points !== null && <span className="muted"> {t("points {n}", { n: fmtNumber(s.points) })}</span>}
              {s.indexed !== null && <span className="muted">, {t("indexed {n}", { n: fmtNumber(s.indexed) })}</span>}
              {s.segments !== null && <span className="muted">, {t("segments {n}", { n: s.segments })}</span>}
            </span>,
          ])}
        />
      )}
      {snap.cache.enabled && <span className={r.caption}>{t("Кэш включён: {layers}.", { layers: [snap.cache.emb && "embedding (Redis)", snap.cache.ret && "retrieval (LRU)"].filter(Boolean).join(", ") })}</span>}
    </Card>
  );
}

function LoadCard({ run, snap }: { run: S["RunStatus"]; snap: SnapshotFacts }) {
  const t = useT();
  const concurrency = typeof run.plan?.load?.concurrency === "number" ? run.plan.load.concurrency : snap.load.concurrency;
  return (
    <Card title={t("Нагрузка и повторы")}>
      <KeyValue
        items={[
          [t("Модель"), or(snap.load.type)],
          ["concurrency", or(concurrency)],
          [t("Повторы"), or(run.plan?.repeats ?? snap.repeats)],
          ["shuffle", or(snap.shuffle === null ? null : snap.shuffle ? t("да") : t("нет"))],
          ["seed", or(snap.seed)],
          [t("Бюджет total"), or(snap.totalBudgetMs !== null && t("{ms} мс", { ms: fmtNumber(snap.totalBudgetMs) }))],
        ]}
      />
    </Card>
  );
}

function WarningsCard({ run, facts }: { run: S["RunStatus"]; facts: ManifestFacts | null }) {
  const t = useT();
  const derived = derivedWarnings(facts);
  return (
    <Card title={t("Предупреждения")}>
      {run.warnings.length === 0 && derived.length === 0 ? (
        <span className="small muted">{t("Нет.")}</span>
      ) : (
        <div className={r.warnList}>
          {run.warnings.map((w, i) => (
            <div key={`p${i}`}>
              <Chip tone="warn">{w.code}</Chip> <span className="small muted">{w.message}</span>
            </div>
          ))}
          {derived.map((w, i) => (
            <div key={`d${i}`}>
              <Chip tone="warn">{w.code}</Chip> <span className="small muted">{t(w.text, w.vars)}</span>
            </div>
          ))}
        </div>
      )}
      <span className={r.caption}>{t("Preflight при постановке и то, что видно по manifest на старте прогона.")}</span>
    </Card>
  );
}

/** «Сравнить с предыдущим прогоном сценария»: тот же сценарий, поставлен раньше этого. */
function PreviousRunLink({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const runs = useRuns({ scenario: run.scenario, until: run.enqueued_at, limit: 2, sort: "-enqueued_at" }, false);
  const previous = (runs.data?.items ?? []).find((x) => x.id !== run.id);
  if (!previous) return null;
  return (
    <Link className={`${r.end} small`} to={`/compare?ids=${encodeURIComponent(previous.id)},${encodeURIComponent(run.id)}`}>
      {t("Сравнить с предыдущим прогоном сценария →")}
    </Link>
  );
}

/** Карточка «Условия» в «Обзоре» (артборд RunOverview): кратко, «Подробнее →» ведёт на вкладку. */
export function ConditionsSummary({ run, summaryN }: { run: S["RunStatus"]; summaryN: number | null }) {
  const t = useT();
  const { data } = useConditions(run);
  if (!data) return <Card title={t("Условия")}>{<span className="small muted">{t("Загрузка manifest…")}</span>}</Card>;
  const { facts, snap } = data;
  const ds = splitDataset(run.dataset);
  const ix = facts?.index;
  const storage = facts?.services.find((s) => s.role === "storage");
  const warnings = [...run.warnings.map((w) => w.code), ...derivedWarnings(facts).map((w) => w.code)];
  const concurrency = typeof run.plan?.load?.concurrency === "number" ? run.plan.load.concurrency : snap.load.concurrency;
  const repeats = run.plan?.repeats ?? snap.repeats ?? 1;
  return (
    <Card title={t("Условия")} aside={<Link className="small" to={{ search: "?tab=conditions" }}>{t("Подробнее →")}</Link>}>
      <KeyValue
        items={[
          [
            t("Датасет"),
            <span>
              {ds.source ? `${ds.source}:${ds.id}` : ds.id} <Hash value={facts?.dataset?.revision ?? ds.revision} />
              {snap.datasetSplit && <span className="muted"> split {snap.datasetSplit}</span>}
              {summaryN !== null && <span className="muted">, {t("{n} запросов", { n: fmtNumber(summaryN) })}</span>}
            </span>,
          ],
          [
            t("Индекс"),
            <span>
              <span className="mono">{run.collection}</span>
              {snap.embeddingModel && <>, {snap.embeddingModel}</>}
              {ix?.chunking && <>, {[ix.chunking.method, ix.chunking.size !== null && `${ix.chunking.size}/${ix.chunking.overlap ?? 0}`].filter(Boolean).join(" ")}</>}
              {ix?.hnsw && <>, HNSW {ix.hnsw.m}/{ix.hnsw.efConstruct}</>}
            </span>,
          ],
          ["Rerank", snap.rerankModel ? <span>{snap.rerankModel} {facts?.rerank?.revision && <Hash value={facts.rerank.revision} />}</span> : <span className="muted">{t("выключен")}</span>],
          [t("Окружение"), facts?.env ? <span>{envLine(t, facts.env)}{storage?.version && `, Qdrant ${storage.version}`}</span> : <span className="muted">—</span>],
          [t("Нагрузка"), <span>{[snap.load.type, concurrency !== null && `×${concurrency}`, String(repeats) + " " + t.plural(repeats, "повтор|повтора|повторов")].filter(Boolean).join(", ")}</span>],
          [
            t("Предупреждения"),
            warnings.length ? (
              <span className={r.chips}>
                {warnings.map((code, i) => (
                  <Chip key={i} tone="warn">
                    {code}
                  </Chip>
                ))}
              </span>
            ) : (
              <span className="muted">{t("нет")}</span>
            ),
          ],
        ]}
      />
    </Card>
  );
}
