import { useScenarioMeta, useStand } from "../../api/queries";
import { finalDepth } from "../../config/hints";
import { enumAt } from "../../config/schema";
import { useEditorActions, useValue } from "../../editor/derived";
import s from "../../editor/editor.module.css";
import {
  CheckField,
  CheckSetField,
  Grid,
  Note,
  NumberCsvField,
  NumberField,
  NumberListField,
  SelectField,
  TextField,
} from "../../editor/fields";
import { Section } from "../../editor/Section";
import { useEditor, useEditorContext } from "../../editor/store";
import { rich, useT } from "../../i18n/i18n";
import { Chip } from "../../ui/Chip";

function useToggle(path: string) {
  const { edit } = useEditorActions();
  const on = useValue<boolean>(path) ?? false;
  return { on, set: (v: boolean) => edit(path.split("."), v) };
}

export function RerankSection() {
  const t = useT();
  const meta = useScenarioMeta().data!;
  const toggle = useToggle("rerank.enabled");
  const provider = useValue<string>("rerank.provider") ?? "cross_encoder";
  const model = useValue<string>("rerank.model");
  const inputTopK = useValue<number>("rerank.input_top_k") ?? 30;
  const outputTopN = useValue<number>("rerank.output_top_n") ?? 10;
  return (
    <Section
      id="rerank"
      title="Rerank"
      toggle={toggle}
      summary={toggle.on ? `${provider}${model ? ` · ${model}` : ""} · ${inputTopK} → ${outputTopN}` : t("выключен")}
    >
      <Grid cols={4}>
        <SelectField path="rerank.provider" label="provider" options={enumAt(meta.json_schema, ["rerank", "provider"])} />
        <TextField path="rerank.model" label="model" mono span={2} placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2" />
        <TextField path="rerank.revision" label="revision" mono placeholder="main" />
      </Grid>
      <Grid cols={4}>
        <NumberField path="rerank.input_top_k" label="input_top_k" min={1} />
        <NumberField path="rerank.output_top_n" label="output_top_n" min={1} />
      </Grid>
      {!toggle.on && <Note>{t("Стадия выключена: значения сохранятся в YAML и вернутся при включении.")}</Note>}
    </Section>
  );
}

export function ContextSection() {
  const t = useT();
  const meta = useScenarioMeta().data!;
  const toggle = useToggle("context.enabled");
  const generation = useValue<boolean>("generation.enabled") ?? false;
  const policy = useValue<string>("context.policy") ?? "top_n";
  const maxTokens = useValue<number>("context.max_tokens") ?? 1200;
  return (
    <Section
      id="context"
      title="Context"
      toggle={{ ...toggle, disabled: toggle.on && generation }}
      summary={toggle.on ? `${policy} · max_tokens ${maxTokens}` : t("выключен")}
    >
      <Grid cols={4}>
        <SelectField path="context.policy" label="policy" options={enumAt(meta.json_schema, ["context", "policy"])} />
        <NumberField path="context.max_tokens" label={t("max_tokens — оценка по словам")} min={1} />
        {(policy === "threshold" || policy === "adaptive") && <NumberField path="context.threshold" label="threshold" integer={false} />}
        <div style={{ alignSelf: "end" }}>
          <CheckField path="context.include_metadata" label="include_metadata" />
        </div>
      </Grid>
      {toggle.on && generation && <Note>{t("Context нужен generation: выключить его можно после generation.")}</Note>}
    </Section>
  );
}

export function GenerationSection() {
  const t = useT();
  const meta = useScenarioMeta().data!;
  const { edit } = useEditorActions();
  const on = useValue<boolean>("generation.enabled") ?? false;
  const context = useValue<boolean>("context.enabled") ?? false;
  const provider = useValue<string>("generation.provider") ?? "noop";
  const model = useValue<string>("generation.model");
  const keyEnv = useValue<string>("generation.api_key_env") ?? "OPENAI_API_KEY";
  const stand = useStand();
  const keySet = stand.data?.env[keyEnv];
  const setOn = (v: boolean) => {
    edit(["generation", "enabled"], v);
    if (v && !context) edit(["context", "enabled"], true); // generation требует context (валидатор схемы)
  };
  return (
    <Section id="generation" title="Generation" toggle={{ on, set: setOn }} summary={on ? `${provider}${model ? ` · ${model}` : ""}` : t("выключена")}>
      {!on && !context && <Note>{t("Включение generation включит и Context: без него схема не пропустит сценарий.")}</Note>}
      <Grid cols={4}>
        <SelectField path="generation.provider" label="provider" options={enumAt(meta.json_schema, ["generation", "provider"])} />
        <TextField path="generation.model" label="model" mono span={2} />
        <TextField path="generation.base_url" label="base_url" mono />
      </Grid>
      <Grid cols={4}>
        <TextField
          path="generation.api_key_env"
          label="api_key_env"
          mono
          extra={keySet !== undefined && <span><Chip tone={keySet ? "ok" : "warn"}>{keySet ? t("задана на сервере") : t("не задана на сервере")}</Chip></span>}
        />
        <NumberField path="generation.temperature" label="temperature" integer={false} min={0} />
        <NumberField path="generation.max_output_tokens" label="max_output_tokens" min={1} />
      </Grid>
      <div className={s.chkRow}>
        <CheckField path="generation.save.prompt" label="save.prompt" />
        <CheckField path="generation.save.answer" label="save.answer" />
        <CheckField path="generation.save.raw_response" label="save.raw_response" />
      </div>
      <Note>{t("Ключа в сценарии нет: api_key_env — имя переменной окружения сервера.")}</Note>
    </Section>
  );
}

export function CacheSection() {
  const t = useT();
  const { edit, remove } = useEditorActions();
  const stand = useStand();
  const toggle = useToggle("cache.enabled");
  const embedding = useValue<Record<string, unknown> | null>("cache.embedding");
  const retrieval = useValue<Record<string, unknown> | null>("cache.retrieval");
  const layers = [embedding ? "embedding (redis)" : null, retrieval ? "retrieval (memory_lru)" : null].filter(Boolean);
  return (
    <Section id="cache" title={t("Кэш")} toggle={toggle} summary={toggle.on ? layers.join(" · ") || t("нет слоёв") : t("выключен")}>
      <div className={s.itemCard}>
        <label className={s.chk}>
          <input
            type="checkbox"
            checked={Boolean(embedding)}
            onChange={(e) =>
              e.target.checked
                ? edit(["cache", "embedding"], { type: "redis", endpoint: stand.data?.redis?.url ?? "redis://localhost:6379/0" })
                : remove(["cache", "embedding"])
            }
          />
          {rich(t("{name} — Redis, общий между прогонами"), { name: <b>embedding</b> })}
        </label>
        {embedding && (
          <>
            <Grid cols={4}>
              <TextField path="cache.embedding.endpoint" label="endpoint" mono span={2} optional={false} />
              <NumberField path="cache.embedding.ttl_seconds" label="ttl_seconds" min={1} />
              <TextField path="cache.embedding.namespace" label="namespace" mono placeholder={t("свой у каждого прогона")} />
            </Grid>
            <Note>{t("Пароль в URL Redis попадёт в snapshot прогона. Общий namespace не чистится между повторами — только при run.repeats = 1.")}</Note>
          </>
        )}
      </div>
      <div className={s.itemCard}>
        <label className={s.chk}>
          <input
            type="checkbox"
            checked={Boolean(retrieval)}
            onChange={(e) => (e.target.checked ? edit(["cache", "retrieval"], { type: "memory_lru", size: 10000 }) : remove(["cache", "retrieval"]))}
          />
          {rich(t("{name} — LRU в памяти процесса"), { name: <b>retrieval</b> })}
        </label>
        {retrieval && (
          <Grid cols={4}>
            <NumberField path="cache.retrieval.size" label="size" min={1} />
          </Grid>
        )}
        <Note>{t("Кэш живёт в процессе прогона: попадания возможны только при run.repeats > 1.")}</Note>
      </div>
    </Section>
  );
}

export function LoadSection() {
  const t = useT();
  const meta = useScenarioMeta().data!;
  const repeats = useValue<number>("run.repeats") ?? 1;
  const loadType = useValue<string>("run.load.type") ?? "closed";
  const concurrency = useValue<number>("run.load.concurrency") ?? 1;
  const shuffle = useValue<boolean>("run.shuffle") ?? true;
  const seed = useValue<number>("seed") ?? 42;
  const perQuery = useValue<boolean>("run.save_per_query") ?? true;
  const params = (meta.load_models[loadType]?.properties ?? {}) as Record<string, { minimum?: number }>;
  const repeatsWord = repeats === 1 ? t("повтор") : t("повт.");
  const shuffleWord = shuffle ? t("вкл") : t("выкл");
  const perQueryWord = perQuery ? t("сохраняется") : t("не сохраняется");
  return (
    <Section
      id="load"
      title={t("Нагрузка и повторы")}
      summary={`${repeats} ${repeatsWord} · ${loadType} × ${concurrency} · shuffle ${shuffleWord} · seed ${seed} · per-query ${perQueryWord}`}
    >
      <Grid cols={4}>
        <NumberField path="run.repeats" label="run.repeats" min={1} />
        <NumberField path="seed" label="seed" />
        <SelectField path="run.load.type" label="run.load.type" options={Object.keys(meta.load_models)} />
        {"concurrency" in params && <NumberField path="run.load.concurrency" label="run.load.concurrency" min={params.concurrency?.minimum ?? 1} />}
      </Grid>
      <div className={s.chkRow}>
        <CheckField path="run.shuffle" label={t("run.shuffle — перемешать запросы")} />
        <CheckField path="run.save_per_query" label={t("run.save_per_query — сохранять per-query")} />
      </div>
      <Note>{t("run.artifacts_dir в форме нет: каталог прогона назначает сервер.")}</Note>
    </Section>
  );
}

export function TimeoutsSection() {
  const t = useT();
  const rerank = useValue<boolean>("rerank.enabled") ?? false;
  const generation = useValue<boolean>("generation.enabled") ?? false;
  const total = useValue<number>("timeouts.total_ms") ?? 6000;
  const retrieval = useValue<number>("timeouts.retrieval_ms") ?? 1500;
  const rerankMs = useValue<number>("timeouts.rerank_ms") ?? 800;
  const llm = useValue<number>("timeouts.llm_ms") ?? 3500;
  const ms = t("мс");
  const parts = [`total ${total}`, `retrieval ${retrieval}`, rerank ? `rerank ${rerankMs}` : null, generation ? `llm ${llm}` : null].filter(Boolean);
  return (
    <Section id="timeouts" title={t("Таймауты")} summary={`${parts.join(" · ")} ${ms}`}>
      <Grid cols={4}>
        <NumberField path="timeouts.total_ms" label="total_ms" min={0} />
        <NumberField path="timeouts.retrieval_ms" label="retrieval_ms" min={0} />
        {rerank && <NumberField path="timeouts.rerank_ms" label="rerank_ms" min={0} />}
        {generation && <NumberField path="timeouts.llm_ms" label="llm_ms" min={0} />}
      </Grid>
    </Section>
  );
}

export function MetricsSection() {
  const t = useT();
  const meta = useScenarioMeta().data!;
  const { defaults } = useEditorContext();
  const data = useEditor((st) => st.parsed.data);
  const { depth } = finalDepth(data, defaults);
  const ks = useValue<number[]>("metrics.quality.k") ?? [];
  const doc = useValue<string[]>("metrics.quality.doc");
  const format = useValue<string>("output.format") ?? "parquet";
  const enabled = {
    rerank: useValue<boolean>("rerank.enabled") ?? false,
    context: useValue<boolean>("context.enabled") ?? false,
    generation: useValue<boolean>("generation.enabled") ?? false,
  };
  const offStages = new Set(Object.entries(enabled).filter(([, on]) => !on).map(([name]) => name));
  return (
    <Section
      id="metrics"
      title={t("Метрики и вывод")}
      summary={`k ${(ks.length ? ks : [1, 3, 5, 10]).join(", ")} · doc ${(doc ?? ["ndcg", "recall", "hit", "mrr"]).join(", ")} · ${format}`}
    >
      <NumberListField path="metrics.quality.k" label="metrics.quality.k" bad={(n) => (n > depth ? "error" : null)} />
      <Grid cols={2}>
        <CheckSetField path="metrics.quality.doc" label="quality.doc" options={meta.metrics.doc ?? []} />
        <CheckSetField path="metrics.quality.chunk" label="quality.chunk" options={meta.metrics.chunk ?? []} />
      </Grid>
      <div className={s.grid} style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2fr)" }}>
        <NumberCsvField path="metrics.percentiles" label="percentiles" />
        <CheckSetField path="metrics.latency_steps" label="latency_steps" options={meta.latency_steps} disabled={offStages} />
      </div>
      <div className={s.grid} style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2fr)", alignItems: "end" }}>
        <SelectField path="output.format" label="output.format" options={enumAt(meta.json_schema, ["output", "format"])} />
        <CheckField path="output.include_debug_fields" label={t("Сохранять текст контекста в per-query (output.include_debug_fields)")} />
      </div>
    </Section>
  );
}
