import { useScenarioMeta } from "../../api/queries";
import { fieldDomId } from "../../config/registry";
import { enumAt } from "../../config/schema";
import { useEditorActions, useValue } from "../../editor/derived";
import s from "../../editor/editor.module.css";
import { CheckField, FieldMessages, Grid, Note, NumberField, SelectField, TextField } from "../../editor/fields";
import { Section } from "../../editor/Section";
import { Button } from "../../ui/Button";
import { cx } from "../../ui/cx";

type Source = { name?: string; kind?: string; weight?: number; params?: Record<string, unknown> };

export function sourceSummary(src: Source): string {
  const kind = src.kind ?? "vector";
  const params = Object.entries(src.params ?? {}).map(([k, v]) => `${k} ${String(v)}`);
  return `${src.name ?? "?"} (${[kind, ...params].join(", ")})`;
}

export function RetrievalSection() {
  const meta = useScenarioMeta().data!;
  const { append, removeAt } = useEditorActions();
  const strategy = useValue<string>("retrieval.strategy") ?? "single";
  const topK = useValue<number>("retrieval.top_k") ?? 30;
  const asyncOn = useValue<boolean>("retrieval.async") ?? true;
  const sources = useValue<Source[]>("retrieval.sources") ?? [];
  const method = useValue<string>("retrieval.merge.method") ?? "rrf";
  const rrfK = useValue<number>("retrieval.merge.rrf_k") ?? 60;
  const mode = useValue<string>("storage.mode") ?? "single";
  const backends = useValue<Array<{ id?: string }>>("storage.backends") ?? [];
  const fanout = strategy === "fanout_merge";
  const shown = fanout ? sources : sources.slice(0, 1);
  const targetIds = backends.map((b) => b.id).filter((x): x is string => Boolean(x));

  const summary = [
    strategy,
    `top_k ${topK}`,
    fanout ? `async ${asyncOn ? "вкл" : "выкл"}` : null,
    sources.length
      ? `${sources.length} ${sources.length === 1 ? "источник" : "источника"}: ${sources.map(sourceSummary).join("; ")}`
      : "источник по умолчанию",
    fanout ? `${method}${method === "rrf" ? ` k=${rrfK}` : ""}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Section id="retrieval" title="Retrieval" summary={summary}>
      <Grid cols={4}>
        <SelectField
          path="retrieval.strategy"
          label="strategy"
          options={meta.retrieval_strategies.map((v) => ({ value: v, label: v === "fanout_merge" ? "fanout_merge — несколько источников" : v }))}
        />
        <NumberField path="retrieval.top_k" label="top_k" min={1} />
        {fanout ? <CheckField path="retrieval.async" label="async — источники параллельно" /> : <div />}
      </Grid>

      <div className={s.cards} id={fieldDomId("retrieval.sources")}>
        <h3 className={s.subhead}>Источники</h3>
        <FieldMessages path="retrieval.sources" />
        {sources.length === 0 && <Note>Источников нет: single использует один vector-источник по умолчанию.</Note>}
        {shown.map((src, i) => {
          const base = `retrieval.sources[${i}]`;
          const kind = src.kind ?? "vector";
          return (
            <div key={i} className={s.itemCard} id={fieldDomId(base)}>
              <div className={s.itemHead}>
                <span className="mono grow">{src.name || `источник ${i + 1}`}</span>
                <Button small onClick={() => removeAt(["retrieval", "sources"], i)}>
                  Убрать
                </Button>
              </div>
              <FieldMessages path={base} />
              <Grid cols={4}>
                <TextField path={`${base}.name`} label="name" mono optional={false} />
                <SelectField path={`${base}.kind`} label="kind" options={enumAt(meta.json_schema, ["retrieval", "sources", 0, "kind"])} />
                <NumberField path={`${base}.weight`} label="weight" integer={false} min={0} />
              </Grid>
              <FieldMessages path={`${base}.params`} />
              <Grid cols={4}>
                {kind === "vector" ? (
                  <>
                    <NumberField path={`${base}.params.ef_search`} label="params.ef_search" min={1} placeholder="по умолчанию Qdrant" />
                    {mode !== "single" && (
                      <SelectField path={`${base}.params.backend_ref`} label="params.backend_ref" options={targetIds} mono />
                    )}
                  </>
                ) : (
                  <>
                    <NumberField path={`${base}.params.k1`} label="params.k1" integer={false} min={0} placeholder="1.5" />
                    <NumberField path={`${base}.params.b`} label="params.b" integer={false} min={0} placeholder="0.75" />
                  </>
                )}
              </Grid>
            </div>
          );
        })}
        {!fanout && sources.length > 1 && <Note>Ещё {sources.length - 1} в YAML: single берёт только первый, остальные не показаны.</Note>}
        {(fanout || sources.length === 0) && (
          <div>
            <Button small onClick={() => append(["retrieval", "sources"], { name: `source_${sources.length + 1}`, kind: "vector", weight: 1.0 })}>
              + источник
            </Button>
          </div>
        )}
      </div>

      {fanout && (
        <div className={cx(s.cards)}>
          <h3 className={s.subhead}>Слияние (merge)</h3>
          <Grid cols={4}>
            <SelectField path="retrieval.merge.method" label="method" options={enumAt(meta.json_schema, ["retrieval", "merge", "method"])} />
            {method === "rrf" && <NumberField path="retrieval.merge.rrf_k" label="rrf_k" min={1} />}
            <CheckField path="retrieval.merge.deduplicate" label="deduplicate — влияет только на interleave" />
          </Grid>
        </div>
      )}
    </Section>
  );
}
