import { useState } from "react";
import type { S } from "../../api/client";
import { useRunInclude } from "../../api/queries";
import { parseYaml } from "../../config/yamlDoc";
import { fmtMetric, fmtMs, fmtNumber, fmtPercent, fmtQps, plural } from "../../lib/format";
import k from "../../ui/kit.module.css";
import { Card, Empty } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import s from "../task/task.module.css";

type Row = Record<string, unknown>;

const num = (row: Row | undefined, key: string): number | null => {
  const v = row?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
};

/** Колонка агрегата: среднее и разброс по повторам (summary.aggregate.json) или значение единственного повтора. */
export function metric(summary: S["RunSummaryOut"], key: string): { mean: number | null; std: number | null } {
  const agg = summary.aggregate;
  const mean = num(agg, `${key}_mean_over_repeats`) ?? num(summary.repeats[0], key);
  return { mean, std: num(agg, `${key}_std_over_repeats`) };
}

export function availableK(summary: S["RunSummaryOut"]): number[] {
  const keys = [...Object.keys(summary.aggregate), ...Object.keys(summary.repeats[0] ?? {})];
  const ks = new Set<number>();
  for (const key of keys) {
    const m = /^doc_ndcg@(\d+)_mean/.exec(key);
    if (m) ks.add(Number(m[1]));
  }
  return [...ks].sort((a, b) => a - b);
}

function Tile({ label, value, sub, muted, small }: { label: string; value: string; sub?: string; muted?: boolean; small?: boolean }) {
  return (
    <div className={s.tile}>
      <span className={s.tileL}>{label}</span>
      <span className={`${s.tileV}${muted ? " muted" : ""}`} style={small ? { fontSize: 18 } : undefined} title={value}>
        {value}
      </span>
      {sub && <span className={s.tileS}>{sub}</span>}
    </div>
  );
}

/**
 * «Обзор» в объёме 31 (решение 1 ревью): верхний ряд плиток и таблица «Повторы» артборда RunOverview,
 * без графиков. Остальное на этом артборде — ragglet-32.
 */
export function RunOverview({ run }: { run: S["RunStatus"] }) {
  const query = useRunInclude(run.id, "summary");
  const snapshot = useRunInclude(run.id, "scenario");
  const summary = query.data?.summary ?? null;
  const ks = summary ? availableK(summary) : [];
  const [kSel, setK] = useState<number | null>(null);
  if (query.error) return <ErrorBox error={query.error} />;
  if (query.isPending) return <span className="muted">Загрузка summary…</span>;
  if (!summary) return <Empty>У прогона нет summary: ядро пишет его в конце успешного прогона.</Empty>;

  const kv = kSel ?? (ks.includes(10) ? 10 : (ks[ks.length - 1] ?? 10));
  const repeats = summary.repeats.length || Number(summary.aggregate.repeats ?? 1);
  const pm = (std: number | null, fmt: (x: number) => string) => (repeats > 1 && std !== null ? `± ${fmt(std)}` : null);
  const sub = (...parts: Array<string | null | false>) => parts.filter(Boolean).join(" · ");
  const repeatsLabel = `${repeats} ${plural(repeats, "повтор", "повтора", "повторов")}`;
  const m = (key: string) => metric(summary, key);
  const ndcg = m(`doc_ndcg@${kv}_mean`);
  const recall = m(`doc_recall@${kv}_mean`);
  const mrr = m(`doc_mrr@${kv}_mean`);
  const rcap = m(`doc_r_cap@${kv}_mean`);
  const hit = m(`doc_hit@${kv}_mean`);
  const p50 = m("lat_total_p50");
  const p95 = m("lat_total_p95");
  const p99 = m("lat_total_p99");
  const qps = m("throughput_qps");
  const wall = m("wall_ms");
  const sum = (key: string) => summary.repeats.reduce((acc, row) => acc + (num(row, key) ?? 0), 0);
  // доля попаданий есть в summary и при выключенном кэше (0), поэтому слои берутся из snapshot
  const cache = (parseYaml(snapshot.data?.scenario_yaml ?? "").data?.cache ?? {}) as { enabled?: boolean; embedding?: unknown; retrieval?: unknown };
  const emb = cache.enabled && cache.embedding ? m("emb_hit_rate").mean : null;
  const ret = cache.enabled && cache.retrieval ? m("ret_hit_rate").mean : null;
  const concurrency = run.plan?.load?.concurrency;

  return (
    <>
      {ks.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <label htmlFor="overview-k" className="small muted">
            k
          </label>
          <select id="overview-k" className={`${k.inp} ${k.sm} num`} style={{ width: "auto" }} value={kv} onChange={(e) => setK(Number(e.target.value))}>
            {ks.map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
          <span className="small muted">из metrics.quality.k · уровень doc</span>
        </div>
      )}
      <div className={s.tiles}>
        {ndcg.mean !== null && <Tile label={`nDCG@${kv}`} value={fmtMetric(ndcg.mean)} sub={sub(pm(ndcg.std, fmtMetric), repeatsLabel)} />}
        {recall.mean !== null && (
          <Tile label={`Recall@${kv}`} value={fmtMetric(recall.mean)} sub={sub(pm(recall.std, fmtMetric), rcap.mean !== null && `R_cap@${kv} ${fmtNumber(rcap.mean, 2)}`)} />
        )}
        {mrr.mean !== null && <Tile label={`MRR@${kv}`} value={fmtNumber(mrr.mean, 2)} sub={hit.mean !== null ? `Hit@${kv} ${fmtNumber(hit.mean, 2)}` : undefined} />}
        <Tile label="p50 total" value={fmtMs(p50.mean)} sub={sub("мс", p99.mean !== null && `p99 ${fmtMs(p99.mean)}`, pm(p50.std, fmtMs))} />
        <Tile label="p95 total" value={fmtMs(p95.mean)} sub={sub("мс", pm(p95.std, fmtMs))} />
        <Tile
          label="QPS"
          value={fmtQps(qps.mean)}
          sub={sub(wall.mean !== null && `wall ${fmtNumber(wall.mean / 1000, 1)} с`, concurrency !== undefined && `×${String(concurrency)}`, pm(qps.std, fmtQps))}
        />
        <Tile label="Исходы" value={`${fmtNumber(sum("ok_count"))} · ${fmtNumber(sum("timeout_count"))} · ${fmtNumber(sum("error_count"))}`} sub="ok · timeout · error" small />
        {emb === null && ret === null ? (
          <Tile label="Кэш" value="—" sub="выключен" muted />
        ) : (
          <Tile label="Кэш" value={fmtPercent(emb ?? ret)} sub={[emb !== null ? `emb ${fmtPercent(emb)}` : null, ret !== null ? `ret ${fmtPercent(ret)}` : null].filter(Boolean).join(" · ")} />
        )}
      </div>
      <Card title="Повторы">
        <table className={`${k.tbl} ${k.compact}`} aria-label="Повторы">
          <thead>
            <tr>
              <th>#</th>
              <th className="r">n</th>
              <th className="r">ok</th>
              <th className="r">timeout</th>
              <th className="r">error</th>
              <th className="r">wall, с</th>
              <th className="r">QPS</th>
              <th className="r">p50</th>
              <th className="r">p95</th>
              <th className="r">nDCG@{kv}</th>
            </tr>
          </thead>
          <tbody>
            {summary.repeats.map((row, i) => (
              <tr key={i}>
                <td className="num">{String(row.repeat_idx ?? i)}</td>
                <td className="r num">{fmtNumber(num(row, "n"))}</td>
                <td className="r num">{fmtNumber(num(row, "ok_count"))}</td>
                <td className="r num">{fmtNumber(num(row, "timeout_count"))}</td>
                <td className="r num">{fmtNumber(num(row, "error_count"))}</td>
                <td className="r num">{fmtNumber((num(row, "wall_ms") ?? Number.NaN) / 1000, 1)}</td>
                <td className="r num">{fmtQps(num(row, "throughput_qps"))}</td>
                <td className="r num">{fmtMs(num(row, "lat_total_p50"))}</td>
                <td className="r num">{fmtMs(num(row, "lat_total_p95"))}</td>
                <td className="r num">{fmtMetric(num(row, `doc_ndcg@${kv}_mean`))}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {repeats === 1 && <span className="small muted">Один повтор: std по повторам не считается. Для оценки стабильности задержек нужны repeats ≥ 3.</span>}
      </Card>
      <span className="small muted">Графики, качество по k, задержки по стадиям и запросы — ragglet-32.</span>
    </>
  );
}
