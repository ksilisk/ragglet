import { columnVisibilityFeature, createColumnHelper, flexRender, rowSortingFeature, tableFeatures, useTable, type ColumnVisibilityState } from "@tanstack/react-table";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { useRunInclude, useRunQueries, type QueriesQuery } from "../../api/queries";
import { readFilters, writeFilters, type QueryFilters } from "../../analysis/queryParams";
import { STAGES, latencyGrid, misses, num, type Summary } from "../../analysis/summary";
import { useT } from "../../i18n/i18n";
import { fmtMetric, fmtMs, fmtNumber } from "../../lib/format";
import { Button } from "../../ui/Button";
import { Chip, FilterChip } from "../../ui/Chip";
import { cx } from "../../ui/cx";
import k from "../../ui/kit.module.css";
import { Empty } from "../../ui/Layout";
import { Menu } from "../../ui/Menu";
import { ErrorBox } from "../../ui/ProblemView";
import { QueryPanel } from "./QueryPanel";
import r from "./results.module.css";

const PAGE = 50;
type Row = Record<string, unknown>;
const features = tableFeatures({ columnVisibilityFeature, rowSortingFeature });
const helper = createColumnHelper<typeof features, Row>();

/** Колонки задержек стадий в порядке конвейера: id колонки — имя в per-query. */
const STAGE_COLUMNS = STAGES.map((stage) => ({ stage, id: `lat_${stage}_ms` }));

/**
 * «Запросы» по артборду RunQueries (ragglet-32 §7): таблица на тысячи строк — фильтры, сортировка и
 * страницы делает сервер, всё состояние в URL; клик по строке открывает панель запроса.
 */
export function QueriesTab({ run }: { run: S["RunStatus"] }) {
  const t = useT();
  const [params, setParams] = useSearchParams();
  const f = readFilters(params);
  const set = (patch: Partial<QueryFilters>) => setParams((prev) => writeFilters(prev, patch), { replace: true });
  const summary = useRunInclude(run.id, "summary", run.state === "completed").data?.summary ?? null;

  const query: QueriesQuery = {
    repeat: f.repeat,
    outcome: f.outcome ?? undefined,
    miss_at: f.miss_at ?? undefined,
    min_total_ms: f.min_total_ms ?? undefined,
    q: f.q || undefined,
    sort: f.sort ?? undefined,
    offset: f.offset,
    limit: PAGE,
  };
  const page = useRunQueries(run.id, query);
  const columns = useMemo(() => page.data?.columns ?? [], [page.data?.columns]);
  const ks = useMemo(() => [...new Set(columns.flatMap((c) => /^doc_ndcg@(\d+)$/.exec(c)?.[1] ?? []).map(Number))].sort((a, b) => a - b), [columns]);
  const kv = f.k && ks.includes(f.k) ? f.k : ks.includes(10) ? 10 : (ks[ks.length - 1] ?? 10);

  const counts = useMemo(() => repeatCounts(summary, f.repeat, kv), [summary, f.repeat, kv]);
  const [minMs, setMinMs] = useState(f.min_total_ms === null ? "" : String(f.min_total_ms));
  const [text, setText] = useState(f.q);
  // поиск — с задержкой: запрос к серверу не на каждую букву
  useEffect(() => {
    if (text === f.q) return;
    const timer = setTimeout(() => set({ q: text }), 300);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  if (page.error instanceof ApiProblem && page.error.status === 404 && page.error.has("run.no_per_query")) {
    return <Empty>{t("per-query у прогона нет: повтор ещё не закончился или run.save_per_query: false.")}</Empty>;
  }
  const filterError = page.error instanceof ApiProblem && page.error.has("request.filter");
  const data = page.data;
  const items = data?.items ?? [];
  const index = items.findIndex((row) => String(row.qid) === f.qid);
  const partial = run.state !== "completed" && data ? data.repeats : null;

  return (
    <>
      {partial && (
        <div className={k.problem + " " + k.warnBox}>
          <span>{t("Частичные результаты: записаны повторы {list}.", { list: partial.join(", ") })}</span>
        </div>
      )}
      <div className={r.toolbar}>
        <label htmlFor="queries-repeat" className="small muted">
          {t("повтор")}
        </label>
        <select id="queries-repeat" className={`${k.inp} ${k.sm} num`} style={{ width: "auto" }} value={f.repeat} onChange={(e) => set({ repeat: Number(e.target.value) })}>
          {(data?.repeats ?? [f.repeat]).map((x) => (
            <option key={x} value={x}>
              {x}
            </option>
          ))}
        </select>
        <FilterChip on={f.outcome === null} onClick={() => set({ outcome: null })}>
          {t("все исходы")}
        </FilterChip>
        {(["ok", "timeout", "error"] as const).map((outcome) => (
          <FilterChip key={outcome} on={f.outcome === outcome} onClick={() => set({ outcome: f.outcome === outcome ? null : outcome })}>
            {outcome}
            {counts && <span className={r.count}>{fmtNumber(counts[outcome])}</span>}
          </FilterChip>
        ))}
        {ks.length > 0 && (
          <FilterChip on={f.miss_at !== null} onClick={() => set({ miss_at: f.miss_at === null ? kv : null })}>
            {t("промах при k={k}", { k: f.miss_at ?? kv })}
            {counts?.misses !== null && counts?.misses !== undefined && f.miss_at === null && <span className={r.count}>{fmtNumber(counts.misses)}</span>}
          </FilterChip>
        )}
        <span className={r.toolbar} style={{ gap: 6 }}>
          <label htmlFor="queries-min" className="small muted">
            {t("total от")}
          </label>
          <input
            id="queries-min"
            className={`${k.inp} ${k.sm} num`}
            style={{ width: 80 }}
            inputMode="numeric"
            placeholder={t("мс")}
            value={minMs}
            onChange={(e) => setMinMs(e.target.value)}
            onBlur={() => set({ min_total_ms: minMs.trim() === "" ? null : Number(minMs) || null })}
            onKeyDown={(e) => e.key === "Enter" && set({ min_total_ms: minMs.trim() === "" ? null : Number(minMs) || null })}
          />
          <span className="small muted">{t("мс")}</span>
        </span>
        <input className={`${k.inp} ${k.sm}`} style={{ width: 240 }} placeholder={t("qid или текст запроса")} aria-label={t("Поиск по qid и тексту запроса")} value={text} onChange={(e) => setText(e.target.value)} />
        <span className={r.end + " small muted"}>{data ? t("{n} {queries}", { n: fmtNumber(data.total), queries: t.plural(data.total, "запрос|запроса|запросов") }) : ""}</span>
      </div>
      {filterError && <span className="small muted">{t("В этом формате артефактов нет колонки для фильтра: прогон старого формата.")}</span>}
      {page.error && !filterError && <ErrorBox error={page.error} />}
      <div className={cx(r.queries, f.qid && r.withPanel)}>
        <QueriesTable
          items={items}
          columns={columns}
          k={kv}
          ks={ks}
          sort={f.sort}
          selected={f.qid}
          stagesOn={summary ? latencyGrid(summary).stages : null}
          onSort={(sort) => set({ sort })}
          onK={(x) => set({ k: x })}
          onOpen={(qid) => set({ qid })}
          footer={
            data && data.total > 0 ? (
              <div className={k.tblFoot}>
                <span>
                  {t("{from}–{to} из {total}", { from: fmtNumber(f.offset + 1), to: fmtNumber(Math.min(f.offset + PAGE, data.total)), total: fmtNumber(data.total) })}
                </span>
                <span style={{ display: "flex", gap: 6 }}>
                  <Button small disabled={f.offset === 0} onClick={() => set({ offset: Math.max(0, f.offset - PAGE) })}>
                    {t("‹ назад")}
                  </Button>
                  <Button small disabled={f.offset + PAGE >= data.total} onClick={() => set({ offset: f.offset + PAGE })}>
                    {t("дальше ›")}
                  </Button>
                </span>
              </div>
            ) : data ? (
              <Empty>{t("Запросов с такими фильтрами нет.")}</Empty>
            ) : null
          }
        />
        {f.qid && (
          <QueryPanel
            runId={run.id}
            qid={f.qid}
            repeat={f.repeat}
            k={kv}
            onClose={() => set({ qid: null })}
            onNext={index >= 0 && index + 1 < items.length ? () => set({ qid: String(items[index + 1]!.qid) }) : null}
          />
        )}
      </div>
    </>
  );
}

/** Счётчики чипов по summary повтора: исходы и промахи при k. */
function repeatCounts(summary: Summary | null, repeat: number, kv: number) {
  const row = summary?.repeats[repeat];
  if (!summary || !row || num(row, "ok_count") === null) return null;
  return {
    ok: num(row, "ok_count") ?? 0,
    timeout: num(row, "timeout_count") ?? 0,
    error: num(row, "error_count") ?? 0,
    misses: misses(summary, kv)[repeat] ?? null,
  };
}

function QueriesTable({ items, columns, k: kv, ks, sort, selected, stagesOn, onSort, onK, onOpen, footer }: {
  items: Row[];
  columns: string[];
  k: number;
  ks: number[];
  sort: string | null;
  selected: string | null;
  stagesOn: string[] | null;
  onSort: (sort: string | null) => void;
  onK: (k: number) => void;
  onOpen: (qid: string) => void;
  footer: ReactNode;
}) {
  const t = useT();
  const has = (c: string) => columns.includes(c);
  const stageColumns = STAGE_COLUMNS.filter((s) => has(s.id));
  const [visibility, setVisibility] = useState<ColumnVisibilityState>({});
  // по умолчанию видны стадии, у которых в summary есть задержки
  const visible = (id: string, stage: string) => visibility[id] ?? (stagesOn ? stagesOn.includes(stage) : true);
  const columnVisibility = Object.fromEntries(stageColumns.map((s) => [s.id, visible(s.id, s.stage)]));

  const defs = useMemo(() => {
    const metric = (id: string, label: string) =>
      helper.display({
        id,
        header: label,
        cell: (c) => {
          const v = num(c.row.original, id);
          return <span className={v === 0 ? r.miss : v === 1 ? r.hit : undefined}>{fmtMetric(v)}</span>;
        },
      });
    return [
      helper.display({ id: "qid", header: "qid", cell: (c) => <span className="mono">{String(c.row.original.qid ?? "")}</span> }),
      helper.display({
        id: "query",
        header: () => t("Запрос"),
        cell: (c) => (
          <span className={r.queryText} title={String(c.row.original.query ?? "")}>
            {String(c.row.original.query ?? "")}
          </span>
        ),
      }),
      helper.display({ id: "outcome", header: () => t("Исход"), cell: (c) => <OutcomeChip row={c.row.original} /> }),
      helper.display({ id: "lat_total_ms", header: "total", cell: (c) => fmtMs(num(c.row.original, "lat_total_ms")) }),
      ...stageColumns.map((s) =>
        helper.display({
          id: s.id,
          header: s.stage,
          cell: (c) => {
            const v = num(c.row.original, s.id);
            return v === null ? <span className="muted">—</span> : fmtMs(v);
          },
        }),
      ),
      metric(`doc_ndcg@${kv}`, `nDCG@${kv}`),
      metric(`doc_recall@${kv}`, `Rec@${kv}`),
      helper.display({
        id: "first_relevant_rank",
        header: () => t("1-й рел."),
        cell: (c) => {
          const rank = num(c.row.original, "first_relevant_rank");
          if (rank === null) return <span className={r.miss}>—</span>;
          return <span className={rank > kv ? r.miss : undefined}>{rank > kv ? `> ${kv}` : rank}</span>;
        },
      }),
      helper.display({ id: "n_docs", header: "docs", cell: (c) => fmtNumber(num(c.row.original, "n_docs")) }),
    ].filter((d) => d.id === "query" || d.id === "outcome" || has(d.id!));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columns, kv, t]);

  const table = useTable<typeof features, Row>({
    features,
    columns: defs,
    data: items,
    manualSorting: true,
    state: { columnVisibility },
    onColumnVisibilityChange: (updater) => setVisibility((prev) => ({ ...prev, ...(typeof updater === "function" ? updater(columnVisibility) : updater) })),
    getRowId: (row) => String(row.qid),
  });

  const RIGHT = new Set(["lat_total_ms", "n_docs", "first_relevant_rank", `doc_ndcg@${kv}`, `doc_recall@${kv}`, ...stageColumns.map((s) => s.id)]);
  const sortable = (id: string) => id !== "query" && id !== "outcome" && has(id);
  const nextSort = (id: string) => (sort === `-${id}` ? id : sort === id ? null : `-${id}`);

  return (
    <section className={`${k.card} ${k.flush}`}>
      <div className={r.tableHead}>
        {ks.length > 1 && (
          <span className={r.toolbar} style={{ gap: 6 }}>
            <label htmlFor="queries-k" className="small muted">
              k
            </label>
            <select id="queries-k" className={`${k.inp} ${k.sm} num`} style={{ width: "auto" }} value={kv} onChange={(e) => onK(Number(e.target.value))}>
              {ks.map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
          </span>
        )}
        {stageColumns.length > 0 && (
          <Menu
            trigger={<Button small>{t("Колонки стадий ▾")}</Button>}
            items={stageColumns.map((s) => ({
              label: (
                <span>
                  {columnVisibility[s.id] ? "✓ " : " "}
                  {s.stage}
                </span>
              ),
              onSelect: () => setVisibility((prev) => ({ ...prev, [s.id]: !columnVisibility[s.id] })),
            }))}
          />
        )}
      </div>
      <div className={r.scroll}>
        <table className={`${k.tbl} ${k.compact} ${r.queryTable}`} aria-label={t("Запросы прогона")}>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((h) => {
                  const id = h.column.id;
                  const dir = sort === id ? "↑" : sort === `-${id}` ? "↓" : "";
                  return (
                    <th
                      key={h.id}
                      className={cx(RIGHT.has(id) && "r", sortable(id) && k.sortable)}
                      aria-sort={dir === "↑" ? "ascending" : dir === "↓" ? "descending" : undefined}
                      onClick={sortable(id) ? () => onSort(nextSort(id)) : undefined}
                    >
                      {flexRender(h.column.columnDef.header, h.getContext())}
                      {dir && ` ${dir}`}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className={cx(r.clickable, row.id === selected && k.sel)} onClick={() => onOpen(row.id)}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className={cx(RIGHT.has(cell.column.id) && "r num", cell.column.id === "query" && r.queryCell)}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {footer}
    </section>
  );
}

export function OutcomeChip({ row }: { row: Row }) {
  const outcome = typeof row.outcome === "string" ? row.outcome : null;
  if (outcome === null) return <span className="muted">—</span>;
  if (outcome === "ok") return <Chip tone="ok">ok</Chip>;
  const stage = typeof row.timeout_stage === "string" ? row.timeout_stage : null;
  return (
    <Chip tone="fail">
      {outcome}
      {stage && <span style={{ fontWeight: 400 }}>{stage}</span>}
    </Chip>
  );
}
