import { useState } from "react";
import type { S } from "../../api/client";
import { useQueryDetail } from "../../api/queries";
import { STAGES, num } from "../../analysis/summary";
import { useT } from "../../i18n/i18n";
import { fmtMs, fmtNumber } from "../../lib/format";
import { Button } from "../../ui/Button";
import { cx } from "../../ui/cx";
import { Card } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { OutcomeChip } from "./QueriesTab";
import r from "./results.module.css";

const FIRST = 10;

/**
 * Панель запроса (ragglet-32 §7, артборд RunQueries): текст, релевантные по qrels с оценками и
 * заголовками, финальный список с рангами, rerank-score и контекстом, текст чанка по кнопке. Чего нет,
 * сказано прямо: grades и texts говорят почему.
 */
export function QueryPanel({ runId, qid, repeat, k: kv, onClose, onNext }: {
  runId: string;
  qid: string;
  repeat: number;
  k: number;
  onClose: () => void;
  onNext: (() => void) | null;
}) {
  const t = useT();
  const detail = useQueryDetail(runId, qid, repeat);
  const d = detail.data;
  return (
    <Card
      className={r.panel}
      label={t("Запрос {qid}", { qid })}
      title={
        <h2>
          <span className="mono">{qid}</span>
        </h2>
      }
      aside={
        <span className={r.toolbar} style={{ gap: 8 }}>
          {d && <OutcomeChip row={d.row} />}
          {d && <span className="num small">{t("{ms} мс", { ms: fmtMs(num(d.row, "lat_total_ms")) })}</span>}
          <button type="button" className={r.close} aria-label={t("Закрыть панель")} onClick={onClose}>
            ×
          </button>
        </span>
      }
    >
      {detail.error && <ErrorBox error={detail.error} />}
      {!d && !detail.error && <span className="small muted">{t("Загрузка запроса…")}</span>}
      {d && <PanelBody d={d} k={kv} />}
      <div className={r.toolbar}>
        <Button small disabled={!onNext} onClick={() => onNext?.()}>
          {t("Следующий →")}
        </Button>
      </div>
    </Card>
  );
}

function PanelBody({ d, k: kv }: { d: S["QueryDetail"]; k: number }) {
  const t = useT();
  const [allTruth, setAllTruth] = useState(false);
  const [open, setOpen] = useState<Set<number>>(new Set());
  const row = d.row;
  const stages = [...STAGES, "total" as const].map((stage) => [stage, num(row, `lat_${stage}_ms`)] as const).filter(([, v]) => v !== null);
  // найденные — по рангу, ненайденные — по оценке
  const truth = [...d.truth].sort((a, b) => (a.doc_rank ?? Infinity) - (b.doc_rank ?? Infinity) || (b.grade ?? 0) - (a.grade ?? 0));
  const shown = allTruth ? truth : truth.slice(0, FIRST);
  const answer = typeof row.answer === "string" && row.answer ? row.answer : null;
  const context = typeof row.context_text === "string" && row.context_text ? row.context_text : null;
  const toggle = (rank: number) => setOpen((prev) => {
    const next = new Set(prev);
    if (next.has(rank)) next.delete(rank);
    else next.add(rank);
    return next;
  });

  return (
    <>
      <div className={r.queryQuote}>«{d.query}»</div>
      <div className={r.stageLine}>
        {stages.map(([stage, v]) => (
          <span key={stage}>
            {stage} <b className="num">{fmtMs(v)}</b>
          </span>
        ))}
        <span className="muted">{t("мс")}</span>
        {num(row, "t_start_ms") !== null && <span className="muted">{t("старт {t} с", { t: fmtNumber((num(row, "t_start_ms") ?? 0) / 1000, 2) })}</span>}
      </div>

      <section className={r.panelSection}>
        <div className={r.sectionHead}>
          {t("Релевантные по qrels")} <span className={r.count}>{fmtNumber(d.truth.length)}</span>
        </div>
        {shown.map((doc) => (
          <div key={doc.doc_id} className={r.docRow}>
            <Grade grade={doc.grade} relevant />
            <span className="mono">{doc.doc_id}</span>
            <span className={r.docTitle} title={doc.title ?? undefined}>
              {doc.title ?? ""}
            </span>
            <span className={cx("num small", (doc.doc_rank === null || doc.doc_rank > kv) && r.miss)}>
              {doc.doc_rank === null ? t("не найден") : doc.doc_rank > kv ? `> ${kv}` : t("ранг {rank}", { rank: doc.doc_rank })}
            </span>
          </div>
        ))}
        {truth.length > FIRST && (
          <button type="button" className={r.more} onClick={() => setAllTruth((x) => !x)}>
            {allTruth ? t("свернуть") : t("ещё {n}", { n: truth.length - FIRST })}
          </button>
        )}
      </section>

      <section className={r.panelSection}>
        <div className={r.sectionHead}>
          {t("Финальный список")} <span className={r.count}>{fmtNumber(d.results.length)}</span>
        </div>
        {d.results.length === 0 && <span className="small muted">{t("Список пуст: запрос без ответа — промах, или в per-query этого формата нет final_ids.")}</span>}
        {d.results.map((chunk) => (
          <div key={chunk.rank}>
            <div className={r.resultRow}>
              <span className="num small muted">{chunk.rank}</span>
              <Grade grade={chunk.grade} relevant={chunk.relevant} />
              <span className="mono" title={chunk.chunk_id}>
                {chunk.doc_id}
              </span>
              <span className={r.docTitle} title={chunk.title ?? undefined}>
                {chunk.title ?? ""}
              </span>
              <span className="num small" title={t("rerank-score")}>
                {chunk.rerank_score === null ? "" : fmtNumber(chunk.rerank_score, 2)}
              </span>
              <span className="small muted">{chunk.in_context === true ? t("в контексте") : ""}</span>
              {chunk.text ? (
                <button type="button" className={r.more} aria-expanded={open.has(chunk.rank)} onClick={() => toggle(chunk.rank)}>
                  {open.has(chunk.rank) ? t("скрыть") : t("текст")}
                </button>
              ) : (
                <span />
              )}
            </div>
            {open.has(chunk.rank) && chunk.text && <div className={r.chunkText}>{chunk.text}</div>}
          </div>
        ))}
      </section>

      {answer && <Collapsible title={t("Ответ генерации")} text={answer} />}
      {context && <Collapsible title={t("Текст контекста")} text={context} />}

      <div className={r.caption}>
        {!d.grades.available && <div>{t("Оценки и заголовки недоступны: {why}", { why: d.grades.message ?? d.grades.code ?? "—" })}</div>}
        {!d.texts.available && <div>{t("Тексты чанков недоступны: {why}", { why: d.texts.message ?? d.texts.code ?? "—" })}</div>}
        <div>{t("Score retrieval и источник (vector / keyword) в per-query не сохраняются.")}</div>
      </div>
    </>
  );
}

/** Оценка из qrels плашкой; без оценок — только «релевантен / нет». */
function Grade({ grade, relevant }: { grade: number | null | undefined; relevant: boolean }) {
  const t = useT();
  const on = grade !== null && grade !== undefined ? grade > 0 : relevant;
  return (
    <span className={on ? r.rel : r.norel} title={on ? t("релевантен") : t("не релевантен")}>
      {grade !== null && grade !== undefined ? grade : on ? "+" : ""}
    </span>
  );
}

function Collapsible({ title, text }: { title: string; text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <section className={r.panelSection}>
      <button type="button" className={r.expander} aria-expanded={open} onClick={() => setOpen((x) => !x)}>
        <span className={r.caret} aria-hidden>
          {open ? "▾" : "▸"}
        </span>
        <span className={r.sectionHead}>{title}</span>
      </button>
      {open && <div className={r.chunkText}>{text}</div>}
    </section>
  );
}
