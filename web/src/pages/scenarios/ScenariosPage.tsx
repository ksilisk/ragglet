import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { useScenarios, useStand, useTasks } from "../../api/queries";
import { compactNames } from "../../lib/names";
import { fmtDataset, fmtMetric, fmtNumber, fmtWhen } from "../../lib/format";
import { Button, LinkButton } from "../../ui/Button";
import { Chip, PipeChip } from "../../ui/Chip";
import { useConfirm } from "../../ui/Dialog";
import { Empty, Page, PageHead } from "../../ui/Layout";
import { Menu } from "../../ui/Menu";
import { ErrorBox } from "../../ui/ProblemView";
import { StateChip } from "../../ui/states";
import k from "../../ui/kit.module.css";
import { ImportDialog, NameDialog } from "../library/LibraryDialogs";
import { createDocument, deleteDocument, duplicate, exportYaml, invalidateLibrary, skeleton } from "../library/library";
import s from "./scenarios.module.css";

export function pipelineChips(p: S["Pipeline"]): Array<{ label: string; on: boolean }> {
  const chips = [p.strategy === "fanout_merge" ? { label: `fanout ×${p.sources}`, on: true } : { label: p.strategy, on: false }];
  if (p.rerank) chips.push({ label: "rerank", on: true });
  if (p.context) chips.push({ label: "context", on: true });
  if (p.generation) chips.push({ label: "generation", on: true });
  if (p.cache) chips.push({ label: "cache", on: true });
  if (p.concurrency && p.concurrency > 1) chips.push({ label: `×${p.concurrency}`, on: false });
  if (p.repeats > 1) chips.push({ label: `×${p.repeats} повт.`, on: false });
  return chips;
}

type Dialogs = { kind: "new" } | { kind: "import" } | { kind: "duplicate"; id: string } | null;

export default function ScenariosPage() {
  const scenarios = useScenarios();
  const stand = useStand();
  const [params, setParams] = useSearchParams();
  const [dialog, setDialog] = useState<Dialogs>(null);
  const q = params.get("q") ?? "";
  const dataset = params.get("dataset") ?? "";

  const items = useMemo(() => scenarios.data ?? [], [scenarios.data]);
  const datasets = useMemo(() => [...new Set(items.map((i) => i.dataset).filter((d): d is string => Boolean(d)))].sort(), [items]);
  const ids = useMemo(() => new Set(items.map((i) => i.id)), [items]);
  const needle = q.trim().toLowerCase();
  const shown = items.filter(
    (i) =>
      (!dataset || i.dataset === dataset) &&
      (!needle || [i.id, i.description ?? "", ...i.tags].some((v) => v.toLowerCase().includes(needle))),
  );

  const setParam = (key: string, value: string) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (value) next.set(key, value);
        else next.delete(key);
        return next;
      },
      { replace: true },
    );

  return (
    <Page>
      <PageHead
        title="Сценарии"
        count={scenarios.data ? items.length : undefined}
        actions={
          <>
            <input
              className={`${k.inp} ${s.search}`}
              placeholder="поиск по имени и описанию"
              aria-label="Поиск сценариев"
              value={q}
              onChange={(e) => setParam("q", e.target.value)}
            />
            <select className={`${k.inp} ${s.filter}`} aria-label="Фильтр по датасету" value={dataset} onChange={(e) => setParam("dataset", e.target.value)}>
              <option value="">все датасеты</option>
              {datasets.map((d) => (
                <option key={d} value={d}>
                  {fmtDataset(d)}
                </option>
              ))}
            </select>
            <Menu
              trigger={<Button variant="primary">Новый сценарий ▾</Button>}
              items={[
                { label: "Пустой…", onSelect: () => setDialog({ kind: "new" }) },
                { label: "Из YAML…", onSelect: () => setDialog({ kind: "import" }) },
              ]}
            />
          </>
        }
      />
      {scenarios.error && <ErrorBox error={scenarios.error} />}
      <section className={`${k.card} ${k.flush}`}>
        <table className={`${k.tbl} ${s.tight}`} aria-label="Библиотека сценариев">
          <thead>
            <tr>
              <th>Имя</th>
              <th>Описание</th>
              <th>Датасет</th>
              <th>Коллекция</th>
              <th>Конвейер</th>
              <th>Последний прогон</th>
              <th>Изменён</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {shown.map((item) => (
              <ScenarioRow key={item.id} item={item} onDuplicate={() => setDialog({ kind: "duplicate", id: item.id })} />
            ))}
          </tbody>
        </table>
        {scenarios.data && shown.length === 0 && <Empty>{items.length ? "Ничего не нашлось." : "Библиотека пуста: создайте сценарий."}</Empty>}
      </section>
      <span className="small muted">
        Библиотека — каталог <span className="mono">configs/scenarios/</span> на сервере: то же, что видит CLI. «Запустить выбранные» ставит
        сценарии в очередь по порядку, каждый проходит dry-run.
      </span>

      {dialog?.kind === "new" && (
        <NameDialog
          kind="scenario"
          open
          onClose={() => setDialog(null)}
          title="Новый сценарий"
          initial="new_scenario"
          confirm="Создать"
          taken={ids}
          onSubmit={(name) => createDocument("scenario", skeleton("scenario", name, stand.data?.qdrant[0]?.url))}
        />
      )}
      {dialog?.kind === "duplicate" && (
        <NameDialog
          kind="scenario"
          open
          onClose={() => setDialog(null)}
          title={`Дублировать ${dialog.id}`}
          initial={`${dialog.id}_copy`}
          confirm="Дублировать"
          taken={ids}
          onSubmit={(name) => duplicate("scenario", dialog.id, name)}
        />
      )}
      <ImportDialog kind="scenario" open={dialog?.kind === "import"} onClose={() => setDialog(null)} />
    </Page>
  );
}

function ScenarioRow({ item, onDuplicate }: { item: S["ScenarioItem"]; onDuplicate: () => void }) {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const navigate = useNavigate();
  const href = `/scenarios/${encodeURIComponent(item.id)}`;

  const onDelete = async () => {
    const ok = await confirm({
      title: `Удалить сценарий ${item.id}?`,
      body: (
        <>
          Файл <span className="mono">configs/scenarios/{item.id}.yaml</span> будет удалён. Прогоны останутся: у каждого есть свой snapshot
          сценария.
        </>
      ),
      confirm: "Удалить",
      danger: true,
    });
    if (!ok) return;
    try {
      await deleteDocument("scenario", item.id, item.etag);
    } catch (e) {
      if (e instanceof ApiProblem && e.status === 409) {
        await confirm({ title: "Файл изменился", body: "Сценарий изменили вне редактора — обновите список и повторите.", confirm: "Понятно" });
      } else throw e;
    } finally {
      invalidateLibrary(qc, "scenario");
    }
  };

  return (
    <tr>
      <td>
        <Link className={`${k.rowLink} mono`} to={href}>
          {item.id}
        </Link>
      </td>
      <td className={`muted ${s.desc}`}>
        {item.valid ? (
          <span className={s.clamp} title={item.description ?? undefined}>
            {item.description}
          </span>
        ) : (
          <Chip tone="warn" title="Файл не проходит схему: откройте его в редакторе">черновик</Chip>
        )}
      </td>
      <td>{item.dataset ? fmtDataset(item.dataset) : <span className="muted">—</span>}</td>
      <td className="mono">{compactNames(item.collections.map((c) => c.collection)) || <span className="muted">—</span>}</td>
      <td>
        <div className={s.chips}>{item.pipeline && pipelineChips(item.pipeline).map((c) => <PipeChip key={c.label} on={c.on}>{c.label}</PipeChip>)}</div>
      </td>
      <td className="nowrap">
        <LastRun run={item.last_run ?? null} />
      </td>
      <td className="muted nowrap">{fmtWhen(item.updated_at)}</td>
      <td>
        <div className={s.acts}>
          <LinkButton small to={href}>
            Открыть
          </LinkButton>
          <Button small onClick={onDuplicate}>
            Дублировать
          </Button>
          <Menu
            trigger={
              <Button small aria-label={`Ещё: ${item.id}`}>
                ⋯
              </Button>
            }
            items={[
              { label: "Открыть", onSelect: () => navigate(href) },
              { label: "Экспорт YAML", onSelect: () => void exportYaml("scenario", item.id) },
              { label: "Удалить…", onSelect: () => void onDelete(), danger: true },
            ]}
          />
        </div>
      </td>
    </tr>
  );
}

function LastRun({ run }: { run: S["RunBrief"] | null }) {
  const tasks = useTasks();
  if (!run) return <span className="muted">—</span>;
  const href = `/runs/${encodeURIComponent(run.id)}`;
  const live = tasks.data?.find((t) => t.kind === "run" && t.id === run.id);
  let detail;
  if (run.state === "running" && live?.progress) {
    detail = `${fmtNumber(live.progress.done)}/${fmtNumber(live.progress.total)}`;
  } else if (run.state === "queued" && live?.position) {
    detail = `#${live.position}`;
  } else if (run.state === "completed" && run.ndcg_at_10 !== null && run.ndcg_at_10 !== undefined) {
    detail = `${fmtWhen(run.enqueued_at)} · nDCG@10 ${fmtMetric(run.ndcg_at_10)}`;
  } else {
    detail = fmtWhen(run.enqueued_at);
  }
  return (
    <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
      <StateChip state={run.state} />
      <Link className="small" to={href}>
        {detail}
      </Link>
    </span>
  );
}
