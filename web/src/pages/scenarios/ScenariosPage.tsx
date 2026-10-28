import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { useScenarios, useStand, useTasks } from "../../api/queries";
import { rich, translate, useT } from "../../i18n/i18n";
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
import { BatchLaunch } from "./BatchLaunch";
import s from "./scenarios.module.css";

export function pipelineChips(p: S["Pipeline"]): Array<{ label: string; on: boolean }> {
  const chips = [p.strategy === "fanout_merge" ? { label: `fanout ×${p.sources}`, on: true } : { label: p.strategy, on: false }];
  if (p.rerank) chips.push({ label: "rerank", on: true });
  if (p.context) chips.push({ label: "context", on: true });
  if (p.generation) chips.push({ label: "generation", on: true });
  if (p.cache) chips.push({ label: "cache", on: true });
  if (p.concurrency && p.concurrency > 1) chips.push({ label: `×${p.concurrency}`, on: false });
  if (p.repeats > 1) chips.push({ label: translate("×{n} повт.", { n: p.repeats }), on: false });
  return chips;
}

type Dialogs = { kind: "new" } | { kind: "import" } | { kind: "duplicate"; id: string } | { kind: "launch"; ids: string[] } | null;

export default function ScenariosPage() {
  const t = useT();
  const scenarios = useScenarios();
  const stand = useStand();
  const [params, setParams] = useSearchParams();
  const [dialog, setDialog] = useState<Dialogs>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
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
        title={t("Сценарии")}
        count={scenarios.data ? items.length : undefined}
        actions={
          <>
            <input
              className={`${k.inp} ${s.search}`}
              placeholder={t("поиск по имени и описанию")}
              aria-label={t("Поиск сценариев")}
              value={q}
              onChange={(e) => setParam("q", e.target.value)}
            />
            <select className={`${k.inp} ${s.filter}`} aria-label={t("Фильтр по датасету")} value={dataset} onChange={(e) => setParam("dataset", e.target.value)}>
              <option value="">{t("все датасеты")}</option>
              {datasets.map((d) => (
                <option key={d} value={d}>
                  {fmtDataset(d)}
                </option>
              ))}
            </select>
            <Button
              disabled={selected.size === 0}
              onClick={() => setDialog({ kind: "launch", ids: shown.map((i) => i.id).filter((id) => selected.has(id)) })}
            >
              {t("Запустить выбранные · {n}", { n: selected.size })}
            </Button>
            <Menu
              trigger={<Button variant="primary">{t("Новый сценарий ▾")}</Button>}
              items={[
                { label: t("Пустой…"), onSelect: () => setDialog({ kind: "new" }) },
                { label: t("Из YAML…"), onSelect: () => setDialog({ kind: "import" }) },
              ]}
            />
          </>
        }
      />
      {scenarios.error && <ErrorBox error={scenarios.error} />}
      <section className={`${k.card} ${k.flush}`}>
        <table className={`${k.tbl} ${s.tight}`} aria-label={t("Библиотека сценариев")}>
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  aria-label={t("Выбрать все")}
                  checked={shown.length > 0 && shown.every((i) => selected.has(i.id))}
                  onChange={(e) => setSelected(e.target.checked ? new Set(shown.map((i) => i.id)) : new Set())}
                />
              </th>
              <th>{t("Имя")}</th>
              <th>{t("Описание")}</th>
              <th>{t("Датасет")}</th>
              <th>{t("Коллекция")}</th>
              <th>{t("Конвейер")}</th>
              <th>{t("Последний прогон")}</th>
              <th>{t("Изменён")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {shown.map((item) => (
              <ScenarioRow
                key={item.id}
                item={item}
                selected={selected.has(item.id)}
                onSelect={(on) =>
                  setSelected((prev) => {
                    const next = new Set(prev);
                    if (on) next.add(item.id);
                    else next.delete(item.id);
                    return next;
                  })
                }
                onDuplicate={() => setDialog({ kind: "duplicate", id: item.id })}
                onLaunch={() => setDialog({ kind: "launch", ids: [item.id] })}
              />
            ))}
          </tbody>
        </table>
        {scenarios.data && shown.length === 0 && <Empty>{items.length ? t("Ничего не нашлось.") : t("Библиотека пуста: создайте сценарий.")}</Empty>}
      </section>
      <span className="small muted">
        {rich(t("Библиотека — каталог {dir} на сервере: то же, что видит CLI. «Запустить выбранные» ставит сценарии в очередь по порядку, каждый проходит dry-run."), {
          dir: <span className="mono">configs/scenarios/</span>,
        })}
      </span>

      {dialog?.kind === "new" && (
        <NameDialog
          kind="scenario"
          open
          onClose={() => setDialog(null)}
          title={t("Новый сценарий")}
          initial="new_scenario"
          confirm={t("Создать")}
          taken={ids}
          onSubmit={(name) => createDocument("scenario", skeleton("scenario", name, stand.data?.qdrant[0]?.url))}
        />
      )}
      {dialog?.kind === "duplicate" && (
        <NameDialog
          kind="scenario"
          open
          onClose={() => setDialog(null)}
          title={t("Дублировать {id}", { id: dialog.id })}
          initial={`${dialog.id}_copy`}
          confirm={t("Дублировать")}
          taken={ids}
          onSubmit={(name) => duplicate("scenario", dialog.id, name)}
        />
      )}
      <ImportDialog kind="scenario" open={dialog?.kind === "import"} onClose={() => setDialog(null)} />
      {dialog?.kind === "launch" && (
        <BatchLaunch
          ids={dialog.ids}
          onClose={() => {
            setDialog(null);
            setSelected(new Set());
          }}
        />
      )}
    </Page>
  );
}

function ScenarioRow({ item, selected, onSelect, onDuplicate, onLaunch }: {
  item: S["ScenarioItem"];
  selected: boolean;
  onSelect: (on: boolean) => void;
  onDuplicate: () => void;
  onLaunch: () => void;
}) {
  const t = useT();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const navigate = useNavigate();
  const href = `/scenarios/${encodeURIComponent(item.id)}`;

  const onDelete = async () => {
    const ok = await confirm({
      title: t("Удалить сценарий {id}?", { id: item.id }),
      body: rich(t("Файл {file} будет удалён. Прогоны останутся: у каждого есть свой snapshot сценария."), {
        file: <span className="mono">configs/scenarios/{item.id}.yaml</span>,
      }),
      confirm: t("Удалить"),
      danger: true,
    });
    if (!ok) return;
    try {
      await deleteDocument("scenario", item.id, item.etag);
    } catch (e) {
      if (e instanceof ApiProblem && e.status === 409) {
        await confirm({ title: t("Файл изменился"), body: t("Сценарий изменили вне редактора — обновите список и повторите."), confirm: t("Понятно") });
      } else throw e;
    } finally {
      invalidateLibrary(qc, "scenario");
    }
  };

  return (
    <tr className={selected ? k.sel : undefined}>
      <td>
        <input type="checkbox" aria-label={t("Выбрать {id}", { id: item.id })} checked={selected} onChange={(e) => onSelect(e.target.checked)} />
      </td>
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
          <Chip tone="warn" title={t("Файл не проходит схему: откройте его в редакторе")}>
            {t("черновик")}
          </Chip>
        )}
      </td>
      <td>{item.dataset ? fmtDataset(item.dataset) : <span className="muted">—</span>}</td>
      <td className="mono">{compactNames(item.collections.map((c) => c.collection)) || <span className="muted">—</span>}</td>
      <td>
        <div className={s.chips}>{item.pipeline && pipelineChips(item.pipeline).map((c) => <PipeChip key={c.label} on={c.on}>{c.label}</PipeChip>)}</div>
      </td>
      <td>
        <LastRun run={item.last_run ?? null} />
      </td>
      <td className="muted nowrap">{fmtWhen(item.updated_at)}</td>
      <td>
        <div className={s.acts}>
          <LinkButton small to={href}>
            {t("Открыть")}
          </LinkButton>
          <Button small onClick={onDuplicate}>
            {t("Дублировать")}
          </Button>
          <Menu
            trigger={
              <Button small aria-label={t("Ещё: {id}", { id: item.id })}>
                ⋯
              </Button>
            }
            items={[
              { label: t("Открыть"), onSelect: () => navigate(href) },
              { label: t("Запустить…"), onSelect: onLaunch, disabled: !item.valid },
              { label: t("Экспорт YAML"), onSelect: () => void exportYaml("scenario", item.id) },
              { label: t("Удалить…"), onSelect: () => void onDelete(), danger: true },
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
    <span style={{ display: "inline-flex", gap: "2px 6px", alignItems: "center", flexWrap: "wrap", minWidth: 120 }}>
      <StateChip state={run.state} />
      <Link className="small" to={href}>
        {detail}
      </Link>
    </span>
  );
}
