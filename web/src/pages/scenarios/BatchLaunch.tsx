import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "react-router";
import type { Problem, S } from "../../api/client";
import { keys } from "../../api/queries";
import { Button } from "../../ui/Button";
import { Chip } from "../../ui/Chip";
import { Dialog } from "../../ui/Dialog";
import { IssueList } from "../../ui/ProblemView";
import k from "../../ui/kit.module.css";
import { getDocument } from "../library/library";
import { dryRunScenario, enqueueScenario, planLine } from "../runs/enqueue";

type Item =
  | { id: string; status: "checking" }
  | { id: string; status: "ready"; text: string; check: S["RunCheck"]; pick: boolean }
  | { id: string; status: "problems"; problem: Problem }
  | { id: string; status: "started"; runId: string }
  | { id: string; status: "error"; message: string };

/**
 * «Запустить выбранные» (ragglet-15, решение 4): тексты из библиотеки, по одному dry-run на сценарий
 * по порядку; прошедшие ставятся в очередь в том же порядке, не прошедшие остаются со своими проблемами.
 */
export function BatchLaunch({ ids, onClose }: { ids: string[]; onClose: () => void }) {
  const qc = useQueryClient();
  const [items, setItems] = useState<Item[]>(() => ids.map((id) => ({ id, status: "checking" })));
  const [launching, setLaunching] = useState(false);
  const [done, setDone] = useState(false);

  const update = (id: string, item: Item) => setItems((prev) => prev.map((x) => (x.id === id ? item : x)));

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      for (const id of ids) {
        if (cancelled) return;
        try {
          const doc = await getDocument("scenario", id);
          const out = await dryRunScenario(doc.yaml, id);
          if (cancelled) return;
          update(id, out.ok ? { id, status: "ready", text: doc.yaml, check: out.check, pick: true } : { id, status: "problems", problem: out.problem });
        } catch (e) {
          if (!cancelled) update(id, { id, status: "error", message: e instanceof Error ? e.message : String(e) });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ids]);

  const picked = items.filter((x): x is Extract<Item, { status: "ready" }> => x.status === "ready" && x.pick);
  const checking = items.some((x) => x.status === "checking");

  const start = async () => {
    setLaunching(true);
    for (const item of picked) {
      try {
        const out = await enqueueScenario(item.text, item.id);
        update(item.id, out.ok ? { id: item.id, status: "started", runId: out.accepted.id } : { id: item.id, status: "problems", problem: out.problem });
      } catch (e) {
        update(item.id, { id: item.id, status: "error", message: e instanceof Error ? e.message : String(e) });
      }
    }
    void qc.invalidateQueries({ queryKey: keys.tasks });
    void qc.invalidateQueries({ queryKey: keys.runs });
    void qc.invalidateQueries({ queryKey: keys.scenarios });
    setLaunching(false);
    setDone(true);
  };

  return (
    <Dialog
      open
      wide
      onOpenChange={(o) => !o && onClose()}
      title={ids.length === 1 ? `Запустить ${ids[0]}` : `Запустить выбранные · ${ids.length}`}
      footer={
        done ? (
          <Button variant="primary" onClick={onClose}>
            Готово
          </Button>
        ) : (
          <>
            <Button onClick={onClose}>Отмена</Button>
            <Button variant="primary" onClick={() => void start()} disabled={checking || launching || picked.length === 0}>
              {launching ? "Постановка…" : `Запустить ${picked.length}`}
            </Button>
          </>
        )
      }
    >
      <span className="muted">Каждый сценарий проходит dry-run; в очередь встают по порядку. С предупреждениями — только если оставить отметку.</span>
      <table className={`${k.tbl} ${k.compact}`} aria-label="Сценарии к запуску">
        <tbody>
          {items.map((item, i) => (
            <tr key={item.id}>
              <td className="num muted">{i + 1}</td>
              <td style={{ width: 28 }}>
                {item.status === "ready" && (
                  <input
                    type="checkbox"
                    aria-label={`Запустить ${item.id}`}
                    checked={item.pick}
                    disabled={launching || done}
                    onChange={(e) => update(item.id, { ...item, pick: e.target.checked })}
                  />
                )}
              </td>
              <td>
                <Link className="mono" to={`/scenarios/${encodeURIComponent(item.id)}`}>
                  {item.id}
                </Link>
                <Detail item={item} />
              </td>
              <td className="r nowrap">
                <Status item={item} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Dialog>
  );
}

function Status({ item }: { item: Item }) {
  switch (item.status) {
    case "checking":
      return <Chip tone="q">проверка…</Chip>;
    case "ready":
      return item.check.warnings.length ? <Chip tone="warn">предупреждений: {item.check.warnings.length}</Chip> : <Chip tone="ok">готов</Chip>;
    case "problems":
      return <Chip tone="fail">не прошёл</Chip>;
    case "started":
      return (
        <Link to={`/runs/${encodeURIComponent(item.runId)}`}>
          <Chip tone="run">в очереди →</Chip>
        </Link>
      );
    case "error":
      return <Chip tone="fail">ошибка</Chip>;
  }
}

function Detail({ item }: { item: Item }) {
  if (item.status === "ready") {
    return (
      <div className="small muted">
        {planLine(item.check)}
        {item.check.warnings.length > 0 && <IssueList issues={item.check.warnings} />}
      </div>
    );
  }
  if (item.status === "problems") return <IssueList issues={item.problem.issues} />;
  if (item.status === "error") return <div className="small" style={{ color: "var(--danger-text)" }}>{item.message}</div>;
  return null;
}
