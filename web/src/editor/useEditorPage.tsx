import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useBlocker, useNavigate } from "react-router";
import type { StoreApi } from "zustand";
import { ApiProblem, type Problem, type S } from "../api/client";
import { keys } from "../api/queries";
import { setIn } from "../config/yamlDoc";
import { fmtTime, fmtWhen } from "../lib/format";
import { NameDialog } from "../pages/library/LibraryDialogs";
import { LIBRARY, createDocument, deleteDocument, downloadText, getDocument, invalidateLibrary, type LibraryKind } from "../pages/library/library";
import { Button } from "../ui/Button";
import { Dialog, useConfirm } from "../ui/Dialog";
import { ErrorBox } from "../ui/ProblemView";
import { dryRun, save } from "./commands";
import type { SaveStep } from "./launch";
import s from "./editor.module.css";
import { isDirty, type EditorState } from "./store";
import { useStore } from "zustand";

export function useLibraryDocument(kind: LibraryKind, id: string) {
  return useQuery({
    queryKey: kind === "scenario" ? keys.scenario(id) : keys.profile(id),
    queryFn: () => getDocument(kind, id),
    staleTime: Infinity,
    refetchOnWindowFocus: true,
    retry: false,
  });
}

/**
 * Команды страницы редактора: сохранить (PUT с etag, 409 → диалог), проверить (dry-run без
 * сохранения), «Сохранить как…», сбросить, экспорт, удалить; синхронизация с файлом на диске и защита
 * несохранённых правок при уходе со страницы.
 */
export function useEditorPage(store: StoreApi<EditorState>, kind: LibraryKind, server: S["ConfigDocument"] | undefined) {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const navigate = useNavigate();
  const dirty = useStore(store, isDirty);
  const id = useStore(store, (st) => st.id);
  const savedEtag = useStore(store, (st) => st.saved.etag);
  const broken = useStore(store, (st) => st.parsed.error !== null);
  const [busy, setBusy] = useState<null | "save" | "check" | "launch">(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [conflict, setConflict] = useState<Problem | null>(null);
  const [external, setExternal] = useState<S["ConfigDocument"] | null>(null);
  const [saveAs, setSaveAs] = useState(false);
  const leaving = useRef(false);
  const docKey = kind === "scenario" ? keys.scenario(id) : keys.profile(id);

  // Файл на диске изменился (IDE, git): без правок — тихо перечитать, с правками — плашка.
  useEffect(() => {
    if (!server || server.etag === savedEtag) return;
    if (!isDirty(store.getState())) store.getState().loadSaved(server, true);
    else setExternal(server);
  }, [server, savedEtag, store]);

  // Несохранённые правки: уход по ссылке спрашивает, закрытие вкладки — браузерное предупреждение.
  const blocker = useBlocker(({ currentLocation, nextLocation }) => !leaving.current && isDirty(store.getState()) && currentLocation.pathname !== nextLocation.pathname);
  useEffect(() => {
    if (blocker.state !== "blocked") return;
    void confirm({
      title: "Уйти без сохранения?",
      body: "В редакторе есть несохранённые правки. Они пропадут.",
      confirm: "Уйти",
      cancel: "Остаться",
      danger: true,
    }).then((ok) => (ok ? blocker.proceed() : blocker.reset()));
  }, [blocker, confirm]);
  useEffect(() => {
    if (!dirty) return;
    const onUnload = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [dirty]);

  const run = useCallback(async <T,>(what: "save" | "check" | "launch", fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(what);
    setError(null);
    setNote(null);
    try {
      return await fn();
    } catch (e) {
      setError(e);
      return undefined;
    } finally {
      setBusy(null);
    }
  }, []);

  /** Сохранить, если есть правки: шаг save автомата запуска. */
  const doSave = useCallback(async (): Promise<SaveStep> => {
    if (!isDirty(store.getState())) return { kind: "saved", valid: store.getState().saved.valid };
    const out = await save(store);
    if (out.kind === "saved") {
      qc.setQueryData(docKey, out.doc);
      invalidateLibrary(qc, kind);
      setExternal(null);
      setNote(`Сохранено ${fmtTime(new Date())}${out.doc.valid ? "" : " · черновик: не проходит схему"}`);
      return { kind: "saved", valid: out.doc.valid };
    }
    if (out.kind === "conflict") {
      setConflict(out.problem);
      return { kind: "conflict", problem: out.problem };
    }
    if (out.kind === "rejected") {
      store.getState().setTab(out.problem.issues.some((i) => i.code === "config.yaml") ? "yaml" : store.getState().tab);
      return { kind: "rejected" };
    }
    return { kind: "blocked" };
  }, [store, qc, docKey, kind]);

  const onSave = () => run("save", doSave);
  const onCheck = () => run("check", () => dryRun(store));

  const reload = async () => {
    const fresh = await getDocument(kind, store.getState().id);
    store.getState().loadSaved(fresh, true);
    store.getState().setCheck(null);
    qc.setQueryData(docKey, fresh);
    setConflict(null);
    setExternal(null);
  };

  const copyMine = async () => {
    await navigator.clipboard.writeText(store.getState().text);
    setNote("Текст редактора скопирован в буфер обмена");
  };

  const onReset = async () => {
    const ok = await confirm({ title: "Сбросить к сохранённому?", body: "Правки в редакторе пропадут.", confirm: "Сбросить", danger: true });
    if (ok) store.getState().setText(store.getState().saved.text);
  };

  const onDelete = async () => {
    const ok = await confirm({
      title: `Удалить ${LIBRARY[kind].noun} ${id}?`,
      body: kind === "scenario" ? "Файл будет удалён. Прогоны останутся: у каждого есть свой snapshot." : "Файл будет удалён. Коллекции в Qdrant останутся.",
      confirm: "Удалить",
      danger: true,
    });
    if (!ok) return;
    await run("save", async () => {
      await deleteDocument(kind, id, store.getState().saved.etag);
      invalidateLibrary(qc, kind);
      leaving.current = true;
      navigate(kind === "scenario" ? "/scenarios" : "/indexes?tab=profiles");
    });
  };

  const onExport = () => downloadText(`${id}.yaml`, store.getState().text);

  const dialogs: ReactNode = (
    <>
      <Dialog
        open={conflict !== null}
        onOpenChange={(o) => !o && setConflict(null)}
        title="Файл изменён вне редактора"
        footer={
          <>
            <Button onClick={() => setConflict(null)}>Отмена</Button>
            <Button onClick={() => void copyMine()}>Скопировать мой YAML</Button>
            <Button variant="danger" onClick={() => void reload()}>
              Перезагрузить
            </Button>
          </>
        }
      >
        <span>{conflict?.issues[0]?.message}</span>
        <span className="muted">
          Перезагрузка покажет версию с диска, правки из редактора пропадут. Чтобы их не потерять, сначала скопируйте текст. Слияния нет.
        </span>
      </Dialog>
      {saveAs && (
        <NameDialog
          kind={kind}
          open
          onClose={() => setSaveAs(false)}
          title="Сохранить как…"
          initial={`${id}_copy`}
          confirm="Создать копию"
          onSubmit={async (name) => {
            const doc = await createDocument(kind, setIn(store.getState().text, [LIBRARY[kind].idField], name));
            invalidateLibrary(qc, kind);
            leaving.current = true;
            return doc;
          }}
        >
          <span className="muted small">Новый файл получит текущий текст редактора; исходный {LIBRARY[kind].noun} не изменится.</span>
        </NameDialog>
      )}
    </>
  );

  const banners: ReactNode = (
    <>
      {external && (
        <div className={s.banner}>
          Файл изменили вне редактора ({fmtWhen(external.updated_at)}), а здесь есть несохранённые правки.
          <Button small onClick={() => void copyMine()}>
            Скопировать мой YAML
          </Button>
          <Button small variant="danger" onClick={() => void reload()}>
            Перезагрузить
          </Button>
        </div>
      )}
      {error !== null && !(error instanceof ApiProblem && error.status === 409) && <ErrorBox error={error} />}
    </>
  );

  return {
    dirty,
    broken,
    busy,
    note,
    setNote,
    run,
    doSave,
    onSave,
    onCheck,
    onReset,
    onDelete,
    onExport,
    onSaveAs: () => setSaveAs(true),
    leaving,
    dialogs,
    banners,
  };
}
