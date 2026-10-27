import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent, type ReactNode } from "react";
import { useNavigate } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorBox } from "../../ui/ProblemView";
import k from "../../ui/kit.module.css";
import { ID_PATTERN, LIBRARY, createDocument, invalidateLibrary, type LibraryKind } from "./library";

/** Диалог имени: дублировать, новый пустой, из прогона. Имя — id файла, поэтому [A-Za-z0-9_-]. */
export function NameDialog({ kind, open, onClose, title, initial, confirm, taken, onSubmit, children }: {
  kind: LibraryKind;
  open: boolean;
  onClose: () => void;
  title: string;
  initial: string;
  confirm: string;
  taken?: ReadonlySet<string>;
  onSubmit: (name: string) => Promise<S["ConfigDocument"]>;
  children?: ReactNode;
}) {
  const [name, setName] = useState(initial);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  const invalid = !ID_PATTERN.test(name) ? "только латиница, цифры, «_» и «-», до 64 символов" : taken?.has(name) ? "такое имя уже есть" : null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (invalid) return;
    setBusy(true);
    setError(null);
    try {
      const doc = await onSubmit(name);
      onClose();
      navigate(LIBRARY[kind].route(doc.id));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && onClose()}
      title={title}
      footer={
        <>
          <Button onClick={onClose}>Отмена</Button>
          <Button variant="primary" type="submit" form="name-dialog" disabled={Boolean(invalid) || busy}>
            {confirm}
          </Button>
        </>
      }
    >
      <form id="name-dialog" onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <label htmlFor="name-dialog-input" className="small muted">
          Имя — оно же имя файла и каталога прогонов
        </label>
        <input
          id="name-dialog-input"
          className={`${k.inp} mono${invalid ? ` ${k.err}` : ""}`}
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoFocus
          spellCheck={false}
        />
        {invalid && <span className="small" style={{ color: "var(--danger-text)" }}>{invalid}</span>}
      </form>
      {children}
      {error !== null && <ErrorBox error={error} />}
    </Dialog>
  );
}

/** Новый из YAML: вставить текст или выбрать файл; id берётся из name / id в тексте. */
export function ImportDialog({ kind, open, onClose }: { kind: LibraryKind; open: boolean; onClose: () => void }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const qc = useQueryClient();
  const navigate = useNavigate();

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const doc = await createDocument(kind, text);
      invalidateLibrary(qc, kind);
      onClose();
      setText("");
      navigate(LIBRARY[kind].route(doc.id));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && onClose()}
      wide
      title={kind === "scenario" ? "Новый сценарий из YAML" : "Новый профиль из YAML"}
      footer={
        <>
          <Button onClick={onClose}>Отмена</Button>
          <Button variant="primary" onClick={submit} disabled={!text.trim() || busy}>
            Создать
          </Button>
        </>
      }
    >
      <span className="muted">
        Имя {LIBRARY[kind].genitive} берётся из поля <span className="mono">{LIBRARY[kind].idField}</span>. Невалидный по схеме текст сохраняется
        черновиком.
      </span>
      <input
        type="file"
        accept=".yaml,.yml,text/yaml"
        aria-label="Файл YAML"
        onChange={async (e) => {
          const file = e.target.files?.[0];
          if (file) setText(await file.text());
        }}
      />
      <textarea
        className={`${k.inp} mono`}
        rows={16}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={`${LIBRARY[kind].idField}: my_${kind}\n…`}
        aria-label="Текст YAML"
        spellCheck={false}
      />
      {error !== null && <ErrorBox error={error} />}
      {error instanceof ApiProblem && error.status === 409 && <span className="small muted">Такое имя уже занято: поменяйте его в тексте.</span>}
    </Dialog>
  );
}
