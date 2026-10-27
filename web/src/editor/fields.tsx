import { useEffect, useMemo, useState, type ReactNode } from "react";
import { parsePath } from "../config/paths";
import { fieldDomId } from "../config/registry";
import { cx } from "../ui/cx";
import k from "../ui/kit.module.css";
import { useDefault, useEditorActions, useHintsAt, useIssuesAt, useValue } from "./derived";
import s from "./editor.module.css";
import { useEditor } from "./store";

type Span = 1 | 2 | 3 | "full";
const spanClass = (span?: Span) => (span === 2 ? s.span2 : span === 3 ? s.span3 : span === "full" ? s.full : undefined);

const show = (v: unknown) => (v === null || v === undefined ? "" : Array.isArray(v) ? v.join(", ") : String(v));

/** Подпись, контрол, проблемы проверки и подсказки у поля; id — для прокрутки из панели «Проверка». */
export function FieldShell({ path, label, children, span, extra, resettable = true }: {
  path: string;
  label: ReactNode;
  children: ReactNode;
  span?: Span;
  extra?: ReactNode;
  resettable?: boolean;
}) {
  const issues = useIssuesAt(path);
  const hints = useHintsAt(path);
  const value = useValue(path);
  const def = useDefault(path);
  const { remove } = useEditorActions();
  const readOnly = useEditor((st) => st.parsed.error !== null);
  const canReset = resettable && !readOnly && value !== undefined && def !== undefined && JSON.stringify(value) !== JSON.stringify(def);
  return (
    <div className={cx(s.field, spanClass(span))} id={fieldDomId(path)}>
      <div className={s.label}>
        <label htmlFor={`${fieldDomId(path)}-input`}>{label}</label>
        {canReset && (
          <button type="button" className={s.reset} onClick={() => remove(parsePath(path))} title={`Удалить ключ: будет умолчание схемы (${show(def)})`}>
            по умолчанию
          </button>
        )}
      </div>
      {children}
      {issues.map((issue, i) => (
        <span key={`i${i}`} className={cx(s.hint, issue.severity === "error" ? s.err : s.warn)}>
          {issue.message}
        </span>
      ))}
      {hints.map((hint, i) => (
        <span key={`h${i}`} className={cx(s.hint, hint.severity === "warning" && s.warn)}>
          {hint.message}
        </span>
      ))}
      {extra}
    </div>
  );
}

function useControl(path: string) {
  const value = useValue(path);
  const def = useDefault(path);
  const issues = useIssuesAt(path);
  const hints = useHintsAt(path);
  const { edit, remove } = useEditorActions();
  const readOnly = useEditor((st) => st.parsed.error !== null);
  const segs = useMemo(() => parsePath(path), [path]);
  return {
    value,
    def,
    readOnly,
    id: `${fieldDomId(path)}-input`,
    tone: issues.some((i) => i.severity === "error") ? k.err : issues.length || hints.some((h) => h.severity === "warning") ? k.warn : undefined,
    set: (v: unknown) => edit(segs, v),
    reset: () => remove(segs),
  };
}

/** Строка. Пустое значение при уходе с поля удаляет ключ (optional): так ключ не прыгает в конец секции при наборе. */
export function TextField({ path, label, mono, placeholder, span, optional = true, list, readOnly: forceRO, extra }: {
  path: string;
  label: ReactNode;
  mono?: boolean;
  placeholder?: string;
  span?: Span;
  optional?: boolean;
  list?: string;
  readOnly?: boolean;
  extra?: ReactNode;
}) {
  const c = useControl(path);
  const value = typeof c.value === "string" || typeof c.value === "number" ? String(c.value) : "";
  return (
    <FieldShell path={path} label={label} span={span} extra={extra}>
      <input
        id={c.id}
        className={cx(k.inp, mono && "mono", c.tone)}
        value={value}
        placeholder={placeholder ?? (c.def !== undefined && c.def !== null ? String(c.def) : undefined)}
        readOnly={c.readOnly || forceRO}
        list={list}
        spellCheck={false}
        onChange={(e) => c.set(e.target.value)}
        onBlur={(e) => optional && e.target.value === "" && c.value !== undefined && c.reset()}
      />
    </FieldShell>
  );
}

/** Число: черновик ввода живёт локально («1.», «-»), в текст попадает только число. */
export function NumberField({ path, label, min, integer = true, span, placeholder, unit }: {
  path: string;
  label: ReactNode;
  min?: number;
  integer?: boolean;
  span?: Span;
  placeholder?: string;
  unit?: string;
}) {
  const c = useControl(path);
  const external = typeof c.value === "number" ? String(c.value) : c.value === undefined || c.value === null ? "" : String(c.value);
  const [draft, setDraft] = useState(external);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setDraft(external);
  }, [external, focused]);

  const parsed = draft.trim() === "" ? null : Number(draft.replace(",", "."));
  const invalid =
    parsed !== null && (Number.isNaN(parsed) || (integer && !Number.isInteger(parsed)) ? "не число" + (integer ? " (целое)" : "") : min !== undefined && parsed < min ? `не меньше ${min}` : null);

  return (
    <FieldShell
      path={path}
      label={unit ? `${String(label)}, ${unit}` : label}
      span={span}
      extra={invalid ? <span className={cx(s.hint, s.err)}>{invalid}</span> : null}
    >
      <input
        id={c.id}
        className={cx(k.inp, "num", invalid ? k.err : c.tone)}
        inputMode={integer ? "numeric" : "decimal"}
        value={draft}
        placeholder={placeholder ?? (c.def !== undefined && c.def !== null ? String(c.def) : undefined)}
        readOnly={c.readOnly}
        onFocus={() => setFocused(true)}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = e.target.value.trim() === "" ? null : Number(e.target.value.replace(",", "."));
          if (n !== null && !Number.isNaN(n) && (!integer || Number.isInteger(n)) && (min === undefined || n >= min)) c.set(n);
        }}
        onBlur={() => {
          setFocused(false);
          if (draft.trim() === "" && c.value !== undefined) c.reset();
        }}
      />
    </FieldShell>
  );
}

export interface Option {
  value: string;
  label?: string;
}

export function SelectField({ path, label, options, span, mono, onPick }: {
  path: string;
  label: ReactNode;
  options: ReadonlyArray<string | Option>;
  span?: Span;
  mono?: boolean;
  /** Своя запись вместо set(value): например, переключение режима со связанными полями. */
  onPick?: (value: string) => void;
}) {
  const c = useControl(path);
  const current = c.value ?? c.def;
  const opts = options.map((o) => (typeof o === "string" ? { value: o } : o));
  const unknown = current !== undefined && current !== null && !opts.some((o) => o.value === String(current));
  return (
    <FieldShell path={path} label={label} span={span}>
      <select
        id={c.id}
        className={cx(k.inp, mono && "mono", c.tone)}
        value={current === undefined || current === null ? "" : String(current)}
        disabled={c.readOnly}
        onChange={(e) => (onPick ? onPick(e.target.value) : c.set(e.target.value))}
      >
        {(current === undefined || current === null) && <option value="">—</option>}
        {unknown && <option value={String(current)}>{String(current)} — не поддерживается</option>}
        {opts.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label ?? o.value}
            {c.value === undefined && o.value === String(c.def) ? " (по умолчанию)" : ""}
          </option>
        ))}
      </select>
    </FieldShell>
  );
}

export function CheckField({ path, label, onToggle, disabled }: { path: string; label: ReactNode; onToggle?: (on: boolean) => void; disabled?: boolean }) {
  const c = useControl(path);
  const checked = Boolean(c.value ?? c.def);
  return (
    <div className={s.field} id={fieldDomId(path)}>
      <span className={s.chk}>
        <input id={c.id} type="checkbox" checked={checked} disabled={c.readOnly || disabled} onChange={(e) => (onToggle ? onToggle(e.target.checked) : c.set(e.target.checked))} />
        <label htmlFor={c.id}>{label}</label>
      </span>
      <FieldMessages path={path} />
    </div>
  );
}

export function FieldMessages({ path }: { path: string }) {
  const issues = useIssuesAt(path);
  const hints = useHintsAt(path);
  return (
    <>
      {issues.map((issue, i) => (
        <span key={`i${i}`} className={cx(s.hint, issue.severity === "error" ? s.err : s.warn)}>
          {issue.message}
        </span>
      ))}
      {hints.map((hint, i) => (
        <span key={`h${i}`} className={cx(s.hint, hint.severity === "warning" && s.warn)}>
          {hint.message}
        </span>
      ))}
    </>
  );
}

/** Набор значений из допустимых: метрики, latency_steps. */
export function CheckSetField({ path, label, options, disabled, span }: {
  path: string;
  label: ReactNode;
  options: readonly string[];
  disabled?: ReadonlySet<string>;
  span?: Span;
}) {
  const c = useControl(path);
  const current = (Array.isArray(c.value) ? c.value : Array.isArray(c.def) ? c.def : []) as string[];
  const toggle = (name: string, on: boolean) => {
    const next = on ? [...current, name] : current.filter((x) => x !== name);
    c.set(options.filter((o) => next.includes(o)).concat(next.filter((x) => !options.includes(x))));
  };
  return (
    <FieldShell path={path} label={label} span={span}>
      <div className={s.chkRow}>
        {options.map((name) => {
          const id = `${c.id}-${name}`;
          const off = disabled?.has(name) && !current.includes(name);
          return (
            <span key={name} className={s.chk}>
              <input id={id} type="checkbox" checked={current.includes(name)} disabled={c.readOnly || off} onChange={(e) => toggle(name, e.target.checked)} />
              <label htmlFor={id} className={off ? "muted" : undefined}>
                {name}
              </label>
            </span>
          );
        })}
        {current
          .filter((x) => !options.includes(x))
          .map((x) => (
            <span key={x} className={cx(s.hint, s.warn)}>
              {x} — неизвестно
            </span>
          ))}
      </div>
    </FieldShell>
  );
}

/** Список чисел чипами (.kchip): metrics.quality.k. */
export function NumberListField({ path, label, bad, min = 1, span }: {
  path: string;
  label: ReactNode;
  bad?: (n: number) => "error" | "warning" | null;
  min?: number;
  span?: Span;
}) {
  const c = useControl(path);
  const current = (Array.isArray(c.value) ? c.value : Array.isArray(c.def) ? c.def : []) as number[];
  const [draft, setDraft] = useState("");
  const add = () => {
    const n = Number(draft.trim());
    if (!Number.isInteger(n) || n < min) return;
    if (!current.includes(n)) c.set([...current, n].sort((a, b) => a - b));
    setDraft("");
  };
  return (
    <FieldShell path={path} label={label} span={span}>
      <div className={s.kchips}>
        {current.map((n) => {
          const mark = bad?.(n);
          return (
            <span key={n} className={cx(s.kchip, mark === "error" && s.bad, mark === "warning" && s.warnChip)}>
              {n}
              {!c.readOnly && (
                <button type="button" aria-label={`Убрать ${n}`} onClick={() => c.set(current.filter((x) => x !== n))}>
                  ×
                </button>
              )}
            </span>
          );
        })}
        {!c.readOnly && (
          <input
            id={c.id}
            className={cx(k.inp, k.sm, "num", s.kadd)}
            placeholder="+ k"
            inputMode="numeric"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === ",") {
                e.preventDefault();
                add();
              }
            }}
            onBlur={add}
          />
        )}
      </div>
    </FieldShell>
  );
}

/** Список чисел строкой «50, 95, 99»: percentiles. */
export function NumberCsvField({ path, label, span }: { path: string; label: ReactNode; span?: Span }) {
  const c = useControl(path);
  const external = show(c.value);
  const [draft, setDraft] = useState(external);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setDraft(external);
  }, [external, focused]);
  const parse = (text: string) => text.split(/[,\s]+/).filter(Boolean).map(Number);
  const values = parse(draft);
  const invalid = values.some((n) => !Number.isFinite(n));
  return (
    <FieldShell path={path} label={label} span={span} extra={invalid ? <span className={cx(s.hint, s.err)}>числа через запятую</span> : null}>
      <input
        id={c.id}
        className={cx(k.inp, "num", invalid ? k.err : c.tone)}
        value={draft}
        placeholder={show(c.def)}
        readOnly={c.readOnly}
        onFocus={() => setFocused(true)}
        onChange={(e) => {
          setDraft(e.target.value);
          const next = parse(e.target.value);
          if (next.length && next.every(Number.isFinite)) c.set(next);
        }}
        onBlur={() => {
          setFocused(false);
          if (draft.trim() === "" && c.value !== undefined) c.reset();
        }}
      />
    </FieldShell>
  );
}

/** Теги чипами: tags в шапке. */
export function TagsField({ path }: { path: string }) {
  const c = useControl(path);
  const tags = (Array.isArray(c.value) ? c.value : []).map(String);
  const [draft, setDraft] = useState("");
  const add = () => {
    const t = draft.trim();
    if (t && !tags.includes(t)) c.set([...tags, t]);
    setDraft("");
  };
  return (
    <div className={s.kchips} id={fieldDomId(path)}>
      {tags.map((t) => (
        <span key={t} className={k.chip + " " + k["c-tag"]}>
          {t}
          {!c.readOnly && (
            <button type="button" aria-label={`Убрать тег ${t}`} onClick={() => c.set(tags.filter((x) => x !== t))} style={{ all: "unset", cursor: "pointer" }}>
              ×
            </button>
          )}
        </span>
      ))}
      {!c.readOnly && (
        <input
          id={c.id}
          className={cx(k.inp, k.sm)}
          style={{ width: 120 }}
          placeholder="+ тег"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              add();
            }
          }}
          onBlur={add}
        />
      )}
    </div>
  );
}

export function Grid({ cols, children }: { cols: 2 | 3 | 4 | 5; children: ReactNode }) {
  return <div className={cx(s.grid, s[`g${cols}`])}>{children}</div>;
}

export function Note({ children }: { children: ReactNode }) {
  return <span className={s.note}>{children}</span>;
}
