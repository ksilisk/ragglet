import { useEffect, useMemo, type ReactNode } from "react";
import { parsePath } from "../config/paths";
import { useT } from "../i18n/i18n";
import { bindIssue, fieldDomId } from "../config/registry";
import { Tabs } from "../ui/Tabs";
import { useVisibleIssues } from "./derived";
import { offsetOf, yamlDiagnostics } from "./diagnostics";
import s from "./editor.module.css";
import { ProblemsPanel } from "./ProblemsPanel";
import { SectionNav } from "./Section";
import { useEditor, useEditorContext, type CheckResult } from "./store";
import { YamlEditor } from "./YamlEditor";
import { rangeOf } from "../config/yamlDoc";

/** Общий редактор конфига: шапка страницы, «Форма | YAML», секции с навигацией, панель «Проверка». */
export function EditorShell({ head, sections, children, report }: {
  head: ReactNode;
  sections: ReadonlyArray<{ id: string; title: string }>;
  children: ReactNode;
  report?: (check: CheckResult) => ReactNode;
}) {
  const t = useT();
  const { store, registry } = useEditorContext();
  const tab = useEditor((st) => st.tab);
  const text = useEditor((st) => st.text);
  const parsed = useEditor((st) => st.parsed);
  const focus = useEditor((st) => st.focus);
  const issues = useVisibleIssues();
  const diagnostics = useMemo(() => yamlDiagnostics(text, parsed, issues), [text, parsed, issues]);

  // Клик в панели «Проверка»: поле формы, если оно есть, иначе строка во вкладке YAML.
  useEffect(() => {
    if (!focus) return;
    const st = store.getState();
    const bound = focus.path ? bindIssue({ code: "", severity: "error", message: "", path: focus.path, where: null }, registry) : null;
    if (bound?.anchor && st.tab === "form") {
      st.setOpen(bound.section, true);
      requestAnimationFrame(() => {
        const el = document.getElementById(fieldDomId(bound.anchor!));
        if (!el) return goYaml();
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        const control =
          document.getElementById(`${fieldDomId(bound.anchor!)}-input`) ?? el.querySelector<HTMLElement>("input, select, textarea") ?? el.querySelector<HTMLElement>("button");
        control?.focus({ preventScroll: true });
      });
      return;
    }
    goYaml();

    function goYaml() {
      const cur = store.getState();
      cur.setTab("yaml");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- реагируем только на новый запрос фокуса
  }, [focus?.seq]);

  const reveal = useMemo(() => {
    if (!focus) return null;
    if (focus.path && parsed.doc) {
      const range = rangeOf(parsed.doc, parsePath(focus.path));
      if (range) return { offset: range[0], seq: focus.seq };
    }
    if (focus.line) return { offset: offsetOf(text, focus.line), seq: focus.seq };
    return null;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- позиция считается в момент запроса
  }, [focus?.seq]);

  return (
    <>
      {head}
      <Tabs<"form" | "yaml">
        label={t("Вид")}
        value={tab}
        onChange={(next) => store.getState().setTab(next)}
        items={[
          { id: "form", label: t("Форма") },
          { id: "yaml", label: parsed.error ? "YAML ●" : "YAML", title: parsed.error ? t("YAML не разбирается") : undefined },
        ]}
      />
      {tab === "form" ? (
        <>
          {parsed.error && (
            <div className={`${s.banner} ${s.danger}`} role="alert">
              {parsed.error.line
                ? t("YAML не разбирается: строка {line}, столбец {col} — {message}. Форма только для чтения, исправьте текст во вкладке YAML.", {
                    line: parsed.error.line,
                    col: parsed.error.col ?? "?",
                    message: t(parsed.error.message),
                  })
                : t("YAML не разбирается — {message}. Форма только для чтения, исправьте текст во вкладке YAML.", { message: t(parsed.error.message) })}
            </div>
          )}
          <div className={s.layout}>
            <SectionNav sections={sections} />
            <div className={s.sections}>{children}</div>
          </div>
        </>
      ) : (
        <div className={s.yaml}>
          <YamlEditor label="YAML" value={text} onChange={(next) => store.getState().setText(next)} diagnostics={diagnostics} reveal={reveal} />
          {parsed.error && (
            <div className={s.yamlErr} role="alert">
              {t("Строка {line}, столбец {col}: {message}. Пока текст не разбирается, сохранить и проверить его нельзя.", {
                line: parsed.error.line ?? "?",
                col: parsed.error.col ?? "?",
                message: t(parsed.error.message),
              })}
            </div>
          )}
        </div>
      )}
      <ProblemsPanel report={report} />
    </>
  );
}
