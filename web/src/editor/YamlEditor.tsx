import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { yaml } from "@codemirror/lang-yaml";
import { HighlightStyle, bracketMatching, foldGutter, indentOnInput, syntaxHighlighting } from "@codemirror/language";
import { lintGutter, setDiagnostics, type Diagnostic } from "@codemirror/lint";
import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";
import { EditorState } from "@codemirror/state";
import { EditorView, drawSelection, highlightActiveLine, highlightActiveLineGutter, keymap, lineNumbers } from "@codemirror/view";
import { tags } from "@lezer/highlight";
import { useEffect, useRef } from "react";

export type { Diagnostic };

// Цвета — токены tokens.css: редактор переключается вместе с темой без пересоздания.
const theme = EditorView.theme({
  "&": { fontSize: "13px", backgroundColor: "var(--surface)", color: "var(--text)" },
  ".cm-content": { fontFamily: "var(--font-mono)", padding: "8px 0", caretColor: "var(--text)" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--text)" },
  ".cm-gutters": { fontFamily: "var(--font-mono)", backgroundColor: "var(--bg)", color: "var(--faint)", borderRight: "1px solid var(--border)" },
  ".cm-activeLine": { backgroundColor: "var(--cm-active-line)" },
  ".cm-activeLineGutter": { backgroundColor: "var(--cm-active-gutter)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "var(--accent-soft)" },
  ".cm-selectionMatch": { backgroundColor: "var(--accent-row)" },
  ".cm-tooltip": { backgroundColor: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)" },
  ".cm-panels": { backgroundColor: "var(--bg)", color: "var(--text)" },
  ".cm-foldPlaceholder": { backgroundColor: "var(--hover-strong)", color: "var(--muted)", border: "none" },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { lineHeight: "20px" },
});

const highlight = HighlightStyle.define([
  { tag: [tags.propertyName, tags.definition(tags.propertyName)], color: "var(--syn-key)" },
  { tag: [tags.string, tags.special(tags.string), tags.content], color: "var(--syn-string)" },
  { tag: [tags.number, tags.integer, tags.float], color: "var(--syn-number)" },
  { tag: [tags.bool, tags.null, tags.atom, tags.keyword], color: "var(--syn-bool)" },
  { tag: [tags.comment, tags.lineComment], color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: [tags.punctuation, tags.separator, tags.squareBracket, tags.brace, tags.meta], color: "var(--syn-punct)" },
  { tag: [tags.labelName, tags.typeName], color: "var(--syn-bool)" },
]);

/**
 * YAML в CodeMirror 6: правка (вкладка YAML) или только чтение (snapshot прогона). Диагностика —
 * снаружи: ошибки разбора и проблемы проверки по диапазонам узлов.
 */
export function YamlEditor({ value, onChange, readOnly, diagnostics, reveal, minHeight = 480, label }: {
  value: string;
  onChange?: (text: string) => void;
  readOnly?: boolean;
  diagnostics?: Diagnostic[];
  /** Прокрутить и выделить смещение; seq — чтобы повторный запрос сработал. */
  reveal?: { offset: number; seq: number } | null;
  minHeight?: number;
  label: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    const v = new EditorView({
      parent: host.current!,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(),
          highlightActiveLineGutter(),
          foldGutter(),
          history(),
          drawSelection(),
          indentOnInput(),
          bracketMatching(),
          highlightActiveLine(),
          highlightSelectionMatches(),
          syntaxHighlighting(highlight),
          yaml(),
          lintGutter(),
          keymap.of([...defaultKeymap, ...historyKeymap, ...searchKeymap, indentWithTab]),
          EditorState.readOnly.of(Boolean(readOnly)),
          EditorView.contentAttributes.of({ "aria-label": label }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) onChangeRef.current?.(update.state.doc.toString());
          }),
          theme,
          EditorView.theme({ ".cm-scroller": { minHeight: `${minHeight}px` } }),
        ],
      }),
    });
    view.current = v;
    return () => v.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- редактор создаётся один раз; текст синхронизируется ниже
  }, []);

  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const current = v.state.doc.toString();
    if (current !== value) v.dispatch({ changes: { from: 0, to: current.length, insert: value } });
  }, [value]);

  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const max = v.state.doc.length;
    const clamped = (diagnostics ?? []).map((d) => ({ ...d, from: Math.min(d.from, max), to: Math.min(Math.max(d.to, d.from), max) }));
    v.dispatch(setDiagnostics(v.state, clamped));
  }, [diagnostics, value]);

  useEffect(() => {
    const v = view.current;
    if (!v || !reveal) return;
    const offset = Math.min(reveal.offset, v.state.doc.length);
    v.dispatch({ selection: { anchor: offset }, effects: EditorView.scrollIntoView(offset, { y: "center" }) });
    v.focus();
  }, [reveal]);

  return <div ref={host} />;
}
