import { Document, Scalar, YAMLMap, isCollection, isMap, isScalar, isSeq, parseDocument, type Node } from "yaml";
import { msg, translate } from "../i18n/i18n";
import { getAt, type Path, type PathSeg } from "./paths";

/**
 * Модель редактора (ragglet-31 §4): источник правды — YAML-текст. Значение, которое уже есть в
 * тексте (скаляр или однострочная flow-коллекция), заменяется точечно по диапазону узла, остальной
 * текст не трогается. Структурные правки идут через документ `yaml` и полную сериализацию: комментарии
 * и незнакомые ключи сохраняются, форматирование нетронутых мест иногда нормализуется.
 *
 * Разбор — в режиме YAML 1.1, как у PyYAML на сервере: `yes` / `off` — булевы.
 */

export interface ParseError {
  message: string;
  line: number | null; // с 1
  col: number | null; // с 1
  offset: number;
}

export interface Parsed {
  doc: Document.Parsed | null;
  data: Record<string, unknown> | null;
  error: ParseError | null;
}

const PARSE_OPTIONS = { version: "1.1" as const, prettyErrors: true };

export function parseYaml(text: string): Parsed {
  const doc = parseDocument(text, PARSE_OPTIONS);
  const first = doc.errors[0];
  if (first) {
    const pos = first.linePos?.[0];
    return {
      doc: null,
      data: null,
      error: {
        message: first.message.split("\n")[0]!.replace(/ at line \d+, column \d+:?$/, ""),
        line: pos?.line ?? null,
        col: pos?.col ?? null,
        offset: first.pos[0],
      },
    };
  }
  const data: unknown = doc.toJS({ maxAliasCount: 100 });
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    return { doc: null, data: null, error: { message: msg("корень YAML должен быть словарём (mapping)"), line: 1, col: 1, offset: 0 } };
  }
  return { doc, data: data as Record<string, unknown>, error: null };
}

/** Есть ли в тексте блочные списки с отступом относительно ключа: так их и пишем. */
export function detectIndentSeq(text: string): boolean {
  const match = /^( *)[^\s#-][^\n]*:[ \t]*(?:#[^\n]*)?\n( *)- /m.exec(text);
  return match ? match[2]!.length > match[1]!.length : true;
}

function serialize(doc: Document, original: string): string {
  separateNewSections(doc, original);
  return doc.toString({ lineWidth: 0, minContentWidth: 0, flowCollectionPadding: false, indentSeq: detectIndentSeq(original) });
}

/** Секции верхнего уровня в файле разделены пустой строкой — новая секция тоже. */
function separateNewSections(doc: Document, original: string) {
  if (!/\n[ \t]*\n[^\s#-]/.test(original) || !isMap(doc.contents)) return;
  doc.contents.items.forEach((pair, i) => {
    if (i === 0 || isScalar(pair.key)) return; // ключи из текста — узлы; новые, созданные setIn, — строки
    const key = new Scalar(pair.key);
    key.spaceBefore = true;
    pair.key = key;
  });
}

function mustParse(text: string): Document.Parsed {
  const parsed = parseYaml(text);
  if (!parsed.doc) throw new Error(translate("YAML не разбирается: {message}", { message: parsed.error?.message ?? "" }));
  return parsed.doc;
}

const isPlainScalar = (v: unknown) => v === null || ["string", "number", "boolean"].includes(typeof v);

function sameValue(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Однострочный текст значения со стилем исходного узла; null — не умещается в строку. */
function inlineText(value: unknown, original: Node): string | null {
  let node: Node;
  if (isScalar(original)) {
    if (!isPlainScalar(value)) return null;
    if (original.type !== "PLAIN" && original.type !== "QUOTE_DOUBLE" && original.type !== "QUOTE_SINGLE") return null;
    const scalar = new Scalar(value);
    if (typeof value === "string" && (original.type === "QUOTE_DOUBLE" || original.type === "QUOTE_SINGLE")) scalar.type = original.type;
    node = scalar;
  } else if (isCollection(original) && original.flow) {
    if (!Array.isArray(value) && (value === null || typeof value !== "object")) return null;
    node = new Document().createNode(value, { flow: true });
  } else {
    return null;
  }
  const out = new Document(node, { version: "1.1" }).toString({ lineWidth: 0, flowCollectionPadding: false }).replace(/\n$/, "");
  return out.includes("\n") ? null : out;
}

/**
 * Все собственные префиксы пути — коллекции: недостающие и null-секции («rerank:») становятся
 * словарями. Сами промежуточные узлы setIn создавать нельзя: в схеме YAML 1.1 он делает их !!omap.
 */
function ensureCollections(doc: Document, path: Path) {
  for (let i = 1; i < path.length; i += 1) {
    const prefix = path.slice(0, i);
    const node = doc.getIn(prefix, true);
    if (!isCollection(node)) doc.setIn(prefix, new YAMLMap());
  }
}

function createValueNode(doc: Document, value: unknown): unknown {
  if (Array.isArray(value) && value.every(isPlainScalar)) return doc.createNode(value, { flow: true }); // [1, 3, 5, 10]
  return value !== null && typeof value === "object" ? doc.createNode(value) : value;
}

/** Записать значение по пути; undefined — удалить ключ. Возвращает новый текст. */
export function setIn(text: string, path: Path, value: unknown): string {
  if (value === undefined) return deleteIn(text, path);
  const parsed = parseYaml(text);
  if (!parsed.doc) throw new Error(translate("YAML не разбирается: {message}", { message: parsed.error?.message ?? "" }));
  if (sameValue(getAt(parsed.data, path), value)) return text;
  const doc = parsed.doc;
  const current = doc.getIn(path, true) as Node | undefined;

  if (current && current.range) {
    const inline = inlineText(value, current);
    if (inline !== null) {
      const [start, end] = current.range;
      return text.slice(0, start) + inline + text.slice(end);
    }
  }
  ensureCollections(doc, path);
  doc.setIn(path, createValueNode(doc, value));
  return serialize(doc, text);
}

/** Удалить ключ (сброс к умолчанию схемы); опустевшие словари-родители удаляются тоже. */
export function deleteIn(text: string, path: Path): string {
  const doc = mustParse(text);
  if (!doc.hasIn(path)) return text;
  doc.deleteIn(path);
  for (let i = path.length - 1; i >= 1; i -= 1) {
    const parent = doc.getIn(path.slice(0, i), true);
    if (isMap(parent) && parent.items.length === 0) doc.deleteIn(path.slice(0, i));
    else break;
  }
  return serialize(doc, text);
}

/** Добавить элемент в конец списка (список создаётся, если его нет). */
export function appendItem(text: string, path: Path, item: unknown): string {
  const doc = mustParse(text);
  const list = doc.getIn(path, true);
  if (isSeq(list)) {
    list.add(doc.createNode(item));
    return serialize(doc, text);
  }
  ensureCollections(doc, path);
  doc.setIn(path, doc.createNode([item]));
  return serialize(doc, text);
}

export function removeItem(text: string, path: Path, index: number): string {
  return deleteIn(text, [...path, index] as PathSeg[]);
}

/** Диапазон узла в тексте [начало, конец) — для подсветки проблемы во вкладке YAML. */
export function rangeOf(doc: Document.Parsed, path: Path): [number, number] | null {
  for (let len = path.length; len >= 1; len -= 1) {
    const node = doc.getIn(path.slice(0, len), true) as Node | undefined;
    if (node?.range && node.range[1] > node.range[0]) return [node.range[0], node.range[1]];
    // ключ без значения («rerank:») или путь ведёт в отсутствующий ключ: ищем пару в родителе
    const parent = len > 1 ? doc.getIn(path.slice(0, len - 1), true) : doc.contents;
    if (isMap(parent)) {
      const key = path[len - 1];
      const pair = parent.items.find((p) => (isScalar(p.key) ? p.key.value : p.key) === key);
      const keyNode = pair?.key as Node | undefined;
      if (keyNode?.range) return [keyNode.range[0], keyNode.range[1]];
    }
  }
  return null;
}
