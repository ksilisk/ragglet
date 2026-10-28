// Путь поля конфига так, как его отдаёт API (ragglet-28 §3): "retrieval.sources[1].params".

export type PathSeg = string | number;
export type Path = PathSeg[];

const SEGMENT = /([^.[\]]+)|\[(\d+)\]/g;

export function parsePath(path: string): Path {
  const out: Path = [];
  for (const match of path.matchAll(SEGMENT)) {
    out.push(match[2] !== undefined ? Number(match[2]) : match[1]!);
  }
  return out;
}

export function formatPath(path: readonly PathSeg[]): string {
  let out = "";
  for (const seg of path) {
    if (typeof seg === "number") out += `[${seg}]`;
    else out += out ? `.${seg}` : seg;
  }
  return out;
}

/** Шаблон пути: индексы списков заменены на []: "retrieval.sources[].params". */
export function patternOf(path: readonly PathSeg[]): string {
  return formatPath(path).replace(/\[\d+\]/g, "[]");
}

export function getAt(data: unknown, path: readonly PathSeg[]): unknown {
  let cur: unknown = data;
  for (const seg of path) {
    if (cur === null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string | number, unknown>)[seg];
  }
  return cur;
}
