/**
 * CSV таблиц для статьи (ragglet-32 §10): машинные значения — ключи summary в заголовке, точка как
 * десятичный разделитель, без разделителей тысяч, null — пустая ячейка; RFC 4180, UTF-8.
 */
export type Cell = string | number | boolean | null | undefined;

function cell(value: Cell): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(header: string[], rows: Cell[][]): string {
  return [header, ...rows].map((row) => row.map(cell).join(",")).join("\r\n") + "\r\n";
}

/** Скачать файл, собранный в браузере: Blob и временная ссылка <a download>. */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadCsv(filename: string, header: string[], rows: Cell[][]) {
  downloadBlob(new Blob([toCsv(header, rows)], { type: "text/csv;charset=utf-8" }), filename);
}

/** Имя файла из id и названия таблицы: только безопасные символы. */
export const fileName = (...parts: Array<string | number>) => parts.join("_").replace(/[^\w.@-]+/g, "-");
