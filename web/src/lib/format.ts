// Числа и даты — по-русски (ru-RU): «0,331», «9 696». Времена с сервера в UTC, показываются локально.

const formats = new Map<number, Intl.NumberFormat>();

function numberFormat(digits: number): Intl.NumberFormat {
  let f = formats.get(digits);
  if (!f) {
    f = new Intl.NumberFormat("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
    formats.set(digits, f);
  }
  return f;
}

export const DASH = "—";
/** Неразрывный пробел между числом и единицей: «2,2 %», «16 ГБ». */
export const NBSP = "\u00a0";

export function fmtNumber(x: number | null | undefined, digits = 0): string {
  return x === null || x === undefined || Number.isNaN(x) ? DASH : numberFormat(digits).format(x);
}

/** Метрики качества: три знака. */
export const fmtMetric = (x: number | null | undefined) => fmtNumber(x, 3);
/** Задержки, мс: целые, меньше 10 — с одним знаком. */
export const fmtMs = (x: number | null | undefined) => fmtNumber(x, x !== null && x !== undefined && Math.abs(x) < 10 ? 1 : 0);
export const fmtQps = (x: number | null | undefined) => fmtNumber(x, 1);

export function fmtPercent(rate: number | null | undefined): string {
  if (rate === null || rate === undefined) return DASH;
  const pct = rate * 100;
  return `${fmtNumber(pct, pct !== 0 && Math.abs(pct) < 10 ? 1 : 0)}${NBSP}%`;
}

export function toDate(value: string | Date | null | undefined): Date | null {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

const pad = (n: number) => String(n).padStart(2, "0");

export function fmtTime(value: string | Date | null | undefined, seconds = true): string {
  const d = toDate(value);
  if (!d) return DASH;
  return `${pad(d.getHours())}:${pad(d.getMinutes())}${seconds ? `:${pad(d.getSeconds())}` : ""}`;
}

/** «сегодня 17:15», «24.10 18:09», «24.10.2025 18:09». */
export function fmtWhen(value: string | Date | null | undefined, now: Date = new Date()): string {
  const d = toDate(value);
  if (!d) return DASH;
  const time = fmtTime(d, false);
  if (d.toDateString() === now.toDateString()) return `сегодня ${time}`;
  const date = `${pad(d.getDate())}.${pad(d.getMonth() + 1)}`;
  return d.getFullYear() === now.getFullYear() ? `${date} ${time}` : `${date}.${d.getFullYear()} ${time}`;
}

/** Длительность в секундах: «0:41», «2:03», «1:02:03». */
export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return DASH;
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return DASH;
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${fmtNumber(value, unit === 0 || value >= 100 ? 0 : 1)}${NBSP}${units[unit]}`;
}

/** «hf:BeIR/nfcorpus@b5026a0e96e8+a451b3b26d3a» → «hf · BeIR/nfcorpus» и ревизия отдельно. */
export function splitDataset(label: string | null | undefined): { source: string; id: string; revision: string | null } {
  if (!label) return { source: "", id: DASH, revision: null };
  const colon = label.indexOf(":");
  const source = colon > 0 ? label.slice(0, colon) : "";
  const rest = colon > 0 ? label.slice(colon + 1) : label;
  const at = rest.indexOf("@");
  return { source, id: at >= 0 ? rest.slice(0, at) : rest, revision: at >= 0 ? rest.slice(at + 1) : null };
}

export function fmtDataset(label: string | null | undefined): string {
  const { source, id } = splitDataset(label);
  return source ? `${source} · ${id}` : id;
}

export function plural(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}
