import type { S } from "../api/client";
import { msg, useT } from "../i18n/i18n";
import { Chip, type ChipTone } from "./Chip";

type TaskState = S["TaskState"];

/** Подписи — msg(): переводятся при выводе, t(STATE[s].label). */
export const STATE: Record<TaskState, { label: string; tone: ChipTone }> = {
  queued: { label: msg("В очереди"), tone: "q" },
  running: { label: msg("Выполняется"), tone: "run" },
  completed: { label: msg("Готово"), tone: "ok" },
  failed: { label: msg("Ошибка"), tone: "fail" },
  cancelled: { label: msg("Отменён"), tone: "cancel" },
};

export const ACTIVE: ReadonlySet<TaskState> = new Set(["queued", "running"]);

export function StateChip({ state, suffix }: { state: TaskState; suffix?: string }) {
  const t = useT();
  const { label, tone } = STATE[state];
  return <Chip tone={tone}>{suffix ? `${t(label)} · ${suffix}` : t(label)}</Chip>;
}

/** msg(): при выводе — t(KIND_LABEL[kind]). */
export const KIND_LABEL: Record<S["TaskKind"], string> = { run: msg("Прогон"), build: msg("Сборка") };

export function taskHref(kind: S["TaskKind"], id: string): string {
  return kind === "run" ? `/runs/${encodeURIComponent(id)}` : `/builds/${encodeURIComponent(id)}`;
}
