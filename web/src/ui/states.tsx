import type { S } from "../api/client";
import { Chip, type ChipTone } from "./Chip";

type TaskState = S["TaskState"];

export const STATE: Record<TaskState, { label: string; tone: ChipTone }> = {
  queued: { label: "В очереди", tone: "q" },
  running: { label: "Выполняется", tone: "run" },
  completed: { label: "Готово", tone: "ok" },
  failed: { label: "Ошибка", tone: "fail" },
  cancelled: { label: "Отменён", tone: "cancel" },
};

export const ACTIVE: ReadonlySet<TaskState> = new Set(["queued", "running"]);

export function StateChip({ state, suffix }: { state: TaskState; suffix?: string }) {
  const { label, tone } = STATE[state];
  return <Chip tone={tone}>{suffix ? `${label} · ${suffix}` : label}</Chip>;
}

export const KIND_LABEL: Record<S["TaskKind"], string> = { run: "Прогон", build: "Сборка" };

export function taskHref(kind: S["TaskKind"], id: string): string {
  return kind === "run" ? `/runs/${encodeURIComponent(id)}` : `/builds/${encodeURIComponent(id)}`;
}
