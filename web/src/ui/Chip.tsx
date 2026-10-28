import type { ReactNode } from "react";
import { cx } from "./cx";
import s from "./kit.module.css";

export type ChipTone = "run" | "q" | "ok" | "fail" | "cancel" | "warn" | "tag";

export function Chip({ tone, children, title }: { tone: ChipTone; children: ReactNode; title?: string }) {
  return (
    <span className={cx(s.chip, s[`c-${tone}`])} title={title}>
      {children}
    </span>
  );
}

export function Tag({ children }: { children: ReactNode }) {
  return <Chip tone="tag">{children}</Chip>;
}

/** Чип конвейера в списке сценариев (.p, .p.on). */
export function PipeChip({ on, children }: { on?: boolean; children: ReactNode }) {
  return <span className={cx(s.p, on && s.on)}>{children}</span>;
}

/** Фильтр-чип над таблицей (.f, .f.on). */
export function FilterChip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" className={cx(s.f, on && s.on)} aria-pressed={on} onClick={onClick}>
      {children}
    </button>
  );
}

export type DotTone = "ok" | "busy" | "down" | "warn" | "idle";

export function Dot({ tone = "ok", title }: { tone?: DotTone; title?: string }) {
  return (
    <span
      className={cx(s.dot, tone === "busy" && s.busy, tone === "down" && s.down, tone === "warn" && s.warn, tone === "idle" && s.idle)}
      title={title}
      aria-hidden={title ? undefined : true}
    />
  );
}
