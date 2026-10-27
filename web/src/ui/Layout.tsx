import type { ReactNode } from "react";
import { Link } from "react-router";
import { cx } from "./cx";
import s from "./kit.module.css";

export function Page({ children, gap }: { children: ReactNode; gap?: number }) {
  return (
    <main className={s.page} style={gap ? { gap } : undefined}>
      {children}
    </main>
  );
}

export function PageHead({ title, count, actions }: { title: ReactNode; count?: number; actions?: ReactNode }) {
  return (
    <div className={s.pageHead}>
      <h1>
        {title}
        {count !== undefined && <span className={s.count}> · {count}</span>}
      </h1>
      {actions && <div className={s.actions}>{actions}</div>}
    </div>
  );
}

export function Crumbs({ items }: { items: Array<{ label: ReactNode; to?: string }> }) {
  return (
    <div className={s.crumbs}>
      {items.map((item, i) => (
        <span key={i}>
          {i > 0 && " › "}
          {item.to ? <Link to={item.to}>{item.label}</Link> : item.label}
        </span>
      ))}
    </div>
  );
}

export function Card({ children, flush, className, title, aside, label }: {
  children: ReactNode;
  flush?: boolean;
  className?: string;
  title?: ReactNode;
  aside?: ReactNode;
  label?: string;
}) {
  return (
    <section className={cx(s.card, flush && s.flush, !flush && s.cardStack, className)} aria-label={label}>
      {(title || aside) && (
        <div className={s.cardHead}>
          {typeof title === "string" ? <h2>{title}</h2> : title}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export function Actions({ children }: { children: ReactNode }) {
  return <div className={s.actions}>{children}</div>;
}

export function KeyValue({ items }: { items: Array<[ReactNode, ReactNode] | null | false> }) {
  return (
    <dl className={s.kv}>
      {items.filter((x): x is [ReactNode, ReactNode] => Boolean(x)).map(([k, v], i) => (
        <div key={i} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ProgressBar({ fraction, big, label }: { fraction: number | null; big?: boolean; label: string }) {
  const pct = fraction === null ? null : Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  return (
    <div
      className={cx(s.bar, big && s.big, pct === null && s.indeterminate)}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
    >
      <div style={pct === null ? undefined : { width: `${pct}%` }} />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className={s.empty}>{children}</div>;
}
