import type { ReactNode } from "react";
import { cx } from "./cx";
import s from "./kit.module.css";

export interface TabItem<T extends string> {
  id: T;
  label: ReactNode;
  disabled?: boolean;
  title?: string;
}

export function Tabs<T extends string>({ items, value, onChange, label }: {
  items: Array<TabItem<T>>;
  value: T;
  onChange: (id: T) => void;
  label: string;
}) {
  return (
    <div className={s.tabs} role="tablist" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={item.id === value}
          aria-disabled={item.disabled || undefined}
          title={item.title}
          className={cx(s.tab, item.id === value && s.on, item.disabled && s.off)}
          onClick={() => !item.disabled && onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
