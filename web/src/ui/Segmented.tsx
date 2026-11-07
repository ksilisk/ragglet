import { cx } from "./cx";
import s from "./kit.module.css";

/** Переключатель из двух-трёх вариантов: «линии | столбики», «doc | chunk». */
export function Segmented<T extends string>({ items, value, onChange, label }: {
  items: Array<{ id: T; label: string }>;
  value: T;
  onChange: (id: T) => void;
  label: string;
}) {
  return (
    <div className={s.seg} role="group" aria-label={label}>
      {items.map((item) => (
        <button key={item.id} type="button" className={cx(s.segItem, item.id === value && s.on)} aria-pressed={item.id === value} onClick={() => onChange(item.id)}>
          {item.label}
        </button>
      ))}
    </div>
  );
}
