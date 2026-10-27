import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import type { ReactNode } from "react";
import { cx } from "./cx";
import s from "./kit.module.css";

export interface MenuItem {
  label: ReactNode;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
}

/** Меню «⋯» и «Новый ▾»: Radix даёт фокус и клавиатуру, вид — из артбордов. */
export function Menu({ trigger, items, align = "end" }: { trigger: ReactNode; items: Array<MenuItem | null | false>; align?: "start" | "end" }) {
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={s.menu} align={align} sideOffset={4}>
          {items.filter((x): x is MenuItem => Boolean(x)).map((item, i) => (
            <DropdownMenu.Item key={i} className={cx(s.menuItem, item.danger && s.danger)} disabled={item.disabled} onSelect={item.onSelect}>
              {item.label}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
