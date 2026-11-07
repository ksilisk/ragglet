import * as Tooltip from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";
import s from "./kit.module.css";

/** Подсказка по наведению и фокусу: определения метрик, полные хэши. Провайдер — в main.tsx. */
export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className={s.tooltip} sideOffset={4} collisionPadding={8}>
          {content}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

/** Значок ⓘ с подсказкой — кнопка, чтобы до неё доходил фокус с клавиатуры. */
export function InfoTip({ content, label }: { content: ReactNode; label: string }) {
  return (
    <Tip content={content}>
      <button type="button" className={s.info} aria-label={label}>
        ⓘ
      </button>
    </Tip>
  );
}
