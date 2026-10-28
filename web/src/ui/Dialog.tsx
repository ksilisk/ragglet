import * as RadixDialog from "@radix-ui/react-dialog";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { useT } from "../i18n/i18n";
import { Button } from "./Button";
import { cx } from "./cx";
import s from "./kit.module.css";

export function Dialog({ open, onOpenChange, title, children, footer, wide }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className={s.overlay} />
        <RadixDialog.Content className={cx(s.dlg, wide && s.wide)} aria-describedby={undefined}>
          <RadixDialog.Title asChild>
            <h2>{title}</h2>
          </RadixDialog.Title>
          {children && <div className={s.dlgBody}>{children}</div>}
          {footer && <div className={s.dlgFoot}>{footer}</div>}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}

interface ConfirmOptions {
  title: ReactNode;
  body?: ReactNode;
  confirm: string;
  cancel?: string;
  danger?: boolean;
  wide?: boolean;
}

type Confirm = (options: ConfirmOptions) => Promise<boolean>;

const ConfirmContext = createContext<Confirm | null>(null);

/** Диалог подтверждения как Promise: `if (await confirm({...})) …`. */
export function ConfirmProvider({ children }: { children: ReactNode }) {
  const t = useT();
  const [options, setOptions] = useState<ConfirmOptions | null>(null);
  const resolver = useRef<((ok: boolean) => void) | null>(null);

  const confirm = useCallback<Confirm>((next) => {
    resolver.current?.(false);
    setOptions(next);
    return new Promise<boolean>((resolve) => {
      resolver.current = resolve;
    });
  }, []);

  const close = (ok: boolean) => {
    resolver.current?.(ok);
    resolver.current = null;
    setOptions(null);
  };

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      <Dialog
        open={options !== null}
        onOpenChange={(open) => !open && close(false)}
        title={options?.title}
        wide={options?.wide}
        footer={
          <>
            <Button onClick={() => close(false)}>{options?.cancel ?? t("Отмена")}</Button>
            <Button variant={options?.danger ? "danger" : "primary"} onClick={() => close(true)} autoFocus>
              {options?.confirm}
            </Button>
          </>
        }
      >
        {options?.body}
      </Dialog>
    </ConfirmContext.Provider>
  );
}

export function useConfirm(): Confirm {
  const confirm = useContext(ConfirmContext);
  if (!confirm) throw new Error("useConfirm outside ConfirmProvider");
  return confirm;
}
