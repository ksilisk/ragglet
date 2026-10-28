import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Link, type LinkProps } from "react-router";
import { cx } from "./cx";
import s from "./kit.module.css";

type Variant = "default" | "primary" | "danger";

interface Look {
  variant?: Variant;
  small?: boolean;
}

function classes({ variant = "default", small }: Look, extra?: string) {
  return cx(s.btn, variant === "primary" && s.pri, variant === "danger" && s.danger, small && s.sm, extra);
}

export function Button({ variant, small, className, type = "button", ...rest }: Look & ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type={type} className={classes({ variant, small }, className)} {...rest} />;
}

export function LinkButton({ variant, small, className, ...rest }: Look & LinkProps) {
  return <Link className={classes({ variant, small }, className)} {...rest} />;
}

/** Ссылка на файл API (скачивание): обычный <a>, роутер его не перехватывает. */
export function FileButton({ href, children, small }: { href: string; children: ReactNode; small?: boolean }) {
  return (
    <a className={classes({ small })} href={href} download>
      {children}
    </a>
  );
}
