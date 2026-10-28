import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { S } from "../../api/client";
import { useT } from "../../i18n/i18n";
import { Card } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";
import { cx } from "../../ui/cx";
import s from "./task.module.css";
import { useTaskLog } from "./useTaskLog";

/** Лог задачи: автопрокрутка, пока пользователь внизу; у упавшей — трейсбек видно сразу. */
export function LogView({ kind, id, active, tall, downloadHref }: { kind: S["TaskKind"]; id: string; active: boolean; tall?: boolean; downloadHref?: string }) {
  const t = useT();
  const log = useTaskLog(kind, id, active);
  const box = useRef<HTMLPreElement>(null);
  const [follow, setFollow] = useState(true);
  const [paused, setPaused] = useState(false);

  useLayoutEffect(() => {
    const el = box.current;
    if (el && follow && !paused) el.scrollTop = el.scrollHeight;
  }, [log.text, follow, paused]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onScroll = () => setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <Card
      title={t("Лог")}
      aside={
        <span className="small muted">
          {active && (
            <>
              {t("автопрокрутка")} ·{" "}
              <button type="button" onClick={() => setPaused((p) => !p)} style={{ all: "unset", cursor: "pointer", color: "var(--accent)" }}>
                {paused ? t("продолжить") : t("пауза")}
              </button>
              {downloadHref && " · "}
            </>
          )}
          {downloadHref && (
            <a href={downloadHref} download>
              {t("скачать")}
            </a>
          )}
        </span>
      }
    >
      {log.truncated && <span className="small muted">{t("Начало лога обрезано (больше 1 МиБ) — полностью он в файле task.log.")}</span>}
      <pre ref={box} className={cx(s.log, tall && s.tall)} aria-label={t("Лог задачи")} tabIndex={0}>
        {log.text ? <Lines text={log.text} /> : <span className="muted">{log.complete ? t("Лог пуст.") : t("Ждём вывода задачи…")}</span>}
      </pre>
      {log.error !== null && <ErrorBox error={log.error} />}
    </Card>
  );
}

const WARN = /\b(warning|warn|WARNING)\b/;
const ERR = /\b(Traceback|Error|ERROR|Exception)\b/;

function Lines({ text }: { text: string }) {
  return (
    <>
      {text.split("\n").map((line, i, all) => (
        <span key={i} className={ERR.test(line) ? s.logErr : WARN.test(line) ? s.logWarn : undefined}>
          {line}
          {i < all.length - 1 ? "\n" : ""}
        </span>
      ))}
    </>
  );
}
