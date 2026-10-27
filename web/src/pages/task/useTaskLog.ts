import { useEffect, useRef, useState } from "react";
import { api, unwrap, type S } from "../../api/client";
import { POLL } from "../../api/queries";
import { stripAnsi } from "../../lib/ansi";

/** В памяти держим не больше этого: начало длинного лога обрезается, полностью он в task.log. */
export const LOG_CAP = 1 << 20;

export interface TaskLog {
  text: string;
  truncated: boolean;
  complete: boolean;
  error: unknown;
}

/**
 * Лог задачи по смещению (ragglet-28 §9): дописываем next_offset за next_offset; у активной задачи —
 * раз в 2 с, у завершённой — подряд до complete. ANSI вырезаются (заметка ragglet-35).
 */
export function useTaskLog(kind: S["TaskKind"], id: string, active: boolean): TaskLog {
  const [state, setState] = useState<TaskLog>({ text: "", truncated: false, complete: false, error: null });
  const offset = useRef(0);
  const activeRef = useRef(active);
  activeRef.current = active;

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    offset.current = 0;
    setState({ text: "", truncated: false, complete: false, error: null });

    const tick = async () => {
      try {
        const chunk =
          kind === "run"
            ? await unwrap(api.GET("/v1/runs/{run_id}/log", { params: { path: { run_id: id }, query: { offset: offset.current } } }))
            : await unwrap(api.GET("/v1/builds/{build_id}/log", { params: { path: { build_id: id }, query: { offset: offset.current } } }));
        if (stopped) return;
        offset.current = chunk.next_offset;
        setState((prev) => {
          let text = prev.text + stripAnsi(chunk.text);
          let truncated = prev.truncated;
          if (text.length > LOG_CAP) {
            text = text.slice(text.length - LOG_CAP);
            text = text.slice(text.indexOf("\n") + 1);
            truncated = true;
          }
          return { text, truncated, complete: chunk.complete, error: null };
        });
        if (chunk.complete) return; // задача завершена и лог дочитан
        // пришёл кусок — читаем дальше сразу (у завершённой задачи или длинного вывода), пусто — ждём новых строк
        const more = chunk.text.length > 0 && (!activeRef.current || chunk.text.length >= 32 * 1024);
        timer = window.setTimeout(tick, more ? 0 : POLL.log);
      } catch (error) {
        if (stopped) return;
        setState((prev) => ({ ...prev, error }));
        timer = window.setTimeout(tick, POLL.log * 2);
      }
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [kind, id]);

  return state;
}
