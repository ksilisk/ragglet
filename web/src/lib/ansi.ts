// CSI (цвета, курсор, очистка строки) и OSC-последовательности: transformers красит вывод и без tty
// (заметка ragglet-35), в логе задачи их быть не должно.
// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g;

export function stripAnsi(text: string): string {
  return text.replace(ANSI, "");
}
