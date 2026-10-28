// localStorage для удобств одного зрителя (тема, язык): бывает недоступен — тогда просто умолчание.
export function readPref(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writePref(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // приватный режим или запрет хранилища: настройка живёт до перезагрузки
  }
}
