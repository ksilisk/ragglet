import { useState } from "react";
import type { QualityMode } from "../charts/options/qualityByK";
import { readPref, writePref } from "../lib/storage";

/**
 * Вид графика «качество по k» — временный выбор на время ревью (решение 5): после выбора лишний
 * вариант и переключатель убираются.
 */
const KEY = "ragglet.qualityChart";

export function useQualityMode(): [QualityMode, (mode: QualityMode) => void] {
  const [mode, setMode] = useState<QualityMode>(() => (readPref(KEY) === "bars" ? "bars" : "lines"));
  return [
    mode,
    (next) => {
      writePref(KEY, next);
      setMode(next);
    },
  ];
}
