import type { ECharts } from "echarts/core";
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { useT } from "../i18n/i18n";
import s from "./charts.module.css";
import { exportChart, type ExportFormat } from "./exportChart";
import { loadECharts, type Build } from "./load";
import { useChartTheme } from "./theme";

export { loadECharts, type Build };

type Lib = typeof import("./echarts");

/** Ручка экспорта: карточка графика держит её и показывает меню «SVG / PNG» у себя в шапке. */
export interface ChartExport {
  download(format: ExportFormat): Promise<void>;
}

export interface ChartProps<D> {
  build: Build<D>;
  data: D; // мемоизированные данные: новая ссылка — новая опция
  height: number;
  label: string; // подпись для экранного диктора
  exportRef?: RefObject<ChartExport | null>;
  exportName?: string; // имя файла без расширения
}

export function Chart<D>({ build, data, height, label, exportRef, exportName }: ChartProps<D>) {
  const t = useT();
  const theme = useChartTheme();
  const option = useMemo(() => build(data, theme), [build, data, theme]);
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<ECharts | null>(null);
  const [lib, setLib] = useState<Lib | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    loadECharts().then(
      (m) => alive && setLib(m),
      () => alive && setFailed(true),
    );
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const el = box.current;
    if (!lib || !el) return;
    const instance = lib.echarts.init(el, null, { renderer: "canvas" });
    chart.current = instance;
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(el);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, [lib]);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true });
  }, [option, lib]);

  useEffect(() => {
    if (!exportRef) return;
    exportRef.current = {
      download: (format) => exportChart(build, data, theme.lang, format, { width: box.current?.clientWidth ?? 900, height }, exportName ?? "chart"),
    };
    return () => {
      exportRef.current = null;
    };
  }, [exportRef, build, data, theme.lang, height, exportName]);

  return (
    <div className={s.chart} style={{ height }} role="img" aria-label={label}>
      <div ref={box} className={s.canvas} />
      {!lib && <span className={s.loading}>{failed ? t("График не загрузился: перезагрузите страницу.") : t("Загрузка графика…")}</span>}
    </div>
  );
}
