import type { EChartsOption } from "echarts";
import type { ECharts } from "echarts/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { useT } from "../i18n/i18n";
import s from "./charts.module.css";
import { useChartTheme, type ChartTheme } from "./theme";

type Lib = typeof import("./echarts");
let loading: Promise<Lib> | null = null;

/** ECharts грузится один раз, при первом графике: плитки и таблицы не ждут его. */
export function loadECharts(): Promise<Lib> {
  loading ??= import("./echarts");
  return loading;
}

/** Опция графика — чистая функция данных и темы (charts/options): её же зовёт экспорт со светлой темой. */
export type Build<D> = (data: D, theme: ChartTheme) => EChartsOption;

export interface ChartProps<D> {
  build: Build<D>;
  data: D; // мемоизированные данные: новая ссылка — новая опция
  height: number;
  label: string; // подпись для экранного диктора
}

export function Chart<D>({ build, data, height, label }: ChartProps<D>) {
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

  return (
    <div className={s.chart} style={{ height }} role="img" aria-label={label}>
      <div ref={box} className={s.canvas} />
      {!lib && <span className={s.loading}>{failed ? t("График не загрузился: перезагрузите страницу.") : t("Загрузка графика…")}</span>}
    </div>
  );
}
