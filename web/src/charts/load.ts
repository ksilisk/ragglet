import type { EChartsOption } from "echarts";
import type { ChartTheme } from "./theme";

type Lib = typeof import("./echarts");
let loading: Promise<Lib> | null = null;

/** ECharts грузится один раз, при первом графике: плитки и таблицы не ждут его. */
export function loadECharts(): Promise<Lib> {
  loading ??= import("./echarts");
  return loading;
}

/** Опция графика — чистая функция данных и темы (charts/options): её же зовёт экспорт со светлой темой. */
export type Build<D> = (data: D, theme: ChartTheme) => EChartsOption;
