import type { EChartsOption, LegendComponentOption, TooltipComponentOption, XAXisComponentOption, YAXisComponentOption } from "echarts";
import type { ChartTheme } from "../theme";

// Общий хром графиков (ragglet-32 §2): тонкие оси, приглушённые подписи, сетка — --border-soft,
// числа — моноширинным шрифтом.

export function base(theme: ChartTheme): EChartsOption {
  return {
    backgroundColor: "transparent",
    animationDuration: 300,
    textStyle: { fontFamily: theme.fontSans, color: theme.text, fontSize: 12 },
  };
}

export function tooltip(theme: ChartTheme, extra: TooltipComponentOption = {}): TooltipComponentOption {
  return {
    backgroundColor: theme.tooltipBg,
    borderWidth: 0,
    padding: [6, 10],
    textStyle: { color: theme.tooltipText, fontSize: 12, fontFamily: theme.fontSans },
    extraCssText: "box-shadow: none; border-radius: 4px;",
    confine: true,
    ...extra,
  };
}

export function legend(theme: ChartTheme, extra: LegendComponentOption = {}): LegendComponentOption {
  return {
    bottom: 0,
    left: 0,
    itemWidth: 18,
    itemHeight: 10,
    itemGap: 16,
    textStyle: { color: theme.muted, fontSize: 12, fontFamily: theme.fontSans },
    inactiveColor: theme.axis,
    ...extra,
  };
}

const axisLabel = (theme: ChartTheme) => ({ color: theme.muted, fontFamily: theme.fontMono, fontSize: 11 });
const axisName = (theme: ChartTheme) => ({ color: theme.muted, fontFamily: theme.fontSans, fontSize: 11 });

export function valueAxis(theme: ChartTheme, extra: Record<string, unknown> = {}): XAXisComponentOption & YAXisComponentOption {
  return {
    type: "value",
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { lineStyle: { color: theme.grid } },
    axisLabel: axisLabel(theme),
    nameTextStyle: axisName(theme),
    ...extra,
  } as XAXisComponentOption & YAXisComponentOption;
}

export function categoryAxis(theme: ChartTheme, data: string[], extra: Record<string, unknown> = {}): XAXisComponentOption & YAXisComponentOption {
  return {
    type: "category",
    data,
    axisLine: { lineStyle: { color: theme.axis } },
    axisTick: { show: false },
    axisLabel: axisLabel(theme),
    nameTextStyle: axisName(theme),
    ...extra,
  } as XAXisComponentOption & YAXisComponentOption;
}

/** Строка подсказки: цветная метка серии, имя, значение. Имена из данных экранируются. */
export function tipRow(marker: string, name: string, value: string): string {
  return `<div style="display:flex;gap:12px;justify-content:space-between"><span>${marker}${esc(name)}</span><b style="font-weight:500">${esc(value)}</b></div>`;
}

export function tipTitle(text: string): string {
  return `<div style="margin-bottom:2px;opacity:.75">${esc(text)}</div>`;
}

export function esc(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

/** Параметр подсказки ECharts в том виде, в каком его читают форматтеры. */
export interface TipParam {
  seriesName?: string;
  seriesType?: string;
  seriesIndex?: number;
  dataIndex: number;
  name?: string;
  value: unknown;
  marker?: string;
  color?: string;
}

export const asParams = (p: unknown): TipParam[] => (Array.isArray(p) ? (p as TipParam[]) : [p as TipParam]);
