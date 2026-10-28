import type { EChartsOption, ScatterSeriesOption } from "echarts";
import type { TimelinePoint } from "../../analysis/perQuery";
import { translate } from "../../i18n/i18n";
import { fmtMs, fmtNumber } from "../../lib/format";
import { outcomeStyle } from "../palette";
import type { ChartTheme } from "../theme";
import { asParams, base, esc, legend, tooltip, valueAxis } from "./common";
import { axisMs } from "./stageBar";

/**
 * Линия времени повтора (ragglet-32 §6): точка — запрос, x — старт от начала повтора, y — общая
 * задержка, цвет и форма — исход или отметка кэша. Видны прогрев, кэш и где копятся таймауты.
 */
export interface TimelineData {
  points: TimelinePoint[];
  groups: string[]; // порядок легенды: ok, timeout, error или hit, miss, none
  budgetMs: number | null;
}

/** Отметки кэша — тем же правилом, что исходы: попадание — как ok, промах — как таймаут. */
function groupStyle(theme: ChartTheme, group: string) {
  if (group === "hit") return { color: theme.series[0]!, symbol: "circle" };
  if (group === "miss") return { color: theme.warning, symbol: "triangle" };
  if (group === "none") return { color: theme.faint, symbol: "rect" };
  return outcomeStyle(theme, group);
}

export function timelineOption(d: TimelineData, theme: ChartTheme): EChartsOption {
  const series: ScatterSeriesOption[] = d.groups.map((group) => {
    const style = groupStyle(theme, group);
    return {
      type: "scatter",
      name: group,
      data: d.points.filter((p) => p.group === group).map((p) => [p.t, p.ms, p.qid]),
      symbol: style.symbol,
      symbolSize: group === "ok" || group === "hit" ? 5 : 8,
      itemStyle: { color: style.color, opacity: group === "ok" ? 0.6 : 0.9 },
      large: d.points.length > 3000,
      emphasis: { focus: "series" },
    };
  });
  // как у CDF: бюджет — линия, только если запросы к нему подходят
  const slowest = Math.max(0, ...d.points.map((p) => p.ms));
  if (d.budgetMs !== null && slowest >= d.budgetMs * 0.5 && series[0]) {
    series[0].markLine = {
      silent: true,
      symbol: "none",
      lineStyle: { color: theme.warning, type: "dashed", width: 1 },
      label: { formatter: translate("бюджет {ms} мс", { ms: fmtMs(d.budgetMs) }), color: theme.muted, fontSize: 11, position: "insideEndTop" },
      data: [{ yAxis: d.budgetMs }],
    };
  }
  return {
    ...base(theme),
    animation: false,
    grid: { left: 0, right: 24, top: 16, bottom: 72, containLabel: true },
    xAxis: valueAxis(theme, { name: translate("с от начала повтора"), nameLocation: "middle", nameGap: 26, min: 0, axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: (v: number) => fmtNumber(v, Number.isInteger(v) ? 0 : 1) } }),
    yAxis: valueAxis(theme, { name: translate("мс"), nameLocation: "end", min: 0, axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: axisMs } }),
    dataZoom: [
      { type: "inside", xAxisIndex: 0, filterMode: "none" },
      { type: "slider", xAxisIndex: 0, height: 16, bottom: 28, borderColor: theme.axis, textStyle: { color: theme.muted, fontSize: 10 }, filterMode: "none" },
    ],
    legend: legend(theme, { data: d.groups }),
    tooltip: tooltip(theme, {
      trigger: "item",
      formatter: (raw) => {
        const p = asParams(raw)[0]!;
        const [t, ms, qid] = p.value as [number, number, string];
        const latency = translate("{ms} мс", { ms: fmtMs(ms) });
        const start = translate("старт {t} с", { t: fmtNumber(t, 2) });
        return `<b>${esc(qid)}</b><br/>${esc(p.seriesName ?? "")}: ${esc(latency)}<br/>${esc(start)}`;
      },
    }),
    series,
  };
}
