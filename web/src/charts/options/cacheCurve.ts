import type { EChartsOption, LineSeriesOption } from "echarts";
import { translate } from "../../i18n/i18n";
import { fmtNumber, fmtPercent } from "../../lib/format";
import { slotColor, slotDash } from "../palette";
import type { ChartTheme } from "../theme";
import { asParams, base, legend, tipRow, tooltip, valueAxis } from "./common";

/**
 * Накопительная доля попаданий кэша по ходу повтора (решение 4 ревью): после каждого обращения —
 * hits / (hits + misses) с начала повтора; последняя точка — <слой>_hit_rate из summary.
 */
export interface CacheCurveData {
  series: Array<{ name: string; slot: number; points: Array<[number, number]> }>;
}

export function cacheCurveOption(d: CacheCurveData, theme: ChartTheme): EChartsOption {
  const series: LineSeriesOption[] = d.series.map((s) => ({
    type: "line",
    name: s.name,
    data: s.points,
    showSymbol: false,
    color: slotColor(theme, s.slot),
    lineStyle: { width: 2, type: slotDash(s.slot) },
  }));
  return {
    ...base(theme),
    grid: { left: 0, right: 24, top: 16, bottom: 44, containLabel: true },
    xAxis: valueAxis(theme, { name: translate("с"), nameLocation: "end", nameGap: 6, min: 0, axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: (v: number) => fmtNumber(v, Number.isInteger(v) ? 0 : 1) } }),
    yAxis: valueAxis(theme, { min: 0, max: 1, interval: 0.25, axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: (v: number) => fmtPercent(v) } }),
    legend: legend(theme, { data: d.series.map((s) => s.name) }),
    tooltip: tooltip(theme, {
      trigger: "axis",
      formatter: (raw) =>
        asParams(raw)
          .map((p) => {
            const [t, rate] = p.value as [number, number];
            return tipRow(p.marker ?? "", p.seriesName ?? "", translate("{rate} к {t} с", { rate: fmtPercent(rate), t: fmtNumber(t, 1) }));
          })
          .join(""),
    }),
    series,
  };
}
