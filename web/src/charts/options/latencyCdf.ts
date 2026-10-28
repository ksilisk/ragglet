import type { EChartsOption, LineSeriesOption } from "echarts";
import { translate } from "../../i18n/i18n";
import { fmtMs, fmtPercent } from "../../lib/format";
import { slotColor, slotDash } from "../palette";
import type { ChartTheme } from "../theme";
import { asParams, base, legend, tipRow, tipTitle, tooltip, valueAxis } from "./common";
import { axisMs } from "./stageBar";

/**
 * Распределение lat_total_ms — CDF (ragglet-32 §6): повторы наложением без перекрытия столбиков,
 * перцентиль читается по оси y, ступенька таймаутов у бюджета видна сразу.
 */
export interface CdfSeries {
  name: string;
  slot: number;
  points: Array<[number, number]>;
}

export interface LatencyCdfData {
  series: CdfSeries[];
  log: boolean;
  budgetMs: number | null; // timeouts.total_ms
}

export function latencyCdfOption(d: LatencyCdfData, theme: ChartTheme): EChartsOption {
  const slowest = Math.max(0, ...d.series.flatMap((s) => s.points.map(([x]) => x)));
  // бюджет — линия, только если запросы к нему подходят: иначе он растянул бы ось и сжал кривые
  const budget = d.budgetMs !== null && slowest >= d.budgetMs * 0.5 ? d.budgetMs : null;
  const series: LineSeriesOption[] = d.series.map((s, i) => ({
    type: "line",
    name: s.name,
    data: s.points,
    step: "end",
    showSymbol: false,
    color: slotColor(theme, s.slot),
    lineStyle: { width: 2, type: slotDash(s.slot) },
    emphasis: { focus: "series" },
    markLine:
      i === 0 && budget !== null
        ? {
            silent: true,
            symbol: "none",
            lineStyle: { color: theme.warning, type: "dashed", width: 1 },
            label: { formatter: translate("бюджет {ms} мс", { ms: fmtMs(budget) }), color: theme.muted, fontSize: 11, position: "insideEndTop" },
            data: [{ xAxis: budget }],
          }
        : undefined,
  }));
  const maxX = Math.max(1, slowest, budget ?? 0);
  return {
    ...base(theme),
    grid: { left: 0, right: 24, top: 16, bottom: 44, containLabel: true },
    xAxis: valueAxis(theme, {
      type: d.log ? "log" : "value",
      name: translate("мс"),
      nameLocation: "end",
      nameGap: 6,
      min: d.log ? undefined : 0,
      max: d.log ? undefined : Math.ceil(maxX * 1.05),
      axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: axisMs },
    }),
    yAxis: valueAxis(theme, {
      min: 0,
      max: 1,
      interval: 0.25,
      axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: (v: number) => fmtPercent(v) },
    }),
    legend: legend(theme, { data: d.series.map((s) => s.name) }),
    tooltip: tooltip(theme, {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: theme.axis } },
      formatter: (raw) => {
        const params = asParams(raw);
        const rows = params.map((p) => {
          const [x, y] = p.value as [number, number];
          return tipRow(p.marker ?? "", p.seriesName ?? "", translate("{share} не медленнее {ms} мс", { share: fmtPercent(y), ms: fmtMs(x) }));
        });
        return tipTitle("lat_total_ms") + rows.join("");
      },
    }),
    series,
  };
}
