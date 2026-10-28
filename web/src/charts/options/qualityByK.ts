import type { BarSeriesOption, CustomSeriesOption, EChartsOption, LineSeriesOption } from "echarts";
import { fmtMetric, fmtNumber } from "../../lib/format";
import { slotColor, slotDash, slotSymbol } from "../palette";
import type { ChartTheme } from "../theme";
import { asParams, base, categoryAxis, legend, tipRow, tipTitle, tooltip, valueAxis } from "./common";

/**
 * Качество по k: серия — метрика («Обзор», «Качество») или прогон («Сравнение»), x — k категориями,
 * шкала 0–1, усы — ± std по повторам. Линии или столбики — решается после просмотра (решение 5
 * ревью), поэтому построитель умеет оба вида.
 */
export type QualityMode = "lines" | "bars";

export interface QualityPoint {
  k: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

export interface QualitySeries {
  name: string;
  slot: number;
  points: QualityPoint[];
}

export interface QualityByKData {
  ks: number[];
  series: QualitySeries[];
  mode: QualityMode;
}

const BAR_GAP = "15%";
const CATEGORY_GAP = "35%";
const clamp = (x: number) => Math.min(1, Math.max(0, x));

export function qualityByKOption(d: QualityByKData, theme: ChartTheme): EChartsOption {
  const at = (s: QualitySeries, k: number) => s.points.find((p) => p.k === k);
  const bars = d.mode === "bars";

  const main: Array<LineSeriesOption | BarSeriesOption> = d.series.map((s) =>
    bars
      ? {
          type: "bar",
          name: s.name,
          data: d.ks.map((k) => at(s, k)?.mean ?? null),
          color: slotColor(theme, s.slot),
          barGap: BAR_GAP,
          barCategoryGap: CATEGORY_GAP,
          itemStyle: { borderRadius: [3, 3, 0, 0] },
        }
      : {
          type: "line",
          name: s.name,
          data: d.ks.map((k) => at(s, k)?.mean ?? null),
          color: slotColor(theme, s.slot),
          symbol: slotSymbol(s.slot),
          symbolSize: 8,
          lineStyle: { width: 2, type: slotDash(s.slot) },
          z: 3,
        },
  );

  // усы: отдельная custom-серия с тем же именем — легенда скрывает их вместе с метрикой
  const whiskers: CustomSeriesOption[] = d.series.map((s, i) => ({
    type: "custom",
    name: s.name,
    color: slotColor(theme, s.slot),
    silent: true,
    z: 4,
    encode: { x: 0, y: [1, 2] },
    data: d.ks.flatMap((k, xi) => {
      const p = at(s, k);
      return p && p.mean !== null && p.std !== null && p.std > 0 ? [[xi, clamp(p.mean - p.std), clamp(p.mean + p.std)]] : [];
    }),
    renderItem: (_params, api) => {
      const xi = api.value(0) as number;
      const top = api.coord([xi, api.value(2)]);
      const bottom = api.coord([xi, api.value(1)]);
      let x = top[0]!;
      if (bars) {
        const layout = (api as unknown as { barLayout: (o: object) => Array<{ offsetCenter: number }> }).barLayout({
          count: d.series.length,
          barGap: BAR_GAP,
          barCategoryGap: CATEGORY_GAP,
        });
        x += layout[i]?.offsetCenter ?? 0;
      }
      const style = { stroke: bars ? theme.text : slotColor(theme, s.slot), lineWidth: 1.5 };
      const cap = 4;
      return {
        type: "group",
        children: [
          { type: "line", shape: { x1: x, y1: top[1]!, x2: x, y2: bottom[1]! }, style },
          { type: "line", shape: { x1: x - cap, y1: top[1]!, x2: x + cap, y2: top[1]! }, style },
          { type: "line", shape: { x1: x - cap, y1: bottom[1]!, x2: x + cap, y2: bottom[1]! }, style },
        ],
      };
    },
  }));

  return {
    ...base(theme),
    grid: { left: 0, right: 16, top: 12, bottom: 40, containLabel: true },
    xAxis: categoryAxis(theme, d.ks.map(String), { name: "k", nameLocation: "end", nameGap: 8, boundaryGap: bars ? true : ["12%", "12%"] }),
    yAxis: valueAxis(theme, { min: 0, max: 1, interval: 0.2, axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: (v: number) => fmtNumber(v, 1) } }),
    legend: legend(theme, { data: d.series.map((s) => s.name) }),
    tooltip: tooltip(theme, {
      trigger: "axis",
      axisPointer: bars ? { type: "shadow", shadowStyle: { color: theme.grid, opacity: 0.6 } } : { type: "line", lineStyle: { color: theme.axis } },
      formatter: (raw) => {
        const params = asParams(raw).filter((p) => p.seriesType !== "custom");
        const xi = params[0]?.dataIndex ?? 0;
        const k = d.ks[xi];
        const rows = params.map((p) => {
          const s = d.series.find((x) => x.name === p.seriesName);
          const point = s && k !== undefined ? at(s, k) : undefined;
          const value = point?.mean === null || point?.mean === undefined ? "—" : fmtMetric(point.mean) + (point.std ? ` ± ${fmtMetric(point.std)}` : "");
          return tipRow(p.marker ?? "", p.seriesName ?? "", value);
        });
        return tipTitle(`@${k ?? ""}`) + rows.join("");
      },
    }),
    series: [...main, ...whiskers],
  };
}
