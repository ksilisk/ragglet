import type { BarSeriesOption, EChartsOption, ScatterSeriesOption } from "echarts";
import { translate } from "../../i18n/i18n";
import { fmtMs, fmtNumber } from "../../lib/format";
import type { Stage } from "../../analysis/summary";
import { STAGE_SLOT, labelOn, slotColor } from "../palette";
import type { ChartTheme } from "../theme";
import { asParams, base, categoryAxis, legend, tipRow, tipTitle, tooltip, valueAxis } from "./common";

/**
 * Задержки по стадиям: строка — прогон (в «Обзоре» одна), сегменты — стадии в порядке конвейера,
 * отметка — total. Цвета стадий одни на всех экранах. Сумма медиан стадий — не медиана total:
 * это подписано под графиком.
 */
export interface StageBarRow {
  label: string;
  values: Partial<Record<Stage, number | null>>;
  total: number | null;
}

export interface StageBarData {
  stages: Stage[];
  rows: StageBarRow[];
  percentile: number;
}

/** Подписи оси: целые без дробной части, меньше 10 — с одним знаком. */
export const axisMs = (v: number) => (Number.isInteger(v) ? fmtNumber(v) : fmtMs(v));

export const stageBarHeight = (rows: number) => 64 + rows * 34;

export function stageBarOption(d: StageBarData, theme: ChartTheme): EChartsOption {
  const totalName = translate("total p{p}", { p: d.percentile });
  const widest = Math.max(1, ...d.rows.map((r) => Math.max(r.total ?? 0, d.stages.reduce((a, st) => a + (r.values[st] ?? 0), 0))));
  const many = d.rows.length > 1;

  const stages: BarSeriesOption[] = d.stages.map((stage) => ({
    type: "bar",
    name: stage,
    stack: "stages",
    barWidth: 22,
    data: d.rows.map((r) => r.values[stage] ?? null),
    itemStyle: { color: slotColor(theme, STAGE_SLOT[stage]), borderColor: theme.surface, borderWidth: 1 },
    label: {
      show: true,
      position: "inside",
      color: labelOn(STAGE_SLOT[stage]),
      fontFamily: theme.fontMono,
      fontSize: 11,
      // подпись — только если сегмент достаточно широк
      formatter: (p) => (typeof p.value === "number" && p.value / widest > 0.12 ? `${stage} ${fmtMs(p.value)}` : ""),
    },
    emphasis: { focus: "series" },
  }));

  const total: ScatterSeriesOption = {
    type: "scatter",
    name: totalName,
    data: d.rows.map((r, i) => (r.total === null ? null : [r.total, i])),
    symbol: "rect",
    symbolSize: [3, 30],
    itemStyle: { color: theme.text },
    label: {
      show: true,
      position: "right",
      color: theme.text,
      fontFamily: theme.fontMono,
      fontSize: 11,
      formatter: (p) => translate("{ms} мс", { ms: fmtMs((p.value as number[])[0]) }),
    },
    z: 5,
  };

  return {
    ...base(theme),
    grid: { left: many ? 8 : 0, right: 56, top: 8, bottom: 36, containLabel: true },
    xAxis: valueAxis(theme, { axisLabel: { color: theme.muted, fontFamily: theme.fontMono, fontSize: 11, formatter: axisMs } }),
    yAxis: categoryAxis(theme, d.rows.map((r) => r.label), { inverse: true, show: many }),
    legend: legend(theme, { data: [...d.stages, totalName] }),
    tooltip: tooltip(theme, {
      trigger: "axis",
      axisPointer: { type: "shadow", shadowStyle: { color: theme.grid, opacity: 0.6 } },
      formatter: (raw) => {
        const params = asParams(raw);
        const row = d.rows[params[0]?.dataIndex ?? 0];
        const lines = params
          .filter((p) => p.value !== null && p.value !== undefined)
          .map((p) => tipRow(p.marker ?? "", p.seriesName ?? "", translate("{ms} мс", { ms: fmtMs(Array.isArray(p.value) ? (p.value[0] as number) : (p.value as number)) })));
        return (many && row ? tipTitle(row.label) : "") + lines.join("");
      },
    }),
    series: [...stages, total],
  };
}
