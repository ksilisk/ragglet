import type { CustomSeriesOption } from "echarts";
import nfcorpus from "../../test/fixtures/summary-nfcorpus.json";
import s0old from "../../test/fixtures/summary-s0-old.json";
import { qualityByMetric, stageRow } from "../../analysis/series";
import { latencyGrid, type Summary } from "../../analysis/summary";
import { stageRowsByRun } from "../../analysis/compare";
import type { S } from "../../api/client";
import { METRIC_SLOT } from "../palette";
import { LIGHT, lightTheme } from "../theme";
import { qualityByKOption } from "./qualityByK";
import { stageBarOption } from "./stageBar";

const one = nfcorpus as unknown as Summary;
const five = s0old as unknown as Summary;
const theme = lightTheme("ru");
const seriesOf = (o: { series?: unknown }) => o.series as Array<{ type: string; name: string; data: unknown[]; color?: string }>;

describe("качество по k", () => {
  it("линии: серия на метрику, усов нет при одном повторе", () => {
    const o = qualityByKOption(qualityByMetric(one, "doc", "lines"), theme);
    const series = seriesOf(o);
    const lines = series.filter((s) => s.type === "line");
    expect(lines.map((s) => s.name)).toEqual(["nDCG", "Recall", "R_cap", "Hit", "MRR"]);
    expect(lines[0]!.data).toHaveLength(4);
    expect(series.filter((s) => s.type === "custom").every((s) => s.data.length === 0)).toBe(true);
  });

  it("усы ± std при пяти повторах, обрезаны шкалой 0–1", () => {
    const o = qualityByKOption(qualityByMetric(five, "doc", "lines"), theme);
    const whiskers = seriesOf(o).filter((s) => s.type === "custom") as unknown as CustomSeriesOption[];
    const hit = whiskers.find((s) => s.name === "Hit")!;
    expect((hit.data as number[][]).length).toBeGreaterThan(0);
    for (const [, lo, hi] of hit.data as number[][]) {
      expect(lo).toBeGreaterThanOrEqual(0);
      expect(hi).toBeLessThanOrEqual(1);
    }
  });

  it("столбики: тот же набор серий", () => {
    const o = qualityByKOption(qualityByMetric(one, "doc", "bars"), theme);
    expect(seriesOf(o).filter((s) => s.type === "bar")).toHaveLength(5);
  });

  it("цвет закреплён за метрикой, а не за местом в списке", () => {
    // у s0 нет nDCG: Recall всё равно второго цвета, а не первого
    const series = seriesOf(qualityByKOption(qualityByMetric(five, "doc", "lines"), theme)).filter((s) => s.type === "line");
    expect(series[0]!.name).toBe("Recall");
    expect(series[0]!.color).toBe(LIGHT.series[METRIC_SLOT.recall]);
  });
});

describe("полоса стадий", () => {
  it.each([true, false])("merge остаётся в таблице, но не складывается с retrieve (retrieve: %s)", (hasRetrieve) => {
    const summary: Summary = {
      repeats: [{ lat_total_p50: 43, lat_merge_p50: 3, ...(hasRetrieve ? { lat_retrieve_p50: 43 } : {}) }],
      aggregate: {},
    };
    const grid = latencyGrid(summary);
    expect(grid.get("merge", 50)?.mean).toBe(3);
    const overview = { stages: grid.stages, rows: [stageRow(summary, "", 50)], percentile: 50 };
    const compare = { ...stageRowsByRun([{ id: "a", summary }, { id: "b", summary }] as S["CompareRun"][], 50, ["a", "b"]), percentile: 50 };
    for (const data of [overview, compare]) {
      const bars = seriesOf(stageBarOption(data, theme)).filter((s) => s.type === "bar");
      expect(bars.map((s) => s.name)).toEqual(hasRetrieve ? ["retrieve"] : []);
      if (hasRetrieve) expect(bars[0]!.data).toEqual(data.rows.map(() => 43));
    }
  });

  it("сегменты стадий в порядке конвейера и отметка total", () => {
    const row = stageRow(one, "", 50);
    const o = stageBarOption({ stages: ["embed", "retrieve"], rows: [row], percentile: 50 }, theme);
    const series = seriesOf(o);
    expect(series.map((s) => s.name)).toEqual(["embed", "retrieve", "total p50"]);
    expect(series[0]!.data[0]).toBeCloseTo(146, 0);
    expect((series[2]!.data[0] as number[])[0]).toBeCloseTo(152.5, 1);
  });
});
