// @vitest-environment node
import * as echarts from "echarts";
import nfcorpus from "../test/fixtures/summary-nfcorpus.json";
import { qualityByMetric } from "../analysis/series";
import type { Summary } from "../analysis/summary";
import { exportOption } from "./exportChart";
import { qualityByKOption } from "./options/qualityByK";
import { timelineOption } from "./options/timeline";
import { LIGHT } from "./theme";

// Экспорт всегда светлый (решение 3 ревью): проверяем опцию и настоящий SVG из ECharts (SSR).
describe("экспорт графиков", () => {
  const data = qualityByMetric(nfcorpus as unknown as Summary, "doc", "lines");

  it("светлая тема, белый фон, без анимации и ползунка масштаба", () => {
    const option = exportOption(qualityByKOption, data, "ru");
    expect(option.backgroundColor).toBe("#ffffff");
    expect(option.animation).toBe(false);
    const timeline = exportOption(timelineOption, { points: [{ t: 0, ms: 10, qid: "q", group: "ok" }], groups: ["ok"], budgetMs: null }, "ru");
    expect((timeline.dataZoom as Array<{ type: string }>).map((z) => z.type)).toEqual(["inside"]);
  });

  it("SVG собирается и несёт цвета светлой палитры", () => {
    const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width: 800, height: 300 });
    chart.setOption(exportOption(qualityByKOption, data, "ru"));
    const svg = chart.renderToSVGString();
    chart.dispose();
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain(LIGHT.series[0]); // nDCG — первый слот
    expect(svg).toContain("nDCG");
  });
});
