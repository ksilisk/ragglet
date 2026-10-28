import type { EChartsOption } from "echarts";
import { downloadBlob } from "../analysis/csv";
import type { Lang } from "../i18n/i18n";
import { loadECharts, type Build } from "./load";
import { lightTheme } from "./theme";

/**
 * Экспорт графика (ragglet-32 §10, решение 3 ревью): всегда светлая тема на белом фоне, без анимации и
 * без ползунка масштаба, размер — как на экране. SVG — отдельный экземпляр с SVGRenderer (ssr), PNG —
 * canvas с двойной плотностью пикселей.
 */
export type ExportFormat = "svg" | "png";

export function exportOption<D>(build: Build<D>, data: D, lang: Lang): EChartsOption {
  const option = build(data, lightTheme(lang));
  const zoom = option.dataZoom;
  return {
    ...option,
    animation: false,
    backgroundColor: "#ffffff",
    // ползунок масштаба в картинке не нужен; масштаб «внутри» остаётся как есть
    dataZoom: Array.isArray(zoom) ? zoom.filter((z) => (z as { type?: string }).type !== "slider") : zoom,
  };
}

export async function exportChart<D>(build: Build<D>, data: D, lang: Lang, format: ExportFormat, size: { width: number; height: number }, name: string) {
  const { echarts } = await loadECharts();
  const option = exportOption(build, data, lang);
  const width = Math.max(320, Math.round(size.width));
  const height = Math.max(200, Math.round(size.height));
  if (format === "svg") {
    const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width, height });
    chart.setOption(option);
    const svg = chart.renderToSVGString();
    chart.dispose();
    downloadBlob(new Blob([svg], { type: "image/svg+xml" }), `${name}.svg`);
    return;
  }
  // canvas рисует только в документе: невидимый контейнер за краем экрана
  const box = document.createElement("div");
  box.style.cssText = `position:fixed;left:-10000px;top:0;width:${width}px;height:${height}px`;
  document.body.appendChild(box);
  try {
    const chart = echarts.init(box, null, { renderer: "canvas", width, height });
    chart.setOption(option);
    const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#ffffff" });
    chart.dispose();
    downloadBlob(await (await fetch(url)).blob(), `${name}.png`);
  } finally {
    box.remove();
  }
}
