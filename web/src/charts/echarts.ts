// ECharts по модулям (ragglet-32 §1): только нужные графики и компоненты, отдельный ленивый чанк.
// Canvas — на экране (тысячи точек линии времени), SVG — для экспорта.
import { BarChart, CustomChart, LineChart, ScatterChart } from "echarts/charts";
import { DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer, SVGRenderer } from "echarts/renderers";

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  MarkAreaComponent,
  DataZoomComponent,
  CanvasRenderer,
  SVGRenderer,
]);

export { echarts };
