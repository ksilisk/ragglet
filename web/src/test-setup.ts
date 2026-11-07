import "@testing-library/jest-dom/vitest";

// CodeMirror меряет текст; в jsdom у Range нет геометрии.
if (typeof Range !== "undefined") {
  Range.prototype.getClientRects = () => ({ length: 0, item: () => null, [Symbol.iterator]: [][Symbol.iterator] }) as unknown as DOMRectList;
  Range.prototype.getBoundingClientRect = () => new DOMRect(0, 0, 0, 0);
}

// Графики в jsdom не рисуются (нет canvas и ResizeObserver): вместо ECharts — фигура с числом серий,
// чтобы компонентные тесты видели, что график построен из данных. Опции проверяются отдельно.
vi.mock("./charts/Chart", async () => {
  const { createElement } = await import("react");
  const { lightTheme } = await import("./charts/theme");
  return {
    loadECharts: () => Promise.reject(new Error("no ECharts in jsdom")),
    Chart: ({ build, data, label }: { build: (d: unknown, t: unknown) => { series?: unknown[] }; data: unknown; label: string }) => {
      const option = build(data, lightTheme("ru"));
      return createElement("figure", { "aria-label": label, "data-series": Array.isArray(option.series) ? option.series.length : 0 });
    },
  };
});

// React Router создаёт Request с AbortSignal из jsdom, а Request из Node (undici) ждёт свой: при смене
// параметров URL (фильтры «Запросов») навигация падала бы. В тестах сигнал отмены не нужен.
{
  const NativeRequest = globalThis.Request;
  globalThis.Request = class extends NativeRequest {
    constructor(input: RequestInfo | URL, init?: RequestInit) {
      if (init?.signal) {
        const { signal: _signal, ...rest } = init;
        void _signal;
        super(input, rest);
      } else {
        super(input, init);
      }
    }
  } as typeof Request;
}
