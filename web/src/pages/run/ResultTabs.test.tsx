import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import nfManifest from "../../test/fixtures/manifest-nfcorpus.json";
import completed from "../../test/fixtures/run-completed.json";
import { json, mockApi, renderRoutes } from "../../test/render";
import RunPage from "./RunPage";

// Вкладки результата на настоящем прогоне через API (s0_baseline_rag, 5 повторов по 100 запросов).
const ID = completed.id;
const { summary, scenario_yaml, manifest, ...light } = completed;
void manifest;
const routes = [{ path: "/runs/:id", Component: RunPage }];

const problem = (status: number, code: string, message: string) =>
  json({ title: "Not found", hint: null, issues: [{ code, severity: "error", path: null, where: null, message }] }, status);

function rowsOf(repeat: number) {
  return Array.from({ length: 100 }, (_, i) => ({
    qid: `q${String(i).padStart(3, "0")}`,
    t_start_ms: i * 10,
    lat_embed_ms: 5 + (i % 7),
    lat_retrieve_ms: 4 + repeat,
    lat_total_ms: 12 + (i % 13),
    outcome: i === 7 ? "timeout" : "ok",
  }));
}

function serve(perQuery: "rows" | "none") {
  return mockApi((_req, url) => {
    if (url.pathname === `/v1/runs/${ID}`) {
      const include = url.searchParams.get("include");
      if (include === "summary") return json({ ...light, summary });
      if (include === "scenario") return json({ ...light, scenario_yaml });
      return json(light);
    }
    if (url.pathname === `/v1/runs/${ID}/queries`) {
      if (perQuery === "none") return problem(404, "run.no_per_query", "no per-query files");
      const repeat = Number(url.searchParams.get("repeat") ?? 0);
      const items = rowsOf(repeat);
      return json({ repeat, repeats: [0, 1, 2, 3, 4], source: `per_query.repeat${repeat}.parquet`, columns: Object.keys(items[0]!), total: items.length, items });
    }
    if (url.pathname === `/v1/runs/${ID}/artifacts`) return json([]);
    return undefined;
  });
}

describe("вкладки результата", () => {
  it("«Качество»: метрика × k, раскрытие по повторам, промахи — ссылки в «Запросы»", async () => {
    serve("rows");
    renderRoutes(routes, `/runs/${ID}?tab=quality`);
    const table = await screen.findByRole("table", { name: "Метрика × k" });
    expect(within(table).getAllByRole("row")).toHaveLength(1 + 4 + 1); // шапка, 4 метрики, «список короче k»
    await userEvent.click(within(table).getByRole("button", { name: "nDCG" }));
    expect(within(table).getAllByRole("row")).toHaveLength(1 + 4 + 5 + 1);
    const misses = screen.getByRole("table", { name: "Промахи при k" });
    expect(within(misses).getAllByRole("link")[0]).toHaveAttribute("href", expect.stringContaining("tab=queries&miss_at=1"));
    expect(screen.getByLabelText("Качество по k, уровень doc")).toHaveAttribute("data-series", "8");
  });

  it("«Задержки и нагрузка»: перцентили с n, CDF по пяти повторам, линия времени", async () => {
    const { calls } = serve("rows");
    renderRoutes(routes, `/runs/${ID}?tab=latency`);
    expect(await screen.findByLabelText("Распределение lat_total_ms")).toHaveAttribute("data-series", "5");
    expect(screen.getByLabelText("Линия времени повтора")).toHaveAttribute("data-series", "2"); // ok и timeout
    const table = screen.getByRole("table", { name: "Перцентили по стадиям" });
    expect(within(table).getAllByRole("row")[1]).toHaveTextContent("500"); // embed завершён у всех 500 запросов
    // per-query — только нужные колонки и одним запросом на повтор
    const queries = calls.filter((c) => c.path.startsWith(`/v1/runs/${ID}/queries`));
    expect(queries).toHaveLength(5);
    expect(queries[0]!.path).toContain("fields=qid%2Ct_start_ms");
    expect(screen.getByRole("table", { name: "Пропускная способность и исходы по повторам" }).querySelectorAll("tbody tr")).toHaveLength(5);
  });

  it("без per-query — подпись вместо графиков, таблицы из summary остаются", async () => {
    serve("none");
    renderRoutes(routes, `/runs/${ID}?tab=latency`);
    expect(await screen.findAllByText(/per-query не сохранялись/)).toHaveLength(2);
    expect(screen.getByRole("table", { name: "Перцентили по стадиям" })).toBeInTheDocument();
  });
});

describe("вкладка «Условия»", () => {
  it("manifest по карточкам: сборка индекса на другом коммите, HNSW не построен, JSON", async () => {
    mockApi((_req, url) => {
      if (url.pathname === `/v1/runs/${ID}`) {
        const include = url.searchParams.get("include");
        if (include === "manifest") return json({ ...light, manifest: nfManifest });
        if (include === "scenario") return json({ ...light, scenario_yaml });
        return json(light);
      }
      if (url.pathname === "/v1/runs") return json({ k: 10, total: 0, items: [] });
      if (url.pathname === `/v1/runs/${ID}/artifacts`) return json([]);
      return undefined;
    });
    renderRoutes(routes, `/runs/${ID}?tab=conditions`);
    expect(await screen.findByText("Окружение сборки индекса")).toBeInTheDocument();
    expect(screen.getAllByText("≠ прогону")).toHaveLength(1); // commit сборки индекса
    expect(screen.getByText("indexed_vectors = 0")).toBeInTheDocument();
    expect(screen.getByText(/HNSW не построен/)).toBeInTheDocument();
    expect(screen.getByText("b5026a0e+a451b3b2")).toBeInTheDocument(); // составная ревизия BeIR коротко
    await userEvent.click(screen.getByRole("button", { name: "JSON" }));
    expect(screen.getByText(/"corpus_sha256": "54ef7cf7/)).toBeInTheDocument();
  });
});
