import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import compare from "../../test/fixtures/compare.json";
import { json, mockApi, renderRoutes } from "../../test/render";
import RunsPage from "../runs/RunsPage";
import ComparePage from "./ComparePage";

// Ответ сравнения — настоящий: два прогона nfcorpus_hf (разная concurrency) и s0_baseline_rag на demo.
const [A, B, C] = compare.runs.map((run) => run.id) as [string, string, string];
const routes = [
  { path: "/compare", Component: ComparePage },
  { path: "/runs", Component: RunsPage },
];

const runItem = (id: string, scenario: string) => ({
  id, scenario, scenario_id: null, origin: "import", run_dir: `results/runs/${id}`, state: "completed",
  enqueued_at: "2026-10-24T18:09:17Z", started_at: null, finished_at: null, duration_s: null,
  dataset: "hf:BeIR/nfcorpus", collection: "nfcorpus_v1", tags: [], repeats: 1,
  ndcg: 0.315, recall: 0.15, p50_ms: 152, p95_ms: 232, qps: 95.4, timeout_rate: 0,
});

function serve() {
  return mockApi((_req, url) => {
    if (url.pathname === "/v1/runs/compare") {
      const ids = (url.searchParams.get("ids") ?? "").split(",");
      return json({ ...compare, runs: ids.map((id) => compare.runs.find((run) => run.id === id)) });
    }
    if (url.pathname === "/v1/runs") return json({ k: 10, total: 3, items: [runItem(A, "nfcorpus_hf"), runItem(B, "nfcorpus_hf"), runItem(C, "s0_baseline_rag")] });
    if (url.pathname === "/v1/scenarios") return json([]);
    return undefined;
  });
}

describe("сравнение прогонов", () => {
  it("плашка сопоставимости, метрики с Δ, графики и различия по классам", async () => {
    serve();
    renderRoutes(routes, `/compare?ids=${A},${B},${C}`);
    expect(await screen.findByText("Датасеты разные — качество несравнимо.")).toBeInTheDocument();
    expect(screen.getByText("Окружение разное — задержки несравнимы.")).toBeInTheDocument(); // у s0 нет секции env

    const metrics = screen.getByRole("table", { name: "Метрики бок о бок" });
    const p50 = within(metrics).getByText("total p50").closest("tr")!;
    expect(p50).toHaveTextContent("+77"); // 75 → 152 мс у второго прогона nfcorpus
    expect(within(metrics).queryByText("emb hit rate")).not.toBeInTheDocument(); // одни нули — строки нет

    expect(screen.getByLabelText("Качество по k по прогонам")).toHaveAttribute("data-series", "6"); // 3 прогона + усы
    expect(screen.getByLabelText("Задержки по стадиям по прогонам")).toHaveAttribute("data-series", "3"); // embed, retrieve, total

    const diffs = screen.getByRole("table", { name: "Различия" });
    expect(within(diffs).getByText(/Настройка RAG \(config\)/)).toBeInTheDocument();
    expect(within(diffs).getByText("dataset.eval_sha256")).toBeInTheDocument();
  });

  it("«сделать базовым» переставляет прогон первым в ids", async () => {
    const { calls } = serve();
    renderRoutes(routes, `/compare?ids=${A},${B}`);
    await screen.findByRole("table", { name: "Метрики бок о бок" });
    await userEvent.click(screen.getAllByRole("button", { name: "сделать базовым" })[0]!);
    expect(calls.at(-1)!.path).toContain(`ids=${encodeURIComponent(`${B},${A}`)}`);
  });

  it("в списке прогонов «Сравнить» оживает с двух отмеченных и ведёт на /compare", async () => {
    serve();
    renderRoutes(routes, "/runs");
    const compareButton = await screen.findByRole("button", { name: /^Сравнить/ });
    expect(compareButton).toBeDisabled();
    await userEvent.click(await screen.findByRole("checkbox", { name: `Выбрать ${B}` }));
    await userEvent.click(screen.getByRole("checkbox", { name: `Выбрать ${A}` }));
    expect(compareButton).toBeEnabled();
    await userEvent.click(compareButton);
    expect(await screen.findByRole("heading", { name: /Сравнение прогонов/ })).toBeInTheDocument();
  });
});
