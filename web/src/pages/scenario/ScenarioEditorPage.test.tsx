import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { json, mockApi, renderRoutes } from "../../test/render";
import ScenarioEditorPage from "./ScenarioEditorPage";

const yaml = `# комментарий шаблона
version: 1
name: "nfcorpus_rerank"
dataset:
  source: "hf"
  id: "BeIR/nfcorpus"
storage:
  mode: "single"
  backend:
    type: "qdrant"
    endpoint: "http://localhost:6333"
    collection: "nfcorpus_v1"
retrieval:
  top_k: 50
rerank:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  input_top_k: 50
  output_top_n: 10
metrics:
  quality:
    k: [1, 10, 30]
`;

const doc = { id: "nfcorpus_rerank", yaml, updated_at: "2026-10-27T10:00:00Z", etag: "e1", valid: true, problem: null };
const problem = {
  title: "Scenario config is invalid",
  hint: null,
  issues: [{ code: "config.value_error", severity: "error", path: "metrics.quality.k", where: null, message: "metrics.quality.k=30 exceeds final list depth 10 (rerank.output_top_n)" }],
};

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

describe("редактор сценария", () => {
  it("422 dry-run: ошибка под полем k, точка у секции, клик в панели — фокус на поле", async () => {
    const { calls } = mockApi((req, url) => {
      if (req.method === "GET" && url.pathname === "/v1/scenarios/nfcorpus_rerank") return json(doc);
      if (req.method === "POST" && url.pathname === "/v1/runs") return json(problem, 422);
      return undefined;
    });
    renderRoutes([{ path: "/scenarios/:id", Component: ScenarioEditorPage }], "/scenarios/nfcorpus_rerank");

    // клиентская подсказка видна ещё до проверки
    await screen.findByText(/k = 30 больше глубины финального списка 10/);

    fireEvent.click(screen.getByRole("button", { name: "Проверить" }));
    const field = await waitFor(() => {
      const el = document.getElementById("f-metrics_quality_k")!;
      expect(within(el).getByText(problem.issues[0]!.message)).toBeInTheDocument();
      return el;
    });
    expect(field).toBeTruthy();

    // dry-run шлёт текст редактора как есть, без сохранения
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.path).toBe("/v1/runs");
    expect(post.body).toEqual({ scenario: yaml, scenario_id: "nfcorpus_rerank", dry_run: true, tags: [] });
    expect(calls.some((c) => c.method === "PUT")).toBe(false);

    const nav = screen.getByRole("navigation", { name: "Секции" });
    expect(within(nav).getByRole("button", { name: /Метрики и вывод/ }).querySelector("[title='есть ошибки']")).not.toBeNull();

    const panel = screen.getByRole("region", { name: "Проверка" });
    fireEvent.click(within(panel).getByRole("button", { name: "metrics.quality.k" }));
    await waitFor(() => expect(document.activeElement?.id).toBe("f-metrics_quality_k-input"));
  });

  it("правка поля скрывает его проблему и помечает панель устаревшей", async () => {
    mockApi((req, url) => {
      if (req.method === "GET" && url.pathname === "/v1/scenarios/nfcorpus_rerank") return json(doc);
      if (req.method === "POST" && url.pathname === "/v1/runs") return json(problem, 422);
      return undefined;
    });
    renderRoutes([{ path: "/scenarios/:id", Component: ScenarioEditorPage }], "/scenarios/nfcorpus_rerank");
    fireEvent.click(await screen.findByRole("button", { name: "Проверить" }));
    const field = document.getElementById("f-metrics_quality_k")!;
    await within(field).findByText(problem.issues[0]!.message);

    fireEvent.click(within(field).getByRole("button", { name: "Убрать 30" }));
    expect(within(field).queryByText(problem.issues[0]!.message)).toBeNull();
    expect(screen.getByText("текст изменён после проверки")).toBeInTheDocument();
    expect(screen.getByText("не сохранено")).toBeInTheDocument();
  });
});
