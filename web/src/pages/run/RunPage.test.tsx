import { act, screen, waitFor } from "@testing-library/react";
import completed from "../../test/fixtures/run-completed.json";
import { json, mockApi, renderRoutes } from "../../test/render";
import RunPage from "./RunPage";

const ID = completed.id;
const { summary, scenario_yaml, manifest, ...light } = completed;
void manifest;

const at = (s: number) => new Date(Date.parse("2026-10-27T18:52:30Z") + s * 1000).toISOString();
const queued = { ...light, state: "queued", phase: null, position: 1, progress: null, started_at: null, finished_at: null };
const preparing = { ...light, state: "running", phase: "preparing", position: null, progress: null, started_at: at(0), finished_at: null };
const progress = {
  ...preparing,
  phase: "progress",
  progress: { stage: "queries", done: 120, total: 500, first_at: at(10), updated_at: at(20) },
};
const failed = {
  ...light,
  state: "failed",
  phase: null,
  error: {
    type: "ValidationError",
    message: "Index is not compatible",
    qid: null,
    problem: { title: "Index is not compatible", hint: null, issues: [{ code: "index.not_green", severity: "error", path: "storage.backend.collection", where: null, message: "collection rebuilt" }] },
  },
};

function serve(statuses: unknown[]) {
  let i = 0;
  return mockApi((_req, url) => {
    if (url.pathname === `/v1/runs/${ID}`) {
      const include = url.searchParams.get("include");
      if (include === "summary") return json({ ...light, summary });
      if (include === "scenario") return json({ ...light, scenario_yaml });
      const status = statuses[Math.min(i, statuses.length - 1)];
      i += 1;
      return json(status);
    }
    if (url.pathname === `/v1/runs/${ID}/log`) return json({ text: "== run started\n", offset: 0, next_offset: 15, complete: false });
    if (url.pathname === `/v1/runs/${ID}/artifacts`) return json([{ name: "summary.json", size: 2048, modified_at: at(30) }]);
    return undefined;
  });
}

const routes = [{ path: "/runs/:id", Component: RunPage }];

describe("страница прогона", () => {
  afterEach(() => vi.useRealTimers());

  it("в очереди: позиция и что впереди", async () => {
    mockApi((_req, url) => {
      if (url.pathname === `/v1/runs/${ID}`) return json(queued);
      if (url.pathname === "/v1/tasks")
        return json([{ kind: "build", id: "demo_x", name: "demo_sharded", state: "running", position: null, progress: { stage: "corpus", done: 12, total: 50, first_at: at(0), updated_at: at(1) }, enqueued_at: at(0), started_at: at(0) }]);
      if (url.pathname === `/v1/runs/${ID}/artifacts`) return json([]);
      if (url.searchParams.get("include") === "scenario") return json({ ...light, scenario_yaml });
      return undefined;
    });
    renderRoutes(routes, `/runs/${ID}`);
    expect(await screen.findByText(/позиция 1/)).toBeInTheDocument();
    expect(await screen.findByText("demo_sharded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Отменить" })).toBeInTheDocument();
  });

  it("выполняется: фаза «Запросы», прогресс и повтор", async () => {
    serve([progress]);
    renderRoutes(routes, `/runs/${ID}`);
    expect(await screen.findByText(/повтор 2 из 5/, { selector: "span" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "24");
    expect(screen.getByRole("button", { name: "Повторить" })).toBeDisabled();
  });

  it("упал на втором preflight: проблемы и «Открыть в редакторе»", async () => {
    serve([failed]);
    renderRoutes(routes, `/runs/${ID}`);
    expect(await screen.findByText("ValidationError: Index is not compatible")).toBeInTheDocument();
    expect(screen.getByText("collection rebuilt", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Открыть в редакторе" })).toBeInTheDocument();
  });

  it("готово: урезанный «Обзор» — плитки и таблица повторов", async () => {
    serve([light]);
    renderRoutes(routes, `/runs/${ID}`);
    expect(await screen.findByText("0,865", { selector: "span" })).toBeInTheDocument(); // плитка nDCG@10
    expect(screen.getByRole("table", { name: "Повторы" }).querySelectorAll("tbody tr")).toHaveLength(5);
    expect(screen.getByText("выключен")).toBeInTheDocument(); // кэш выключен в snapshot
  });

  it("опрос раз в секунду до completed, потом останавливается", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { calls } = serve([queued, preparing, progress, light]);
    renderRoutes(routes, `/runs/${ID}`);
    const polls = () => calls.filter((c) => c.path === `/v1/runs/${ID}`).length;
    await waitFor(() => expect(polls()).toBe(1));
    for (let i = 0; i < 4; i += 1) await act(() => vi.advanceTimersByTimeAsync(1000));
    await waitFor(() => expect(screen.getByText("Готово")).toBeInTheDocument());
    const after = polls();
    await act(() => vi.advanceTimersByTimeAsync(5000));
    expect(polls()).toBe(after);
  });

  it("по-английски: подписи, вкладки и числа в en", async () => {
    serve([light]);
    renderRoutes(routes, `/runs/${ID}`, "en");
    expect(await screen.findByText("0.865", { selector: "span" })).toBeInTheDocument(); // плитка nDCG@10, en-US
    expect(screen.getByText("Done")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Repeats" }).querySelectorAll("tbody tr")).toHaveLength(5);
    expect(screen.getByText("off")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Repeat" })).toBeEnabled();
  });
});
