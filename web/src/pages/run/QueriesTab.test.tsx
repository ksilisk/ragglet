import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import completed from "../../test/fixtures/run-completed.json";
import detail from "../../test/fixtures/query-detail.json";
import page from "../../test/fixtures/queries-page.json";
import { json, mockApi, renderRoutes } from "../../test/render";
import RunPage from "./RunPage";

// Страница и деталь — настоящие ответы API на nfcorpus_hf (5 самых медленных запросов, PLAIN-1161).
const ID = completed.id;
const { summary, scenario_yaml, manifest, ...light } = completed;
void manifest;
const routes = [{ path: "/runs/:id", Component: RunPage }];

function serve(grades: { available: boolean; code: string | null; message: string | null } = detail.grades) {
  return mockApi((_req, url) => {
    if (url.pathname === `/v1/runs/${ID}`) {
      const include = url.searchParams.get("include");
      if (include === "summary") return json({ ...light, summary });
      if (include === "scenario") return json({ ...light, scenario_yaml });
      return json(light);
    }
    if (url.pathname === `/v1/runs/${ID}/queries`) return json(page);
    if (url.pathname.startsWith(`/v1/runs/${ID}/queries/`)) return json({ ...detail, grades });
    if (url.pathname === `/v1/runs/${ID}/artifacts`) return json([]);
    return undefined;
  });
}

describe("вкладка «Запросы»", () => {
  it("фильтры из URL уходят на сервер, клик по строке открывает панель", async () => {
    const { calls } = serve();
    renderRoutes(routes, `/runs/${ID}?tab=queries&outcome=timeout&sort=-lat_total_ms`);
    const table = await screen.findByRole("table", { name: "Запросы прогона" });
    await within(table).findByText("PLAIN-1161");
    expect(table.querySelectorAll("tbody tr")).toHaveLength(5);
    const request = calls.find((c) => c.path.startsWith(`/v1/runs/${ID}/queries?`))!;
    expect(request.path).toContain("outcome=timeout");
    expect(request.path).toContain("sort=-lat_total_ms");
    expect(request.path).toContain("limit=50");
    expect(screen.getByText("323 запроса")).toBeInTheDocument();

    await userEvent.click(within(table).getByText("PLAIN-1161"));
    const panel = await screen.findByRole("region", { name: "Запрос PLAIN-1161" });
    expect(within(panel).getByText("«fava beans»")).toBeInTheDocument();
    expect(within(panel).getByText("MED-4281")).toBeInTheDocument();
    expect(within(panel).getByText("не найден")).toBeInTheDocument();
    expect(within(panel).getByText(/Score retrieval и источник/)).toBeInTheDocument();
    await userEvent.click(within(panel).getAllByRole("button", { name: "текст" })[0]!);
    expect(within(panel).getByRole("button", { name: "скрыть" })).toBeInTheDocument();
  });

  it("чипы меняют фильтр, а недоступные оценки подписаны честно", async () => {
    const { calls } = serve({ available: false, code: "dataset.not_local", message: "dataset is not on this machine" });
    renderRoutes(routes, `/runs/${ID}?tab=queries&qid=PLAIN-1161`);
    expect(await screen.findByText(/Оценки и заголовки недоступны: dataset is not on this machine/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /^timeout/ }));
    const last = calls.filter((c) => c.path.startsWith(`/v1/runs/${ID}/queries?`)).at(-1)!;
    expect(last.path).toContain("outcome=timeout");
    // смена фильтра закрывает панель: она была про другую выборку
    expect(screen.queryByRole("region", { name: "Запрос PLAIN-1161" })).not.toBeInTheDocument();
  });
});
