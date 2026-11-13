import { screen } from "@testing-library/react";
import { json, mockApi, renderRoutes } from "../../test/render";
import stand from "../../test/fixtures/stand.json";
import StandPage from "./StandPage";

function showStand(available: boolean, profiles: string[]) {
  mockApi((_request, url) => {
    switch (url.pathname) {
      case "/v1/stand":
        return json({ ...stand, qdrant: [{ ...stand.qdrant[0], available, collections: available ? 0 : null }] });
      case "/v1/runs":
      case "/v1/builds":
        return json({ items: [], total: 0 });
      case "/v1/index-profiles":
        return json(profiles.map((id) => ({ id, valid: true })));
      case "/v1/scenarios":
        return json([{ id: "baseline_qdrant", valid: true }]);
    }
  });
  renderRoutes([{ path: "/", element: <StandPage /> }], "/");
}

describe("первый индекс на стенде", () => {
  it("на чистом стенде открывает сборку demo с подтверждением", async () => {
    showStand(true, ["demo"]);
    expect(await screen.findByRole("link", { name: "Собрать demo" })).toHaveAttribute("href", "/indexes/profiles/demo?build=1");
    expect(screen.queryByRole("link", { name: "Запустить сценарий" })).not.toBeInTheDocument();
  });

  it("при удалённом demo оставляет ссылку на библиотеку профилей", async () => {
    showStand(true, []);
    await screen.findByText("Стенд свободен");
    expect(screen.getByRole("link", { name: "Собрать индекс" })).toHaveAttribute("href", "/indexes?tab=profiles");
    expect(screen.queryByRole("link", { name: "Собрать demo" })).not.toBeInTheDocument();
  });

  it("недоступный Qdrant не считается пустым", async () => {
    showStand(false, ["demo"]);
    await screen.findByText("Стенд свободен");
    expect(screen.queryByRole("link", { name: "Собрать demo" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "Запустить сценарий" }).length).toBeGreaterThan(0);
  });
});
