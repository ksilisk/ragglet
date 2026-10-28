import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { RouterProvider, createMemoryRouter, type RouteObject } from "react-router";
import { createQueryClient } from "../api/queries";
import { I18nProvider, type Lang } from "../i18n/i18n";
import { ConfirmProvider } from "../ui/Dialog";
import datasets from "./fixtures/datasets.json";
import indexes from "./fixtures/indexes.json";
import metaProfile from "./fixtures/meta-index-profile.json";
import metaScenario from "./fixtures/meta-scenario.json";
import stand from "./fixtures/stand.json";

export type Handler = (req: Request, url: URL) => Response | Promise<Response> | undefined;

export const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

/** Подмена fetch: свои ответы теста, затем общие фикстуры стенда; неизвестный путь — ошибка теста. */
export function mockApi(handler: Handler = () => undefined) {
  const calls: Array<{ method: string; path: string; body: unknown }> = [];
  const fetchMock = vi.fn(async (input: Request) => {
    const url = new URL(input.url);
    const text = input.method === "GET" ? "" : await input.clone().text();
    calls.push({ method: input.method, path: url.pathname + url.search, body: text ? JSON.parse(text) : null });
    const own = await handler(input, url);
    if (own) return own;
    const common: Record<string, unknown> = {
      "/v1/meta/scenario": metaScenario,
      "/v1/meta/index-profile": metaProfile,
      "/v1/indexes": indexes,
      "/v1/datasets": datasets,
      "/v1/stand": stand,
      "/v1/tasks": [],
    };
    if (input.method === "GET" && url.pathname in common) return json(common[url.pathname]);
    throw new Error(`unexpected ${input.method} ${url.pathname}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

export function renderRoutes(routes: RouteObject[], path: string, lang: Lang = "ru"): ReturnType<typeof render> {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  const qc = createQueryClient();
  return render(
    <I18nProvider initialLang={lang}>
      <QueryClientProvider client={qc}>
        <ConfirmProvider>
          <RouterProvider router={router} />
        </ConfirmProvider>
      </QueryClientProvider>
    </I18nProvider>,
  );
}
