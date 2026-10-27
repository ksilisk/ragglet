import { createBrowserRouter } from "react-router";
import { NotFound, RouteError, Soon } from "./pages/misc/NotFound";
import { AppShell } from "./shell/AppShell";

/** Страницы грузятся лениво: CodeMirror и редактор не попадают в первый бандл. */
export const router = createBrowserRouter([
  {
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, lazy: async () => ({ Component: (await import("./pages/stand/StandPage")).default }) },
      { path: "scenarios", lazy: async () => ({ Component: (await import("./pages/scenarios/ScenariosPage")).default }) },
      { path: "scenarios/:id", lazy: async () => ({ Component: (await import("./pages/scenario/ScenarioEditorPage")).default }) },
      { path: "indexes", lazy: async () => ({ Component: (await import("./pages/indexes/IndexesPage")).default }) },
      { path: "indexes/profiles/:id", element: <Soon title="Профиль индекса" /> },
      { path: "runs", element: <Soon title="Прогоны" /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
