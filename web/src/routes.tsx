import { createBrowserRouter } from "react-router";
import { NotFound, RouteError } from "./pages/misc/NotFound";
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
      { path: "indexes/profiles/:id", lazy: async () => ({ Component: (await import("./pages/profile/ProfileEditorPage")).default }) },
      { path: "builds/:id", lazy: async () => ({ Component: (await import("./pages/builds/BuildPage")).default }) },
      { path: "runs", lazy: async () => ({ Component: (await import("./pages/runs/RunsPage")).default }) },
      { path: "runs/:id", lazy: async () => ({ Component: (await import("./pages/run/RunPage")).default }) },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
