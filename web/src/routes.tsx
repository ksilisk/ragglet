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
      { path: "scenarios", element: <Soon title="Сценарии" /> },
      { path: "indexes", element: <Soon title="Индексы" /> },
      { path: "runs", element: <Soon title="Прогоны" /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
