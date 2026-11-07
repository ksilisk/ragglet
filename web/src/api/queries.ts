import { QueryClient, keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiProblem, api, unwrap, type S } from "./client";
import type { RepeatRows } from "../analysis/perQuery";
import type { operations } from "./schema";

// Интервалы опроса (ragglet-31 §3). Скрытая вкладка браузера не опрашивает:
// refetchIntervalInBackground по умолчанию false.
export const POLL = {
  active: 1000, // страница прогона / сборки в queued или running
  log: 2000,
  tasksBusy: 2000,
  tasksIdle: 10000,
  standPage: 10000,
  standStrip: 30000,
  list: 10000,
  indexes: 30000,
} as const;

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // 4xx не лечится повтором; сеть и 5xx — одна попытка
        retry: (count, error) => !(error instanceof ApiProblem && error.status >= 400 && error.status < 500) && count < 1,
        refetchOnWindowFocus: true,
      },
      mutations: { retry: false },
    },
  });
}

export type RunsQuery = NonNullable<operations["list_runs_v1_runs_get"]["parameters"]["query"]>;
export type BuildsQuery = NonNullable<operations["list_builds_v1_builds_get"]["parameters"]["query"]>;

const isActive = (state: S["TaskState"] | undefined) => state === "queued" || state === "running";

export const keys = {
  stand: ["stand"] as const,
  tasks: ["tasks"] as const,
  metaScenario: ["meta", "scenario"] as const,
  metaProfile: ["meta", "index-profile"] as const,
  scenarios: ["scenarios"] as const,
  scenario: (id: string) => ["scenarios", id] as const,
  profiles: ["index-profiles"] as const,
  profile: (id: string) => ["index-profiles", id] as const,
  indexes: ["indexes"] as const,
  datasets: ["datasets"] as const,
  runs: ["runs"] as const,
  runList: (q: RunsQuery) => ["runs", "list", q] as const,
  run: (id: string) => ["runs", "item", id] as const,
  runInclude: (id: string, include: string) => ["runs", "item", id, include] as const,
  runArtifacts: (id: string) => ["runs", "item", id, "artifacts"] as const,
  builds: ["builds"] as const,
  buildList: (q: BuildsQuery) => ["builds", "list", q] as const,
  build: (id: string) => ["builds", "item", id] as const,
  buildInclude: (id: string, include: string) => ["builds", "item", id, include] as const,
};

// -----------------------
// Стенд и задачи
// -----------------------

export function useStand(interval: number = POLL.standPage) {
  return useQuery({ queryKey: keys.stand, queryFn: () => unwrap(api.GET("/v1/stand")), refetchInterval: interval });
}

export function useTasks() {
  return useQuery({
    queryKey: keys.tasks,
    queryFn: () => unwrap(api.GET("/v1/tasks")),
    refetchInterval: (query) => (query.state.data?.length ? POLL.tasksBusy : POLL.tasksIdle),
  });
}

export function useCancelTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ kind, id }: { kind: S["TaskKind"]; id: string }): Promise<S["RunStatus"] | S["BuildStatus"]> =>
      kind === "run"
        ? unwrap(api.POST("/v1/runs/{run_id}/cancel", { params: { path: { run_id: id } } }))
        : unwrap(api.POST("/v1/builds/{build_id}/cancel", { params: { path: { build_id: id } } })),
    onSettled: (_data, _error, { kind, id }) => {
      void qc.invalidateQueries({ queryKey: keys.tasks });
      void qc.invalidateQueries({ queryKey: kind === "run" ? keys.run(id) : keys.build(id) });
      void qc.invalidateQueries({ queryKey: kind === "run" ? keys.runs : keys.builds });
    },
  });
}

// -----------------------
// Meta, библиотека, коллекции, датасеты
// -----------------------

export function useScenarioMeta() {
  return useQuery({ queryKey: keys.metaScenario, queryFn: () => unwrap(api.GET("/v1/meta/scenario")), staleTime: Infinity });
}

export function useProfileMeta() {
  return useQuery({ queryKey: keys.metaProfile, queryFn: () => unwrap(api.GET("/v1/meta/index-profile")), staleTime: Infinity });
}

export function useScenarios() {
  return useQuery({ queryKey: keys.scenarios, queryFn: () => unwrap(api.GET("/v1/scenarios")), refetchInterval: POLL.list });
}

export function useProfiles() {
  return useQuery({ queryKey: keys.profiles, queryFn: () => unwrap(api.GET("/v1/index-profiles")), refetchInterval: POLL.list });
}

export function useIndexes() {
  return useQuery({ queryKey: keys.indexes, queryFn: () => unwrap(api.GET("/v1/indexes")), refetchInterval: POLL.indexes });
}

export function useDatasets() {
  return useQuery({ queryKey: keys.datasets, queryFn: () => unwrap(api.GET("/v1/datasets")), staleTime: 60_000 });
}

// -----------------------
// Прогоны и сборки
// -----------------------

export function useRuns(query: RunsQuery, interval: number | false = POLL.list) {
  return useQuery({
    queryKey: keys.runList(query),
    queryFn: () => unwrap(api.GET("/v1/runs", { params: { query } })),
    refetchInterval: interval,
    placeholderData: keepPreviousData,
  });
}

export function useBuilds(query: BuildsQuery, interval: number | false = POLL.list) {
  return useQuery({
    queryKey: keys.buildList(query),
    queryFn: () => unwrap(api.GET("/v1/builds", { params: { query } })),
    refetchInterval: interval,
    placeholderData: keepPreviousData,
  });
}

/** Лёгкий статус без include: опрос раз в секунду, пока задача активна. */
export function useRun(id: string | null | undefined) {
  return useQuery({
    queryKey: keys.run(id ?? ""),
    queryFn: () => unwrap(api.GET("/v1/runs/{run_id}", { params: { path: { run_id: id! } } })),
    enabled: Boolean(id),
    refetchInterval: (query) => (isActive(query.state.data?.state) ? POLL.active : false),
  });
}

export function useBuild(id: string | null | undefined) {
  return useQuery({
    queryKey: keys.build(id ?? ""),
    queryFn: () => unwrap(api.GET("/v1/builds/{build_id}", { params: { path: { build_id: id! } } })),
    enabled: Boolean(id),
    refetchInterval: (query) => (isActive(query.state.data?.state) ? POLL.active : false),
  });
}

/** Тяжёлые поля прогона (summary, manifest, scenario) — один раз, без опроса. */
export function useRunInclude(id: string, include: "summary" | "manifest" | "scenario", enabled = true) {
  return useQuery({
    queryKey: keys.runInclude(id, include),
    queryFn: () => unwrap(api.GET("/v1/runs/{run_id}", { params: { path: { run_id: id }, query: { include } } })),
    enabled,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

export function useBuildProfileYaml(id: string, enabled = true) {
  return useQuery({
    queryKey: keys.buildInclude(id, "profile"),
    queryFn: () => unwrap(api.GET("/v1/builds/{build_id}", { params: { path: { build_id: id }, query: { include: "profile" } } })),
    enabled,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

export function useRunArtifacts(id: string, enabled = true) {
  return useQuery({
    queryKey: keys.runArtifacts(id),
    queryFn: () => unwrap(api.GET("/v1/runs/{run_id}/artifacts", { params: { path: { run_id: id } } })),
    enabled,
  });
}

const COLUMNS_PAGE = 5000;

/**
 * Per-query колонками для графиков (ragglet-32 §3): все запросы всех записанных повторов, только поля
 * `fields`. Результаты завершённого прогона не меняются — без срока и без опроса.
 */
export function useRunColumns(id: string, fields: string[], enabled = true) {
  return useQuery({
    queryKey: [...keys.runInclude(id, "columns"), fields.join(",")],
    queryFn: async (): Promise<RepeatRows> => {
      const page = (repeat: number, offset: number) =>
        unwrap(api.GET("/v1/runs/{run_id}/queries", { params: { path: { run_id: id }, query: { repeat, offset, limit: COLUMNS_PAGE, fields: fields.join(",") } } }));
      const whole = async (repeat: number) => {
        const first = await page(repeat, 0);
        const items = [...first.items];
        while (items.length < first.total) items.push(...(await page(repeat, items.length)).items);
        return { first, items };
      };
      const zero = await whole(0);
      const rest = await Promise.all(zero.first.repeats.filter((r) => r !== 0).map(async (r) => [r, (await whole(r)).items] as const));
      return { repeats: zero.first.repeats, columns: zero.first.columns, rows: new Map([[0, zero.items], ...rest]) };
    },
    enabled,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

export type QueriesQuery = NonNullable<operations["list_queries_v1_runs__run_id__queries_get"]["parameters"]["query"]>;

/** Страница «Запросов»: фильтры, сортировка и страницы — на сервере (ragglet-32 §7). */
export function useRunQueries(id: string, query: QueriesQuery, enabled = true) {
  return useQuery({
    queryKey: [...keys.runInclude(id, "queries"), query],
    queryFn: () => unwrap(api.GET("/v1/runs/{run_id}/queries", { params: { path: { run_id: id }, query } })),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
}

/** Деталь запроса для панели: оценки qrels, заголовки и тексты чанков. */
export function useQueryDetail(id: string, qid: string | null, repeat: number) {
  return useQuery({
    queryKey: [...keys.runInclude(id, "query"), repeat, qid],
    queryFn: () => unwrap(api.GET("/v1/runs/{run_id}/queries/{qid}", { params: { path: { run_id: id, qid: qid! }, query: { repeat } } })),
    enabled: Boolean(qid),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

/** Сравнение прогонов (ragglet-32 §9): различия, вердикт и summary всех прогонов одним ответом. */
export function useCompare(ids: string[]) {
  return useQuery({
    queryKey: ["runs", "compare", ids.join(",")],
    queryFn: () => unwrap(api.GET("/v1/runs/compare", { params: { query: { ids: ids.join(",") } } })),
    enabled: ids.length >= 2,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
}
