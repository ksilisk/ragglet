import { cdf, column, cumulativeHitRate, stageCount, timelinePoints } from "./perQuery";
import { DEFAULT_FILTERS, queriesSearch, readFilters, writeFilters } from "./queryParams";

const rows = [
  { qid: "a", t_start_ms: 0, lat_total_ms: 100, lat_rerank_ms: 60, outcome: "ok", emb_cache: "miss" },
  { qid: "b", t_start_ms: 50, lat_total_ms: 6000, lat_rerank_ms: null, outcome: "timeout", timeout_stage: "rerank", emb_cache: "miss" },
  { qid: "c", t_start_ms: 20, lat_total_ms: 100, lat_rerank_ms: 70, outcome: "ok", emb_cache: "hit" },
  { qid: "d", t_start_ms: 90, lat_total_ms: 300, lat_rerank_ms: 90, outcome: "ok", emb_cache: null },
];

describe("per-query для графиков", () => {
  it("CDF: по возрастанию, одинаковые значения — одна точка, кончается на 1", () => {
    expect(cdf(column(rows, "lat_total_ms"))).toEqual([
      [100, 0.5],
      [300, 0.75],
      [6000, 1],
    ]);
    expect(cdf([])).toEqual([]);
  });

  it("стадия считается только там, где завершилась", () => {
    expect(stageCount(rows, "rerank")).toBe(3);
    expect(stageCount(rows, "total")).toBe(4);
  });

  it("линия времени: секунды от начала повтора, группа — исход или отметка кэша", () => {
    expect(timelinePoints(rows, "outcome").map((p) => [p.qid, p.t, p.group])).toEqual([
      ["a", 0, "ok"],
      ["b", 0.05, "timeout"],
      ["c", 0.02, "ok"],
      ["d", 0.09, "ok"],
    ]);
    expect(timelinePoints(rows, "emb").map((p) => p.group)).toEqual(["miss", "miss", "hit", "none"]);
  });

  it("накопительная доля попаданий — по времени старта, только обращения к слою", () => {
    expect(cumulativeHitRate(rows, "emb")).toEqual([
      [0, 0], // a: miss
      [0.02, 0.5], // c: hit
      [0.05, 1 / 3], // b: miss
    ]);
  });
});

describe("фильтры «Запросов» в URL", () => {
  it("туда и обратно; умолчания не пишутся", () => {
    const params = writeFilters(new URLSearchParams({ tab: "queries" }), { repeat: 1, outcome: "timeout", miss_at: 10, min_total_ms: 1000, q: "fava", sort: "-lat_total_ms" });
    expect(params.toString()).toBe("tab=queries&repeat=1&outcome=timeout&miss_at=10&min_total_ms=1000&q=fava&sort=-lat_total_ms");
    expect(readFilters(params)).toEqual({ ...DEFAULT_FILTERS, repeat: 1, outcome: "timeout", miss_at: 10, min_total_ms: 1000, q: "fava", sort: "-lat_total_ms" });
    expect(writeFilters(params, { repeat: 0, outcome: null }).toString()).toBe("tab=queries&miss_at=10&min_total_ms=1000&q=fava&sort=-lat_total_ms");
  });

  it("смена фильтра сбрасывает страницу и открытый запрос, смена страницы — нет", () => {
    const open = new URLSearchParams("tab=queries&offset=50&qid=PLAIN-1");
    expect(writeFilters(open, { outcome: "error" }).toString()).toBe("tab=queries&outcome=error");
    expect(writeFilters(open, { offset: 100 }).toString()).toBe("tab=queries&offset=100&qid=PLAIN-1");
    expect(writeFilters(open, { qid: "PLAIN-2" }).toString()).toBe("tab=queries&offset=50&qid=PLAIN-2");
  });

  it("мусор в URL — умолчания", () => {
    expect(readFilters(new URLSearchParams("repeat=x&outcome=nope&miss_at=0&min_total_ms=abc"))).toEqual(DEFAULT_FILTERS);
    expect(queriesSearch({ repeat: 0, miss_at: 10 })).toBe("?tab=queries&miss_at=10");
  });
});
