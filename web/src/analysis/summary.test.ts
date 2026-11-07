import completed from "../test/fixtures/run-completed.json";
import nfcorpus from "../test/fixtures/summary-nfcorpus.json";
import s0old from "../test/fixtures/summary-s0-old.json";
import legacy from "../test/fixtures/summary-s2-legacy.json";
import {
  availableK,
  latencyGrid,
  legacyQualityKeys,
  levels,
  misses,
  outcomeTotals,
  qualityGrid,
  repeatCount,
  stat,
  timeoutStages,
  totalQueries,
  type Summary,
} from "./summary";

// фикстуры — настоящие summary: nfcorpus_hf (1 повтор), s0_baseline_rag 18.10 (5 повторов, без
// счётчиков исходов), s2_rerank (июнь: метрики до ragglet-16), прогон через API (5 повторов)
const one = nfcorpus as unknown as Summary;
const five = s0old as unknown as Summary;
const old = legacy as unknown as Summary;
const api = completed.summary as unknown as Summary;

describe("summary: агрегат по повторам", () => {
  it("среднее и std берутся из агрегата, значения повторов — из строк", () => {
    const p50 = stat(five, "lat_total_p50");
    expect(p50.mean).toBeCloseTo(14.8149, 4);
    expect(p50.std).toBeCloseTo(0.8758, 4);
    expect(p50.perRepeat).toHaveLength(5);
    expect(p50.min).toBeCloseTo(13.915, 3);
    expect(p50.max).toBeCloseTo(16.179, 3);
  });

  it("один повтор: std нет, хотя в агрегате 0", () => {
    const ndcg = stat(one, "doc_ndcg@10_mean");
    expect(ndcg.mean).toBeCloseTo(0.3152, 4);
    expect(ndcg.std).toBeNull();
  });

  it("без агрегата значения считаются по строкам повторов", () => {
    const bare: Summary = { repeats: five.repeats, aggregate: {} };
    const p50 = stat(bare, "lat_total_p50");
    expect(p50.mean).toBeCloseTo(14.8149, 4);
    expect(p50.std).toBeCloseTo(0.8758, 4); // ddof=1, как aggregate_over_repeats
    expect(repeatCount(bare)).toBe(5);
  });

  it("нет ключа — null, а не 0", () => {
    expect(stat(one, "throughput_nope").mean).toBeNull();
    expect(stat(five, "throughput_qps").mean).toBeNull(); // до ragglet-19
  });
});

describe("summary: качество", () => {
  it("сетка уровень × метрика × k", () => {
    const grid = qualityGrid(one, "doc");
    expect(grid.metrics).toEqual(["ndcg", "recall", "r_cap", "hit", "mrr"]);
    expect(grid.ks).toEqual([1, 3, 5, 10]);
    expect(grid.get("ndcg", 10)?.mean).toBeCloseTo(0.3152, 4);
    expect(grid.get("precision", 10)).toBeNull();
    expect(levels(one)).toEqual(["doc"]);
    expect(availableK(api)).toEqual([1, 3, 5, 10]);
  });

  it("метрики до ragglet-16 не переименовываются и показываются как есть", () => {
    expect(qualityGrid(old, "doc").metrics).toEqual([]);
    expect(legacyQualityKeys(old)).toEqual(["mrr_mean", "recall@10_mean", "recall@1_mean", "recall@30_mean", "recall@3_mean", "recall@5_mean"]);
    expect(legacyQualityKeys(one)).toEqual([]);
  });

  it("промахи при k: n − round(n · hit@k), как miss_at на сервере", () => {
    expect(misses(one, 10)).toEqual([101]); // ragglet-30: miss_at=10 даёт 101 строку
    expect(misses(api, 10)).toHaveLength(5);
    expect(misses(old, 10)).toEqual([null, null, null, null, null]);
  });
});

describe("summary: задержки и исходы", () => {
  it("стадии в порядке конвейера, перцентили по возрастанию", () => {
    const grid = latencyGrid(one);
    expect(grid.stages).toEqual(["embed", "retrieve"]);
    expect(grid.percentiles).toEqual([50, 95, 99]);
    expect(grid.hasTotal).toBe(true);
    expect(grid.get("total", 50)?.mean).toBeCloseTo(152.46, 2);
    expect(grid.get("rerank", 50)).toBeNull();
    expect(latencyGrid(old).stages).toEqual(["embed", "retrieve", "rerank"]);
  });

  it("исходы суммируются по повторам; в старом формате их нет", () => {
    expect(outcomeTotals(api)).toEqual({ ok: 500, timeout: 0, error: 0 });
    expect(outcomeTotals(five)).toBeNull();
    expect(totalQueries(api)).toBe(500);
  });

  it("таймауты по стадиям — сумма словарей повторов", () => {
    const rows = [
      { n: 10, timeout_stage_counts: { rerank: 2 } },
      { n: 10, timeout_stage_counts: { rerank: 1, total: 3 } },
      { n: 10, timeout_stage_counts: { total: 1 } },
    ];
    expect(timeoutStages({ repeats: rows, aggregate: {} })).toEqual([
      { stage: "total", count: 4, perRepeat: [0, 3, 1] },
      { stage: "rerank", count: 3, perRepeat: [2, 1, 0] },
    ]);
    expect(timeoutStages(one)).toEqual([]);
  });
});
