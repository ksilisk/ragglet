import { progressView } from "./eta";

const t0 = Date.parse("2026-11-25T10:15:40Z");
const iso = (s: number) => new Date(t0 + s * 1000).toISOString();
const plan = { repeats: 3, queries_per_repeat: 100 };
const progress = (done: number, total: number, first: number, updated: number) => ({
  stage: "queries", done, total, first_at: iso(first), updated_at: iso(updated),
});

describe("progressView", () => {
  it("нет прогресса — нет полосы", () => {
    expect(progressView(null, plan, iso(0), t0)).toBeNull();
  });

  it("первые секунды — оценка ещё не считается", () => {
    const v = progressView(progress(5, 300, 10, 12), plan, iso(0), t0 + 12_000)!;
    expect(v.remainingS).toBeNull();
    expect(v.elapsedS).toBe(12);
  });

  it("линейная оценка по окну first_at…updated_at", () => {
    // 100 запросов за 10 с после первого: 11 запр./с; осталось 199 → ≈18 с
    const v = progressView(progress(101, 300, 10, 20), plan, iso(0), t0 + 20_000)!;
    expect(v.rate).toBeCloseTo(10);
    expect(v.remainingS).toBeCloseTo(19.9);
    expect(v.fraction).toBeCloseTo(101 / 300);
  });

  it("оценка убывает между событиями", () => {
    const v = progressView(progress(101, 300, 10, 20), plan, iso(0), t0 + 25_000)!;
    expect(v.remainingS).toBeCloseTo(14.9);
  });

  it("done == total — осталось 0", () => {
    expect(progressView(progress(300, 300, 10, 40), plan, iso(0), t0 + 41_000)!.remainingS).toBe(0);
  });

  it("номер повтора на границах", () => {
    const at = (done: number) => progressView(progress(done, 300, 0, 10), plan, iso(0), t0)!.repeat;
    expect(at(0)).toEqual({ index: 1, of: 3 });
    expect(at(100)).toEqual({ index: 1, of: 3 });
    expect(at(101)).toEqual({ index: 2, of: 3 });
    expect(at(300)).toEqual({ index: 3, of: 3 });
  });

  it("без плана (импорт из CLI, сборка) повтора нет", () => {
    expect(progressView(progress(5, 10, 0, 5), null, null, t0)!.repeat).toBeNull();
  });
});
