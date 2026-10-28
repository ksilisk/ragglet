import { fmtBytes, fmtDataset, fmtDuration, fmtMetric, fmtMs, fmtNumber, fmtPercent, fmtWhen, plural, splitDataset } from "./format";

const nbsp = (s: string) => s.replace(/ /g, " ");

describe("числа по-русски", () => {
  it("группы разрядов и запятая", () => {
    expect(fmtNumber(9696)).toBe(nbsp("9 696"));
    expect(fmtMetric(0.315246)).toBe("0,315");
    expect(fmtNumber(null)).toBe("—");
    expect(fmtNumber(Number.NaN)).toBe("—");
  });
  it("задержки: меньше 10 мс — с десятыми", () => {
    expect(fmtMs(232.06)).toBe("232");
    expect(fmtMs(9.14)).toBe("9,1");
  });
  it("проценты", () => {
    expect(fmtPercent(0)).toBe(nbsp("0 %"));
    expect(fmtPercent(0.0217)).toBe(nbsp("2,2 %"));
    expect(fmtPercent(0.5)).toBe(nbsp("50 %"));
  });
  it("байты", () => {
    expect(fmtBytes(512)).toBe(nbsp("512 Б"));
    expect(fmtBytes(17179869184)).toBe(nbsp("16,0 ГБ"));
  });
});

describe("время", () => {
  const now = new Date(2026, 9, 27, 18, 0, 0);
  it("сегодня, этот год, прошлый год", () => {
    expect(fmtWhen(new Date(2026, 9, 27, 17, 15), now)).toBe("сегодня 17:15");
    expect(fmtWhen(new Date(2026, 9, 24, 18, 9), now)).toBe("24.10 18:09");
    expect(fmtWhen(new Date(2025, 0, 2, 3, 4), now)).toBe("02.01.2025 03:04");
    expect(fmtWhen(null, now)).toBe("—");
  });
  it("длительности", () => {
    expect(fmtDuration(41)).toBe("0:41");
    expect(fmtDuration(123)).toBe("2:03");
    expect(fmtDuration(3723)).toBe("1:02:03");
    expect(fmtDuration(null)).toBe("—");
  });
});

describe("датасет", () => {
  it("label → источник, id, ревизия", () => {
    expect(splitDataset("hf:BeIR/nfcorpus@b5026a0e96e8+a451b3b26d3a")).toEqual({ source: "hf", id: "BeIR/nfcorpus", revision: "b5026a0e96e8+a451b3b26d3a" });
    expect(fmtDataset("local:datasets/demo")).toBe("local · datasets/demo");
  });
});

it("склонение", () => {
  expect([1, 2, 5, 11, 21, 22, 25].map((n) => plural(n, "запрос|запроса|запросов"))).toEqual([
    "запрос", "запроса", "запросов", "запросов", "запрос", "запроса", "запросов",
  ]);
});
