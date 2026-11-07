import nfcorpus from "../test/fixtures/summary-nfcorpus.json";
import s0old from "../test/fixtures/summary-s0-old.json";
import { fileName, toCsv } from "./csv";
import type { Summary } from "./summary";
import { missesTable, percentilesTable, qualityTable, repeatsTable } from "./tables";

describe("CSV", () => {
  it("RFC 4180: кавычки, запятые и переводы строк; числа с точкой; null — пусто", () => {
    expect(toCsv(["qid", "query", "ms"], [["q1", 'say "hi", ok', 152.46], ["q2", "two\nlines", null], ["q3", "plain", Number.NaN]])).toBe(
      'qid,query,ms\r\nq1,"say ""hi"", ok",152.46\r\nq2,"two\nlines",\r\nq3,plain,\r\n',
    );
  });

  it("имена файлов без опасных символов", () => {
    expect(fileName("nfcorpus_hf_20261024T180917Z", "quality-k", "doc")).toBe("nfcorpus_hf_20261024T180917Z_quality-k_doc");
    expect(fileName("a/b c", "x")).toBe("a-b-c_x");
  });
});

describe("таблицы для CSV — длинный формат, ключи как в summary", () => {
  const one = nfcorpus as unknown as Summary;
  const five = s0old as unknown as Summary;

  it("качество: строка на метрику и k, значения повторов колонками", () => {
    const table = qualityTable(five, "doc");
    expect(table.header).toEqual(["level", "metric", "k", "mean", "std", "min", "max", "repeat_0", "repeat_1", "repeat_2", "repeat_3", "repeat_4"]);
    expect(table.rows[0]!.slice(0, 3)).toEqual(["doc", "recall", 1]);
  });

  it("перцентили, промахи, повторы", () => {
    const p = percentilesTable(one);
    expect(p.rows.map((r) => `${r[0]} p${r[1]}`)).toContain("total p95");
    expect(missesTable(one).rows.find((r) => r[0] === 10)).toEqual([10, 0, 323, 101]);
    const repeats = repeatsTable(one);
    expect(repeats.header).toContain("doc_ndcg@10_mean");
    expect(repeats.rows[0]![repeats.header.indexOf("timeout_stage_counts")]).toBe("{}"); // словарь — JSON
  });
});
