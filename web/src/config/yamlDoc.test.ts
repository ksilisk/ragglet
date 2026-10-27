import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { appendItem, deleteIn, detectIndentSeq, parseYaml, rangeOf, removeItem, setIn } from "./yamlDoc";
import { parsePath } from "./paths";

const configs = resolve(__dirname, "../../../configs");
const templates = ["scenarios", "indexes"].flatMap((dir) =>
  readdirSync(join(configs, dir))
    .filter((f) => f.endsWith(".yaml"))
    .map((f) => ({ name: `${dir}/${f}`, text: readFileSync(join(configs, dir, f), "utf8") })),
);
const scenarios = templates.filter((t) => t.name.startsWith("scenarios/"));

function changedLines(a: string, b: string): number {
  const x = a.split("\n");
  const y = b.split("\n");
  let n = Math.abs(x.length - y.length);
  for (let i = 0; i < Math.min(x.length, y.length); i += 1) if (x[i] !== y[i]) n += 1;
  return n;
}
const comments = (text: string) => text.split("\n").filter((l) => l.includes("#")).map((l) => l.slice(l.indexOf("#")).trim());

describe("шаблоны configs/", () => {
  it("находятся", () => {
    expect(scenarios.length).toBeGreaterThanOrEqual(9);
    expect(templates.length).toBeGreaterThan(scenarios.length);
  });

  it.each(templates)("$name разбирается", ({ text }) => {
    const parsed = parseYaml(text);
    expect(parsed.error).toBeNull();
    expect(parsed.data).toBeTypeOf("object");
  });

  it.each(scenarios)("$name: замена name меняет одну строку", ({ text }) => {
    const out = setIn(text, ["name"], "copy_x");
    expect(parseYaml(out).data!.name).toBe("copy_x");
    expect(changedLines(text, out)).toBe(1);
  });

  it.each(scenarios)("$name: правка скаляра и flow-списка — по строке на правку", ({ text }) => {
    let out = setIn(text, ["retrieval", "top_k"], 77);
    const hadTopK = parseYaml(text).data!.retrieval && (parseYaml(text).data!.retrieval as Record<string, unknown>).top_k !== undefined;
    if (hadTopK) expect(changedLines(text, out)).toBe(1);
    out = setIn(out, ["metrics", "quality", "k"], [1, 5]);
    expect((parseYaml(out).data as { metrics: { quality: { k: number[] } } }).metrics.quality.k).toEqual([1, 5]);
  });

  it.each(templates)("$name: структурная правка сохраняет комментарии и незнакомые ключи", ({ text }) => {
    const withExtra = setIn(text, ["x_custom", "nested"], "keep me");
    const out = setIn(withExtra, ["rerank", "enabled"], true);
    const data = parseYaml(out).data!;
    expect(data.x_custom).toEqual({ nested: "keep me" });
    expect(comments(out)).toEqual(comments(text));
    if ("version" in parseYaml(text).data!) expect(data.version).toBe(parseYaml(text).data!.version);
  });
});

describe("setIn", () => {
  const base = '# шапка\nname: "a"  # имя\nretrieval:\n  top_k: 30 # глубина\n  sources:\n    - name: v\n      kind: vector\nmetrics:\n  quality:\n    k: [1, 3, 5, 10]\n';

  it("скаляр: стиль кавычек и комментарий в строке сохраняются", () => {
    expect(setIn(base, ["name"], "b")).toBe(base.replace('name: "a"', 'name: "b"'));
    expect(setIn(base, ["retrieval", "top_k"], 50)).toBe(base.replace("top_k: 30", "top_k: 50"));
  });

  it("то же значение — текст не меняется", () => {
    expect(setIn(base, ["retrieval", "top_k"], 30)).toBe(base);
  });

  it("flow-список заменяется на месте", () => {
    expect(setIn(base, ["metrics", "quality", "k"], [1, 3])).toBe(base.replace("[1, 3, 5, 10]", "[1, 3]"));
  });

  it("новый ключ в существующей секции и новая секция", () => {
    const out = setIn(setIn(base, ["retrieval", "async"], false), ["rerank", "output_top_n"], 10);
    const data = parseYaml(out).data as Record<string, Record<string, unknown>>;
    expect(data.retrieval!.async).toBe(false);
    expect(data.rerank).toEqual({ output_top_n: 10 });
    expect(out).toContain("# шапка");
    expect(out).toContain("# глубина");
  });

  it("секция без значения («rerank:») становится словарём", () => {
    const out = setIn("name: a\nrerank:\n", ["rerank", "enabled"], true);
    expect(parseYaml(out).data!.rerank).toEqual({ enabled: true });
  });

  it("строка, которая в YAML 1.1 читалась бы как bool, — в кавычках", () => {
    const out = setIn("name: a\nsplit: test\n", ["split"], "yes");
    expect(parseYaml(out).data!.split).toBe("yes");
  });

  it("новый список скаляров пишется flow-стилем", () => {
    expect(setIn("name: a\n", ["tags"], ["beir", "hf"])).toBe("name: a\ntags: [beir, hf]\n");
  });
});

describe("deleteIn, appendItem, removeItem", () => {
  it("удаление ключа и опустевшего родителя", () => {
    const out = deleteIn("name: a\nrerank:\n  revision: main\n", ["rerank", "revision"]);
    expect(parseYaml(out).data).toEqual({ name: "a" });
  });

  it("нет ключа — текст тот же", () => {
    const text = "name: a  # c\n";
    expect(deleteIn(text, ["rerank", "revision"])).toBe(text);
  });

  it("элементы списка", () => {
    let out = appendItem("name: a\n", ["retrieval", "sources"], { name: "v", kind: "vector" });
    out = appendItem(out, ["retrieval", "sources"], { name: "bm25", kind: "keyword" });
    expect((parseYaml(out).data as { retrieval: { sources: unknown[] } }).retrieval.sources).toHaveLength(2);
    out = removeItem(out, ["retrieval", "sources"], 0);
    expect((parseYaml(out).data as { retrieval: { sources: Array<{ name: string }> } }).retrieval.sources.map((s) => s.name)).toEqual(["bm25"]);
  });
});

describe("parseYaml", () => {
  it("YAML 1.1 как у PyYAML", () => {
    expect(parseYaml("run:\n  shuffle: yes\n  save_per_query: off\nseed: 0o17\n").data).toEqual({
      run: { shuffle: true, save_per_query: false },
      seed: "0o17",
    });
  });

  it("ошибка разбора — строка и столбец", () => {
    const parsed = parseYaml("name: a\nretrieval:\n  top_k: [1, 2\n");
    expect(parsed.error).not.toBeNull();
    expect(parsed.error!.line).toBeGreaterThanOrEqual(3);
  });

  it("корень не словарь — ошибка", () => {
    expect(parseYaml("- a\n- b\n").error?.message).toMatch(/mapping/);
    expect(parseYaml("").error).not.toBeNull();
  });
});

describe("rangeOf", () => {
  const text = "name: a\nmetrics:\n  quality:\n    k: [1, 30]\nrerank:\n";
  const doc = parseYaml(text).doc!;
  it("диапазон значения", () => {
    const [a, b] = rangeOf(doc, parsePath("metrics.quality.k"))!;
    expect(text.slice(a, b)).toBe("[1, 30]");
  });
  it("ключ без значения — диапазон ключа", () => {
    const [a, b] = rangeOf(doc, parsePath("rerank.model"))!;
    expect(text.slice(a, b)).toBe("rerank");
  });
  it("неизвестный корень — null", () => {
    expect(rangeOf(doc, parsePath("bogus"))).toBeNull();
  });
});

it("detectIndentSeq", () => {
  expect(detectIndentSeq("a:\n  - x\n")).toBe(true);
  expect(detectIndentSeq("a:\n- x\n")).toBe(false);
  expect(detectIndentSeq("a: 1\n")).toBe(true);
});
