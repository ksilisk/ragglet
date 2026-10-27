import { parseYaml } from "../config/yamlDoc";
import { locate, offsetOf, yamlDiagnostics } from "./diagnostics";

const text = "name: a\nmetrics:\n  quality:\n    k: [1, 30]\n";

it("offsetOf считает строки и столбцы с единицы", () => {
  expect(offsetOf(text, 1)).toBe(0);
  expect(offsetOf(text, 2)).toBe(8);
  expect(offsetOf(text, 4, 5)).toBe(text.indexOf("k:"));
});

it("проблема с path — диапазон узла; config.yaml — позиция из where", () => {
  const parsed = parseYaml(text);
  const [a, b] = locate({ path: "metrics.quality.k", where: null }, text, parsed)!;
  expect(text.slice(a, b)).toBe("[1, 30]");
  const [c] = locate({ path: null, where: "line 2, column 1" }, text, parsed)!;
  expect(text.slice(c, c + 8)).toBe("metrics:");
  expect(locate({ path: null, where: "target default (x)" }, text, parsed)).toBeNull();
});

it("ошибка разбора попадает в диагностику", () => {
  const broken = "name: a\nk: [1, 2\n";
  const diags = yamlDiagnostics(broken, parseYaml(broken), []);
  expect(diags).toHaveLength(1);
  expect(diags[0]!.severity).toBe("error");
});
