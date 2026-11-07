import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { LIGHT } from "./theme";

// Экспорт графиков всегда в светлой теме (решение 3 ревью): её значения — копия :root из tokens.css.
describe("светлая тема графиков", () => {
  it("совпадает с токенами tokens.css", () => {
    const css = readFileSync(resolve(__dirname, "../ui/tokens.css"), "utf8");
    const root = css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {")));
    const token = (name: string) => new RegExp(`${name}:\\s*([^;]+);`).exec(root)?.[1]?.trim();
    expect(token("--text")).toBe(LIGHT.text);
    expect(token("--muted")).toBe(LIGHT.muted);
    expect(token("--faint")).toBe(LIGHT.faint);
    expect(token("--border-soft")).toBe(LIGHT.grid);
    expect(token("--border")).toBe(LIGHT.axis);
    expect(token("--surface")).toBe(LIGHT.surface);
    expect(token("--warning")).toBe(LIGHT.warning);
    expect(token("--danger")).toBe(LIGHT.danger);
    expect(LIGHT.series.map((_, i) => token(`--series-${i + 1}`))).toEqual(LIGHT.series);
  });
});
