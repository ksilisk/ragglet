import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { en } from "./en";
import { pluralize, translate } from "./i18n";

const SRC = resolve(__dirname, "..");
const CYRILLIC = /[А-Яа-яЁё]/;
const SKIP = [/\.test\.tsx?$/, /^test\//, /^i18n\/en\.ts$/, /^api\/schema\.d\.ts$/];

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return files(path);
    return /\.tsx?$/.test(name) ? [path] : [];
  });
}

interface Literal {
  text: string;
  start: number;
  template: boolean;
  dynamic: boolean;
}

/** Строки и код без комментариев: достаточно, чтобы найти русский текст вне t()/msg(). */
function scan(source: string): { code: string; literals: Literal[] } {
  let code = "";
  const literals: Literal[] = [];
  let i = 0;
  while (i < source.length) {
    const c = source[i]!;
    const next = source[i + 1];
    if (c === "/" && next === "/") {
      while (i < source.length && source[i] !== "\n") i += 1;
      continue;
    }
    if (c === "/" && next === "*") {
      const end = source.indexOf("*/", i + 2);
      i = end < 0 ? source.length : end + 2;
      continue;
    }
    if (c === '"' || c === "'" || c === "`") {
      const start = code.length;
      let j = i + 1;
      let text = "";
      let dynamic = false;
      while (j < source.length && source[j] !== c) {
        if (source[j] === "\\") {
          text += source[j + 1];
          j += 2;
          continue;
        }
        if (c === "`" && source[j] === "$" && source[j + 1] === "{") dynamic = true;
        if (c !== "`" && source[j] === "\n") break; // не строка (например, апостроф в JSX-тексте)
        text += source[j];
        j += 1;
      }
      if (j < source.length && source[j] === c) {
        literals.push({ text, start, template: c === "`", dynamic });
        code += c + c; // строка — пустая в коде, её текст в literals
        i = j + 1;
        continue;
      }
    }
    code += c;
    i += 1;
  }
  return { code, literals };
}

const WRAPPED = /(?:\bt|\btranslate|\bmsg)\(\s*$/;
const PLURAL = /(?:\bt\.plural|\bplural|\bpluralize)\([^()]*,\s*$/;

interface Problem {
  where: string;
  text: string;
  why: string;
}

function check(): Problem[] {
  const problems: Problem[] = [];
  for (const path of files(SRC)) {
    const rel = relative(SRC, path);
    if (SKIP.some((re) => re.test(rel))) continue;
    const { code, literals } = scan(readFileSync(path, "utf8"));
    const lineOf = (offset: number) => code.slice(0, offset).split("\n").length;
    for (const lit of literals) {
      if (!CYRILLIC.test(lit.text)) continue;
      const before = code.slice(Math.max(0, lit.start - 120), lit.start);
      const where = `${rel}:${lineOf(lit.start)}`;
      if (lit.dynamic) {
        problems.push({ where, text: lit.text, why: "шаблон с ${…}: перевод через t(\"… {name} …\", { name })" });
      } else if (!WRAPPED.test(before) && !PLURAL.test(before)) {
        problems.push({ where, text: lit.text, why: "не обёрнута в t() / msg() / plural()" });
      } else if (!(lit.text in en)) {
        problems.push({ where, text: lit.text, why: "нет перевода в i18n/en.ts" });
      }
    }
    code.split("\n").forEach((line, n) => {
      if (CYRILLIC.test(line)) problems.push({ where: `${rel}:${n + 1}`, text: line.trim(), why: "русский текст в JSX или коде — нужен {t(\"…\")}" });
    });
  }
  return problems;
}

describe("английская версия UI", () => {
  it("каждая русская строка интерфейса переведена", () => {
    const problems = check();
    expect(problems.map((p) => `${p.where} — ${p.why}: ${p.text}`)).toEqual([]);
  });

  it("подстановки и формы числа", () => {
    expect(translate("сегодня {time}", { time: "17:15" }, "ru")).toBe("сегодня 17:15");
    expect(pluralize(1, "повтор|повтора|повторов", "ru")).toBe("повтор");
    expect(pluralize(3, "повтор|повтора|повторов", "ru")).toBe("повтора");
    expect(pluralize(5, "повтор|повтора|повторов", "ru")).toBe("повторов");
    expect(pluralize(1, "повтор|повтора|повторов", "en")).toBe(en["повтор|повтора|повторов"]!.split("|")[0]);
    expect(pluralize(5, "повтор|повтора|повторов", "en")).toBe(en["повтор|повтора|повторов"]!.split("|")[1]);
  });

  it("в переводах те же подстановки, что в ключах", () => {
    const vars = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
    const bad = Object.entries(en).filter(([ru, tr]) => JSON.stringify(vars(ru)) !== JSON.stringify(vars(tr)));
    expect(bad).toEqual([]);
  });
});
