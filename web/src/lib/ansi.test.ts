import { stripAnsi } from "./ansi";

it("вырезает цвет, жирный и курсор", () => {
  expect(stripAnsi("\x1b[1mBertModel\x1b[0m LOAD REPORT")).toBe("BertModel LOAD REPORT");
  expect(stripAnsi("a\x1b[31;1mred\x1b[39m b\x1b[2K\x1b[1G")).toBe("ared b");
  expect(stripAnsi("\x1b]0;title\x07text")).toBe("text");
  expect(stripAnsi("plain")).toBe("plain");
});
