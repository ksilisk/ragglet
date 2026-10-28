import { enumAt, schemaAt } from "./schema";

const schema = {
  properties: {
    rerank: { $ref: "#/$defs/Rerank" },
    retrieval: { $ref: "#/$defs/Retrieval" },
    description: { anyOf: [{ type: "string" }, { type: "null" }] },
  },
  $defs: {
    Rerank: { properties: { provider: { enum: ["cross_encoder", "noop"], type: "string" } } },
    Retrieval: { properties: { sources: { type: "array", items: { $ref: "#/$defs/Source" } } } },
    Source: { properties: { kind: { enum: ["vector", "keyword"] } } },
  },
};

it("enum через $ref, anyOf и items", () => {
  expect(enumAt(schema, ["rerank", "provider"])).toEqual(["cross_encoder", "noop"]);
  expect(enumAt(schema, ["retrieval", "sources", 0, "kind"])).toEqual(["vector", "keyword"]);
  expect(schemaAt(schema, ["description"])).toEqual({ type: "string" });
  expect(enumAt(schema, ["nope", "x"])).toEqual([]);
});
