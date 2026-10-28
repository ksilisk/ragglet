import { compactNames } from "./names";

it("compactNames", () => {
  expect(compactNames(["chunks_v1_shard0", "chunks_v1_shard1"])).toBe("chunks_v1_shard0, _shard1");
  expect(compactNames(["nfcorpus_v1"])).toBe("nfcorpus_v1");
  expect(compactNames(["a", "b"])).toBe("a, b");
  expect(compactNames(["x", "x"])).toBe("x");
  expect(compactNames([])).toBe("");
});
