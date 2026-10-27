import { createEditorStore } from "./store";

const doc = (yaml: string, problem: null | { title: string; hint: null; issues: never[] } = null) => ({
  id: "s",
  yaml,
  updated_at: "2026-10-27T10:00:00Z",
  etag: "e1",
  valid: problem === null,
  problem,
});

describe("store редактора", () => {
  it("правка формы меняет текст и отмечает поле правленым", () => {
    const store = createEditorStore("scenario", doc('name: "s"\nretrieval:\n  top_k: 30\n'), {});
    store.getState().edit(["retrieval", "top_k"], 50);
    expect(store.getState().text).toBe('name: "s"\nretrieval:\n  top_k: 50\n');
    expect([...store.getState().touched]).toEqual(["retrieval.top_k"]);
    expect(store.getState().text).not.toBe(store.getState().saved.text);
  });

  it("проверка сбрасывает «правленые», новая правка снова отмечает", () => {
    const store = createEditorStore("scenario", doc("name: s\n"), {});
    store.getState().edit(["seed"], 7);
    store.getState().setCheck({ source: "check", at: 0, durationMs: 1, ok: true, title: null, hint: null, issues: [], report: null });
    expect(store.getState().touched.size).toBe(0);
    expect(store.getState().checkedText).toBe(store.getState().text);
  });

  it("битый YAML: форма ничего не пишет, текст правится только напрямую", () => {
    const store = createEditorStore("scenario", doc("name: s\n"), {});
    store.getState().setText("name: [\n");
    expect(store.getState().parsed.error).not.toBeNull();
    store.getState().edit(["seed"], 1);
    expect(store.getState().text).toBe("name: [\n");
  });

  it("сохранение: новая версия; черновик — проблемы схемы в панель", () => {
    const store = createEditorStore("scenario", doc("name: s\n"), {});
    store.getState().edit(["seed"], 1);
    const sent = store.getState().text;
    store.getState().loadSaved({ ...doc(sent), etag: "e2", valid: false, problem: { title: "Scenario config is invalid", hint: null, issues: [] } }, false);
    expect(store.getState().saved.etag).toBe("e2");
    expect(store.getState().saved.text).toBe(sent);
    expect(store.getState().check?.source).toBe("save");
  });

  it("черновик при открытии — сразу проблемы", () => {
    const store = createEditorStore("scenario", doc("name: s\n", { title: "Scenario config is invalid", hint: null, issues: [] }), {});
    expect(store.getState().check?.title).toBe("Scenario config is invalid");
  });
});
