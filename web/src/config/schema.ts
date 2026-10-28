// Чтение JSON Schema из /v1/meta: только enum и описания для формы. Раскладка и ограничения
// «что реализовано» — в UI и в полях meta (ragglet-15, решение 3).

type Schema = Record<string, unknown>;

function resolve(node: unknown, root: Schema): Schema | null {
  let cur = node as Schema | null;
  for (let guard = 0; cur && guard < 10; guard += 1) {
    if (typeof cur.$ref === "string") {
      const name = cur.$ref.replace("#/$defs/", "");
      cur = ((root.$defs as Record<string, Schema> | undefined)?.[name] ?? null) as Schema | null;
      continue;
    }
    const variants = (cur.anyOf ?? cur.oneOf) as Schema[] | undefined;
    if (variants) {
      cur = variants.find((v) => v.type !== "null") ?? null;
      continue;
    }
    return cur;
  }
  return cur;
}

/** Узел схемы по пути; индексы списков ведут в items. */
export function schemaAt(root: Schema, path: readonly (string | number)[]): Schema | null {
  let node = resolve(root, root);
  for (const seg of path) {
    if (!node) return null;
    if (typeof seg === "number") node = resolve(node.items, root);
    else node = resolve((node.properties as Record<string, unknown> | undefined)?.[seg], root);
  }
  return node;
}

export function enumAt(root: Schema, path: readonly (string | number)[]): string[] {
  const node = schemaAt(root, path);
  return Array.isArray(node?.enum) ? (node.enum as unknown[]).map(String) : [];
}
