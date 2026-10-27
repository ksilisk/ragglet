/**
 * Список коллекций коротко, как на артбордах: «chunks_v1_shard0, _shard1» — у следующих имён общий
 * с первым префикс до последнего «_» опускается.
 */
export function compactNames(names: readonly string[]): string {
  const unique = [...new Set(names)];
  const [first, ...rest] = unique;
  if (first === undefined) return "";
  return [
    first,
    ...rest.map((name) => {
      let common = 0;
      while (common < first.length && first[common] === name[common]) common += 1;
      const cut = first.lastIndexOf("_", common - 1);
      return cut > 0 ? name.slice(cut) : name;
    }),
  ].join(", ");
}
