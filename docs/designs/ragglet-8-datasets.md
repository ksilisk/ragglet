# ragglet-8 — Поддержка готовых датасетов

## Проблема

Готовые (open-source, индустриальные) retrieval-датасеты сейчас напрямую не
поддерживаются. Чтобы прогнать такой датасет, его приходится вручную приводить к
нашему внутреннему формату:

- документы разложить файлами в `datasets/<name>/corpus/*.md|txt`;
- запросы положить в `data/queries/*.jsonl`;
- `relevant[]` руками/скриптом свести к нашим chunk-level `external_id` вида
  `doc_path#chunk_idx`.

Барьеров на самом деле два, и их полезно разделять:

1. **Формат ввода.** Индексатор умеет только «корпус = директория файлов»
   (`build_index._iter_files` → `parse_plain_text`), а запросы — только наш JSONL.
   Индустриальные датасеты раздаются иначе (JSONL-корпус + отдельные qrels).
2. **Гранулярность ground truth (ядро задачи).** qrels в готовых датасетах
   ссылается на **id документа**, а мы режем документ на чанки и считаем метрики по
   **chunk-level** `external_id`. Прямого соответствия нет.

Симптом второй проблемы уже виден в наших данных: весь
`data/queries/truth_manifest.jsonl` размечен на `...#0`. То есть «простая v1»-эвристика
(релевантен нулевой чанк) фактически зашита в демо-разметку.

## Индустриальный стандарт формата

Де-факто стандарт для retrieval/RAG-оценки — триада **BEIR** (её же использует MTEB;
в ней раздаются MS MARCO, NQ, HotpotQA, SciFact, FiQA, TREC-COVID и ещё ~15 датасетов):

```
corpus.jsonl      {"_id": "d12", "title": "...", "text": "..."}
queries.jsonl     {"_id": "q3",  "text": "..."}
qrels/<split>.tsv query-id <TAB> corpus-id <TAB> score     # TREC-style, с заголовком
```

Рядом — «голый» TREC qrels (`qid 0 docid rel`) и HuggingFace Hub (`BeIR/*`, `mteb/*` —
та же триада). Все они сходятся к одной модели: **corpus / queries / qrels**, где
корпус — это JSONL, а qrels ссылается на **document id**.

**Scope этой задачи:** `directory` (текущий native) + `beir` (из локальной директории).
HuggingFace Hub — вне скоупа (тянет зависимость `datasets`); закладываем его как
отдельный будущий адаптер в том же реестре, но не реализуем сейчас.

## Решение

Механизм общий, **load-native** — читаем внешний датасет напрямую через адаптер, без
конвертации/материализации на диск. Нет дублирования данных и рассинхрона
«папка ↔ jsonl».

### 1. Абстракция corpus-провайдера и query-loader

Индексатор перестаёт ходить по файлам напрямую и итерируется по провайдеру. Директория
— лишь один из провайдеров.

```python
@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    meta: dict[str, Any]       # title и т.п.

class CorpusSource(Protocol):
    def documents(self) -> Iterator[Document]: ...
```

Реестр форматов — точка расширения; новый формат = новый адаптер, а не правки в ядре:

| format      | corpus                         | doc_id          |
|-------------|--------------------------------|-----------------|
| `directory` | обход `path/**`, plain_text    | относительный путь файла |
| `beir`      | `path/corpus.jsonl`            | поле `_id`      |

Для запросов — симметричный loader, возвращающий `QueryItem` (текущая структура). Для
`beir` он читает `queries.jsonl` + `qrels/<split>.tsv`, собирая `truth_ids` как **список
doc_id с score > 0** (бинарная релевантность; graded-score сохраняем на будущее для nDCG,
см. «Вне скоупа»).

### 2. Канонический payload: `source_doc_id`

При индексации в payload каждого чанка добавляем id документа. `external_id` остаётся
chunk-level без изменений.

```python
payload = {
    "external_id": f"{doc.doc_id}#{idx}",   # как сейчас
    "source_doc_id": doc.doc_id,            # НОВОЕ
    "chunk_index": idx,
    "text": chunk_text,
    # doc_path оставляем для directory (== doc_id); для beir тоже = doc_id
}
```

Это и есть «лучшая v1» из задачи — наличие `source_doc_id` позволяет метрикам считать
document-level match, не теряя chunk-level.

### 3. `truth_granularity: doc | chunk`

Гранулярность — это свойство оценки, поэтому живёт на стороне scenario (`dataset`):

```yaml
dataset:
  id: "datasets/scifact"
  format: "beir"            # directory | beir
  split: "test"             # для beir: какой qrels-файл
  truth_granularity: "doc"  # doc | chunk
```

- `chunk` — текущее поведение: `truth_ids` это chunk-`external_id`, точное совпадение строк.
- `doc` — `truth_ids` это doc_id. На этапе метрик каждый retrieved-чанк **проецируется**
  в свой `source_doc_id`, последовательность **дедуплицируется с сохранением лучшего
  (первого) ранга**, и Recall@k / MRR@k считаются уже по doc-последовательности.

### 4. Метрики и проекция

`metrics.py` остаётся granularity-agnostic — он принимает список id и set truth. Вся логика
проекции локализована в `query_loop`:

- `RetrievedItem` уже несёт полный `payload` (`retrieval/engine.py:_to_retrieved`), а
  `final_ids ⊆ retrieved_ids` (rerank/context только фильтруют и переупорядочивают). Значит
  карту `{external_id → source_doc_id}` можно собрать один раз из `retrieved_items` и спроецировать
  любой из `final_ids`/`context_ids`/`reranked_ids`. Правки rerank/context-стадий не нужны.
- Для `doc` granularity: `eval_ids = dedup([map[x] for x in final_ids])`, затем те же
  `recall_at_k` / `mrr_at_k` против set doc-truth.
- **Дедуп обязателен**: без него два чанка одного документа занимают два ранга и завышают
  MRR / искажают recall@k.

Per-query строки получают `final_doc_ids` рядом с `final_ids`, чтобы в артефактах были видны
обе гранулярности.

### 5. Валидация датасета

`bench validate` при `truth_granularity: doc` сверяет relevant-doc_id со множеством
проиндексированных `source_doc_id` (а не chunk-`external_id`). При `chunk` — поведение
как сейчас. Множество `source_doc_id` берём тем же scroll'ом по payload, что и
`external_id`.

### 6. Демо-датасет — оставляем и добавляем

Не переписываем существующую разметку, а держим оба режима как first-class (не shim):

- `data/queries/truth_manifest.jsonl` остаётся как есть — fixture для `truth_granularity: chunk`.
- Добавляем doc-level манифест (`relevant: ["rag_overview_...md"]`) + scenario с
  `truth_granularity: doc` — догфудит основной путь и даёт покрытие `doc`.

Так оба режима реальны и покрыты тестами, существующие baseline-прогоны в `results/`
сохраняют смысл, а `#0`-хак перестаёт быть единственным вариантом разметки.

## Затрагиваемые модули

| Модуль | Изменение |
|--------|-----------|
| `indexing/corpus.py` (новый) | `Document`, `CorpusSource`, `DirectoryCorpus`, `BeirCorpus`, реестр форматов |
| `indexing/build_index.py` | итерация по `CorpusSource`; `source_doc_id` в payload |
| `config/index_profile.py` | блок `source: {format, path, ...}` вместо `corpus`/`parser` |
| `bench/dataset.py` | query-loader с форматами; beir qrels-ридер |
| `config/scenario.py` | `dataset.format`, `dataset.split`, `dataset.truth_granularity` |
| `bench/query_loop.py` | карта chunk→doc, проекция+дедуп для `doc`, `final_doc_ids` |
| `bench/validate_dataset.py` | сверка по `source_doc_id` при `doc` |
| `configs/`, `data/queries/` | beir-пример + doc-level демо-манифест/scenario |

`metrics.py`, `retrieval/*`, `rerank/*`, `context/*` — без изменений.

## Фазы (отдельными коммитами в этот же PR)

1. `CorpusSource` + `DirectoryCorpus`, `source_doc_id` в payload, `source`-блок в index profile
   (directory работает через новую абстракцию; `source_doc_id` появляется в индексе).
2. `truth_granularity: doc|chunk` + проекция/дедуп в `query_loop` + валидатор по `source_doc_id`;
   doc-level демо-манифест/scenario.
3. `BeirCorpus` + beir query/qrels-loader + пример конфигов + README.

## Решения и развилки (зафиксировано)

- **load-native, без конвертации на диск** — единый источник правды, нет дублирования.
- **HuggingFace Hub — вне скоупа**, заложен как будущий адаптер.
- **index profile: cutoff-миграция** `corpus`/`parser` → `source`-блок (мигрируем `demo.yaml`
  и `demo_sharded.yaml`, без compat-слоя).
- **Демо-разметку не переписываем**: оставляем как `chunk`-fixture, добавляем `doc`-вариант.
- **Требуется пересборка индекса** для `doc` granularity (нужен `source_doc_id` в payload) —
  согласуется с существующим требованием пересборки при несовпадении метаданных.

## Вне скоупа

- HuggingFace Hub loader (`format: hf`).
- Graded relevance / nDCG@k (score из qrels сохраняем, но метрики пока бинарные).
- HTML/PDF-парсинг корпуса (ортогонально; отдельная задача про parsers).
