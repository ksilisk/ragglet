# ragglet-8 — Поддержка готовых датасетов

## Проблема

Готовые (open-source, индустриальные) retrieval-датасеты сейчас напрямую не
поддерживаются. Чтобы прогнать такой датасет, его приходится вручную приводить к нашему
кастомному формату:

- документы разложить файлами в `datasets/<name>/corpus/*.md|txt`;
- запросы положить в `data/queries/*.jsonl`;
- `relevant[]` руками/скриптом свести к нашим chunk-level `external_id` вида
  `doc_path#chunk_idx`.

Этот кастомный формат — легаси, и он создаёт два барьера:

1. **Формат ввода.** Индексатор умеет только «корпус = директория файлов»
   (`build_index._iter_files` → `parse_plain_text`), а запросы — только наш JSONL.
   Индустриальные датасеты раздаются иначе.
2. **Гранулярность ground truth.** qrels в готовых датасетах ссылается на **id
   документа**, а мы режем документ на чанки и считаем метрики по chunk-level
   `external_id`. Симптом: весь `truth_manifest.jsonl` размечен на `...#0` — «релевантен
   нулевой чанк» фактически зашито в демо-разметку.

**Решение — не надстройка над легаси, а cutoff: переходим на единый индустриальный
формат (BEIR) целиком**, включая наш собственный демо-датасет. Один формат везде —
снаружи и внутри.

## Индустриальный стандарт формата

Де-факто стандарт для retrieval/RAG-оценки — **BEIR**. Важно: это **унифицирующий
формат**, а не список разных схем. MS MARCO, NQ, HotpotQA, SciFact, FiQA, TREC-COVID и
ещё ~15 датасетов раздаются **в одном и том же виде** — триадой файлов:

```
corpus.jsonl      {"_id": "d12", "title": "...", "text": "..."}
queries.jsonl     {"_id": "q3",  "text": "..."}
qrels/<split>.tsv query-id <TAB> corpus-id <TAB> score   # TREC-style, с заголовком
```

Поэтому **адаптер нужен один** (BEIR-ридер), а не по адаптеру на датасет. Остальное — это
способы *доставки* того же формата:

- **MTEB** — агрегированный бенчмарк, для retrieval переиспользует ту же BEIR-загрузку.
- **TREC qrels** — тот же qrels-файл (`qid 0 docid rel`); отличается только числом
  колонок, ридер это автоопределяет.
- **HuggingFace Hub** (`BeIR/*`, `mteb/*`) — те же BEIR-файлы, просто хостятся на Hub.

Ключевой факт для нашего дизайна: **qrels ссылается на `corpus-id` (документ)**, не на
чанк. Это и диктует, как мы считаем метрики.

## Решение

### Принцип: BEIR — единственный канонический формат

Нет параллельных форматов и нет конвертеров «на лету». Канонический формат датасета на
диске — BEIR-триада. Из этого следует:

- **Внешние датасеты** (BEIR/HF) читаются как есть — они уже в нужном формате.
- **Наш демо-датасет** конвертируется в BEIR один раз (cutoff) и коммитится в этом виде;
  старый directory-корпус и `truth_manifest.jsonl` удаляются.
- Индексатор и bench читают **только** BEIR.

### 1. Источник датасета: `local` | `hf` (ссылкой)

Датасет адресуется единообразно — либо локальной директорией, либо id на HuggingFace Hub.
«Показать ссылкой» — рабочий сценарий:

```yaml
dataset:
  source: "hf"            # local | hf
  id: "BeIR/scifact"      # hf repo id  (для local — path к BEIR-директории)
  split: "test"           # какой qrels-файл
  limit: 2000
```

Один **BEIR-ридер** читает `corpus.jsonl` / `queries.jsonl` / `qrels/<split>.tsv`. Источник
— это тонкий слой *резолва локации перед ридером*:

| source  | что делает                                                        |
|---------|-------------------------------------------------------------------|
| `local` | берёт BEIR-директорию по `path`                                   |
| `hf`    | скачивает snapshot репозитория с Hub в кэш → та же BEIR-директория |

**Стоимость HF — низкая** (ответ на вопрос в ревью): HF-датасеты лежат в BEIR-формате, так
что `hf` не добавляет парсер, а только резолв+кэш (`huggingface_hub` snapshot) перед тем же
ридером. Это одна зависимость и небольшая функция загрузки. Точку расширения (новые
резолверы: URL, S3) оставляем в реестре источников, но сверх `local`/`hf` сейчас ничего не
пишем.

### 2. Индексация: `doc_id` first-class, chunk — внутренний

Чанкинг остаётся деталью индексации (`chunk_size`/`overlap`), но identity переосмысляем —
старый `external_id` как носитель ground-truth убираем:

```python
payload = {
    "doc_id": doc["_id"],              # id документа из BEIR corpus — ПО НЕМУ считаем метрики
    "chunk_index": idx,
    "chunk_id": f"{doc['_id']}#{idx}", # внутренний ключ: дедуп в merge, debug
    "title": doc.get("title", ""),
    "text": chunk_text,
}
# point_id = uuid5(chunk_id) — детерминированный, как и раньше
```

- `doc_id` = `corpus._id` — первичный идентификатор для оценки и валидации.
- `chunk_id` — чисто внутренний (нужен merge-дедупу, чтобы один чанк не попал дважды), не
  фигурирует в разметке и не авторится пользователем.
- Пользователь **больше нигде не пишет `doc#chunk` строки** — ground truth приходит из
  qrels на уровне документа.

### 3. Оценка: всегда document-level (без ручки `truth_granularity`)

Раз qrels в BEIR всегда doc-level, делаем единый понятный режим вместо `doc | chunk`:

- retrieval возвращает чанки → каждый **проецируется в свой `doc_id`**;
- последовательность **дедуплицируется с сохранением лучшего (первого) ранга**;
- `Recall@k` / `MRR@k` считаются по doc-последовательности против множества relevant
  `doc_id` из qrels (score > 0).

Passage-level оценку при необходимости даёт **сам датасет** (в passage-датасетах один
BEIR-«документ» = один пассаж), а не отдельная ручка в конфиге. Так конфиг остаётся
дружелюбным: никакого выбора гранулярности, одно поведение.

### 4. Метрики и проекция

`metrics.py` остаётся granularity-agnostic (список id + set truth). Вся проекция
локализована в `query_loop`:

- `RetrievedItem` уже несёт полный `payload` (`retrieval/engine.py:_to_retrieved`), а
  `final_ids ⊆ retrieved_ids` (rerank/context только фильтруют/переупорядочивают). Значит
  карту `{chunk_id → doc_id}` собираем один раз из `retrieved_items` и проецируем любой из
  `final_ids`/`context_ids`/`reranked_ids`. Правки rerank/context не нужны.
- `eval_ids = dedup([map[x] for x in final_ids])`, затем те же `recall_at_k` / `mrr_at_k`
  против set relevant `doc_id`.
- **Дедуп обязателен**: иначе два чанка одного документа занимают два ранга и завышают
  MRR / искажают recall@k.

Per-query строки получают `retrieved_doc_ids` рядом с chunk-уровнем — для отладки.

### 5. Валидация датасета

`bench validate` сверяет relevant `doc_id` из qrels со множеством проиндексированных
`doc_id` (scroll по payload, как сейчас для `external_id`). Ловит классическую ошибку
«qrels ссылается на документ, которого нет в индексе» ещё до прогона.

### 6. Демо-датасет: конвертируем в BEIR (cutoff)

Текущие 38 markdown-файлов и `truth_manifest.jsonl` **переводим в BEIR и удаляем старое**:

```
datasets/demo/
  corpus.jsonl        # один файл = один документ (_id = имя файла без расширения)
  queries.jsonl       # из question'ов манифеста
  qrels/test.tsv      # doc-level: query-id -> relevant doc _id  (из старого ...#0 -> doc)
```

Конверсию делаем одноразовым скриптом-хелпером `ragglet dataset import` (directory →
BEIR), прогоняем на демо и коммитим результат. Этот хелпер — отдельный authoring-инструмент
(удобно собрать корпус из своих markdown), он **не** является runtime-форматом: индекс и
bench всегда читают BEIR. Старый directory-путь из индексатора удаляется.

## Затрагиваемые модули

| Модуль | Изменение |
|--------|-----------|
| `datasets/beir.py` (новый) | BEIR-ридер: corpus/queries/qrels; авто-детект TREC qrels |
| `datasets/sources.py` (новый) | резолв источника `local`/`hf` → локальная BEIR-директория (реестр) |
| `datasets/import_dir.py` (новый) | authoring-хелпер directory → BEIR (для демо и своих корпусов) |
| `indexing/build_index.py` | индекс из BEIR corpus; `doc_id`/`chunk_id` в payload; удалить directory-обход |
| `config/index_profile.py` | блок `dataset: {source, id, ...}` вместо `corpus`/`parser` |
| `config/scenario.py` | `dataset.source`/`id`/`split`; убрать кастомные `query_field`/`truth_field` |
| `bench/dataset.py` | загрузка queries+qrels через BEIR-ридер; `QueryItem.truth_ids` = relevant doc_id |
| `bench/query_loop.py` | карта chunk→doc, проекция+дедуп, `retrieved_doc_ids` |
| `bench/validate_dataset.py` | сверка по `doc_id` |
| `retrieval/engine.py`, `retrieval/merge.py` | переименование `external_id`→`chunk_id` (дедуп по `chunk_id`) |
| `stores/bm25_store.py`, `stores/qdrant_export.py`, `retrieval/backends.py` | тот же рейнейминг |
| `configs/`, `datasets/demo/`, CLI | BEIR-конфиги + сконвертированный демо-датасет + `dataset import` |
| `pyproject.toml` | зависимость `huggingface_hub` для `source: hf` |

`metrics.py` — без изменений.

## Фазы (отдельными коммитами в этот же PR)

1. BEIR-ридер + `local` источник; `doc_id`/`chunk_id` в индексе; `external_id`→`chunk_id`
   рейнейминг; `dataset`-блок в index profile/scenario.
2. Document-level оценка: проекция/дедуп в `query_loop` + валидатор по `doc_id`.
3. Конверсия демо в BEIR (`dataset import`) + удаление легаси (directory-обход,
   `truth_manifest.jsonl`); обновить configs/README.
4. `hf` источник (`huggingface_hub` snapshot) + пример конфига со ссылкой на `BeIR/*`.

## Решения (обновлено по итогам ревью)

- **BEIR — единственный канонический формат**, cutoff без легаси (не надстройка).
- **Один адаптер** на весь BEIR-саёт; MTEB/TREC/HF — способы доставки того же формата.
- **HF — в скоупе**: тонкий resolver (snapshot в кэш) перед тем же ридером; адресация
  датасета ссылкой (`source: hf`, `id: BeIR/scifact`).
- **`external_id` как truth-ключ убираем**: оценка по `doc_id` (BEIR corpus id); `chunk_id`
  — внутренний дедуп-ключ.
- **Без ручки `truth_granularity`**: оценка всегда document-level (passage-level даёт сам
  passage-датасет).
- **Демо конвертируем в BEIR и коммитим**, старый формат удаляем.
- **Требуется пересборка индекса** (новый payload-контракт) — согласуется с существующей
  проверкой метаданных.

## Вне скоупа

- Graded relevance / **nDCG@k** — каноническая метрика BEIR; score из qrels сохраняем в
  индексе/загрузке, но метрики пока бинарные (Recall@k, MRR@k как сейчас). Отдельная задача.
- HTML/PDF-парсинг корпуса (ортогонально; отдельная задача про parsers).
- Прочие источники (URL/S3) сверх `local`/`hf` — заложены точкой расширения, но не пишем.
