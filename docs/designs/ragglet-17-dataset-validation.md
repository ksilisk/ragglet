# ragglet-17 — Валидация и версионирование датасетов

## Актуальность

После ragglet-8/16 у нас несколько датасетов (`datasets/demo`, `BeIR/nfcorpus`) и
несколько индекс-профилей, а связь «сценарий ↔ индекс» держится только на имени
коллекции. Сейчас защищено немногое:

- `bench run` сверяет с метаданными коллекции только `embedding.model` и `normalize`
  (`indexing/metadata.py:67`);
- `bench validate` — отдельная команда, её нужно не забыть запустить. Она скроллит все
  payload и ловит только случай «doc_id из qrels нет в индексе».

**Тихие сбои, найденные в коде.** Каждый воспроизведён на синтетическом BEIR-наборе:

| # | Что происходит | Где | Последствие |
|---|---|---|---|
| 1 | Повтор `_id` в corpus | `read_corpus` пропускает, `uuid5(chunk_id)` совпадает | второй документ молча **перезаписывает** точки первого |
| 2 | qrels ссылается на `doc_id`, которого нет в corpus | не проверяется | недостижимая истина, recall занижен (ловит только `bench validate`) |
| 3 | qrels ссылается на qid, которого нет в queries | `read_queries` молча выбрасывает | eval-набор меньше задуманного |
| 4 | Битая строка qrels (2 колонки, нечисловой score) | `_read_qrels` молча пропускает | потеря разметки |
| 5 | Повтор `_id` в queries | dict, побеждает последний | подмена текста запроса |
| 6 | Документ из пробелов | `chunk_document` → 0 чанков | релевантный документ не попадает в индекс |
| 7 | `index build` упал на середине | метаданные пишутся **при создании** коллекции, до upsert | половинный индекс выглядит валидным |
| 8 | Коллекция собрана из другого корпуса или после правки corpus | в метаданных индекса нет датасета | метрики на чужом корпусе; ловится, только если id не пересекаются |
| 9 | Сценарий `single` смотрит в одну шарду sharded-индекса | `storage_mode` пишется, но не сверяется | поиск по половине корпуса |
| 10 | Прерванная материализация HF | готовность = «файлы существуют», `qrels/` создаётся заранее (`sources.py:37,73`) | неполный кэш считается готовым |

В артефактах прогона лежит только `scenario.snapshot.*`. По ним нельзя понять, на
каком корпусе, индексе и чанкинге получены цифры. Для статьи это обходили вручную:
`analyze_article_runs.py capture-manifest` хэширует файлы, захардкоженные под demo.
Аудит прямо фиксирует этот пробел (`docs/article-experiments-audit.md:24,147`).

**Данные сейчас чистые.** Прогнал все правила из п. 1 по demo и nfcorpus (все три
split). Нарушений 0, так что строгий режим ничего из текущего не ломает. Риск
не в самих данных, а в эксплуатации: устаревший или половинный индекс, не та
коллекция, правка корпуса. Downstream от этой идентичности зависят D2 (run manifest,
ragglet-23), E3 (реестр запусков), F3 («показывать версии датасета»), H6 и G2.

## Что значит «версия»

**Версия = идентичность по содержимому (content-addressed).** Ручные метки
(`v2`, semver) не используем: они расходятся с данными, а хэш соврать не может.
Идентифицируем три вещи:

| Объект | Идентичность | От чего зависит |
|---|---|---|
| корпус | `corpus.sha256` + число документов | индекс |
| eval-набор | `eval.sha256` — (qid, текст, qrels) после выбора split и `limit` | quality-метрики |
| индекс | `profile_sha256` (параметры сборки) + `corpus.sha256` + `build_id` | retrieval |

**Правило сравнимости.** Quality-метрики двух прогонов можно сравнивать, если у них
равны `corpus.sha256` и `eval.sha256`. Различия индекса (chunking, модель, HNSW)
не блокируются: это и есть экспериментальный фактор. Они только обязаны быть видны
в артефактах. «Несовместимый индекс» — это индекс, собранный не из корпуса сценария,
недособранный или с другой моделью эмбеддингов, чем у сценария. Такой индекс
отклоняется ещё до прогона.

## Решение

### 1. Валидация датасета

Ридер перестаёт молча пропускать данные. Он собирает **все** нарушения, а не падает
на первом, и бросает `DatasetValidationError(issues)`. Принцип: **ошибка** — то, что
тихо искажает метрики или индекс; **предупреждение** — то, что ни на что не влияет,
но попадает в отчёт и manifest.

| Код | Уровень | Правило |
|---|---|---|
| `file.missing` | error | нет corpus/queries/`qrels/<split>`; в сообщении — доступные split |
| `json.invalid`, `record.no_id` | error | строка не JSON-объект; нет или пустой `_id` |
| `corpus.duplicate_id` | error | сбой 1 |
| `queries.duplicate_id` | error | сбой 5 |
| `queries.empty_text` | error | у оцениваемого запроса пустой текст |
| `qrels.malformed` | error | не 3/4 колонки или нечисловой score не в первой строке (заголовок допустим только первой строкой) |
| `qrels.duplicate_pair` | error | пара (qid, doc) повторяется: grade неоднозначен |
| `qrels.unknown_query` | error | сбой 3 |
| `qrels.unknown_doc` | error при grade > 0, warning при grade ≤ 0 | сбой 2; судимый нерелевантный документ на метрики не влияет |
| `qrels.unindexable_doc` | error | сбой 6: у релевантного документа пустые title и text |
| `split.empty` | error | в split нет ни одного запроса с релевантными |
| `corpus.empty_doc` | warning | пустой документ, на который нет ссылок |

Как это выглядит:

```
Dataset validation failed: local:datasets/my (3 errors, 1 warning)
  corpus.duplicate_id  corpus.jsonl:17     _id 'd1' already defined at line 3
  qrels.unknown_doc    qrels/test.tsv:42   q7 -> 'd99': no such document in corpus
  qrels.unknown_doc    qrels/test.tsv:43   q7 -> 'd98': no such document in corpus
  corpus.empty_doc     corpus.jsonl:30     'd5' has empty title and text (not indexed)
```

Где запускается (флагов обхода нет):

- `ragglet dataset validate --source local datasets/my [--split test]` — новая
  команда, Qdrant не нужен. Без `--split` проверяет все split. Тот же отчёт печатает
  `dataset import` по своему результату;
- `index build` — весь датасет со всеми split, **до** создания коллекций. Битый qrels
  должен всплыть до часа эмбеддинга, а не после;
- `bench run` / `bench validate` — corpus и split сценария (п. 5).

Если в публичном датасете найдутся висячие qrels с grade > 0, это сейчас ошибка.
pytrec_eval держит такие документы в знаменателе recall, поэтому вопрос
«сравнимость с опубликованными BEIR-цифрами или достижимость» решаем отдельно, когда
встретим. Обходной путь — исправленная копия как `source: local`.

### 2. Fingerprints

- **Корпус:** потоковый sha256 по каноническим записям в порядке файла:
  `json.dumps([doc_id, title, text], ensure_ascii=False, separators=(",", ":")) + "\n"`.
  Не зависит от порядка ключей, пробелов, лишних полей (`metadata`) и `.gz`. HF parquet
  и raw jsonl одного содержимого дают один хэш. От текста и **порядка** документов
  зависит. Порядок учитываем намеренно: он задаёт порядок вставки в HNSW, а значит
  и приближённый результат поиска. Так что «тот же корпус в другом порядке» — честно
  другой индекс.
- **Eval-набор:** sha256 по отсортированным `[qid, text, sorted(qrels.items())]`
  оцениваемых запросов, то есть после split и `limit`. `shuffle`/`seed` в хэш не
  входят: набор они не меняют.
- **Стоимость:** nfcorpus (5.9 МБ) — 61 мс, около 100 МБ/с. На самых больших BEIR
  (MS MARCO, около 3 ГБ) это порядка минуты на прогон. Кэш fingerprint по
  (size, mtime) пока не делаем, добавим, когда дойдём до таких корпусов.
  Память — O(|doc_id в qrels|), корпус читаем потоком.

### 3. Источник: provenance и атомарная материализация

- `resolve_dataset` возвращает путь и provenance. Для `hf` это revision основного
  репозитория и `-qrels`: sha берём из пути snapshot, в кэше HF это
  `snapshots/<sha>`. Для `local` — путь как задан и абсолютный путь.
- Материализация идёт во временную директорию, затем rename. Последним пишется
  `source.json` (repo, revision, qrels_revision). Нет `source.json` — кэш неполный,
  материализуем заново (сбой 10).
- В `_write_jsonl` заменяем `astype(str)` на `fillna("")`: сейчас null в parquet
  превращается в текст `"None"`/`"nan"`.
- Pin ревизии в конфиге (`dataset.revision`) не добавляем, см. вопросы. От подмены
  данных защищает fingerprint, revision пишем как provenance.

### 4. Контракт метаданных индекса v2 (cutoff)

```jsonc
{"ragglet": {
  "schema": 2,
  "status": "building",                     // → "complete" после последнего upsert
  "build_id": "3f2c…",                      // uuid4, общий для всех targets одной сборки
  "built_at": "2026-10-20T12:00:00Z",
  "index_profile_id": "demo",               // метка; версия — profile_sha256
  "profile_sha256": "…",                    // chunking, embedding.{model,normalize}, index_params,
                                            // storage.{mode,routing.key,число backends}
  "dataset": {"source": "local", "id": "datasets/demo", "revision": null,
              "corpus_sha256": "893aa67b…", "documents": 50, "empty_documents": 0},
  "embedding": {"model": "all-MiniLM-L6-v2", "normalize": true, "dim": 384,
                "probe": [0.0123, …], "library": "sentence-transformers 5.2.3"},
  "chunking": {…}, "index_params": {…},
  "storage": {"mode": "sharded", "backend_id": "shard_0", "backend_count": 2,
              "routing_key": "chunk_id"},
  "points": 27                              // точек в этом target, пишется вместе с complete
}}
```

- **Двухфазная запись** (сбой 7). При создании коллекции пишем `status: building`. После
  последнего upsert — `update_collection(metadata=…)` со `status: complete` и `points`.
  Qdrant 1.17 мёржит metadata, мы перезаписываем ключ `ragglet` целиком.
- **`profile_sha256`** — версия профиля: `id` профиля — свободная строка и при правке
  `chunk_size` не меняется. В хэш не входят endpoint, имена коллекций и `batch_size`:
  на содержимое индекса они не влияют.
- **Версия модели эмбеддингов — через probe.** Имя модели не однозначно: веса на Hub
  обновляются, а `model_card_data.base_model_revision` в sentence-transformers — это sha
  remote `main` (нужна сеть), а не загруженного snapshot. При сборке эмбеддим
  фиксированный probe-текст (сейчас это `vector_size_probe`, нужен только для
  размерности) и сохраняем вектор. В рантайме сравниваем cosine ≥ 0.999. Так ловятся
  дрейф весов, другой pooling и другой `normalize`, без сети. Шум CPU/GPU и fp16 на
  порядки меньше порога. Проверку по имени оставляем: она даёт понятное сообщение.
- Старые индексы без `schema: 2` отклоняются с подсказкой пересобрать. Demo
  пересобирается за секунды, nfcorpus — за минуты на CPU.

### 5. Preflight bench run: совместимость truth ↔ индекс

`bench run` и `bench validate` выполняют один и тот же `preflight(cfg)`. Он идёт
**до** создания каталога артефактов: сейчас ошибка датасета оставляет пустой run dir
(`run_scenario.py:38`).

1. Датасет: валидация corpus и split сценария, fingerprints corpus и eval (п. 1–2).
2. По каждому target, ошибки собираются все сразу:
   - есть `schema == 2`;
   - `status == complete`;
   - `count(exact=True) == points`;
   - `build_id` одинаков у всех targets;
   - `storage.mode` и `backend_count` совпадают с layout сценария (сбой 9);
   - `dataset.corpus_sha256` совпадает с fingerprint корпуса сценария (сбой 8);
   - `embedding.model`/`normalize` совпадают, probe cosine ≥ 0.999 (embedder в run
     всё равно загружается).

```
Index is not compatible with scenario 's0_baseline_rag':
  - target default (chunks_v1): corpus differs — index built from local:datasets/demo
    (893aa67b4bb4, 50 docs), scenario dataset hf:BeIR/nfcorpus (54ef7cf7a72d, 3633 docs)
  Rebuild the index: ragglet index build <index_profile.yaml>
```

**Почему отдельная проверка truth ⊆ index больше не нужна.** Из пп. 1 и 2 это следует
по построению:

- qrels ⊆ индексируемые документы корпуса (п. 1);
- индекс собран ровно из этого корпуса (`corpus_sha256`);
- индекс собран целиком, и после сборки в нём ничего не пропало (`complete` + `count`);
- сценарий видит все targets сборки (`mode`, `backend_count`, `build_id`).

Значит, скролл всех payload из `bench validate` избыточен, и мы его удаляем. Команда
печатает отчёт preflight: секцию датасета и секции targets.

### 6. Артефакты: `manifest.json`

Пишется в run dir сразу после preflight, до запросов: у упавшего прогона
идентичность тоже остаётся.

```jsonc
{
  "dataset": {"source": "hf", "id": "BeIR/nfcorpus", "path": "…",
              "revision": "b5026a0e…", "qrels_revision": "a451b3b2…",
              "split": "test", "limit": 2000,
              "corpus": {"sha256": "54ef7cf7…", "documents": 3633},
              "eval": {"sha256": "…", "queries": 323, "qrels": 12334},
              "warnings": {"corpus.empty_doc": 0, "qrels.unknown_doc": 0}},
  "index": {"build_id": "…", "built_at": "…", "index_profile_id": "nfcorpus_hf",
            "profile_sha256": "…",
            "embedding": {"model": "…", "normalize": true, "dim": 384,
                          "library": "…", "probe_cosine": 1.0},
            "chunking": {…}, "index_params": {…},
            "storage": {"mode": "single",
                        "targets": [{"id": "default", "collection": "nfcorpus_v1", "points": 9696}]}}
}
```

- **Граница с D2 (ragglet-23).** Файл один, A3 пишет секции `dataset` и `index`. D2
  добавит `code` (commit, dirty, lockfile), `environment` и статус прогона рядом, без
  второго файла.
- Fingerprint-колонки в `summary.*` не дублируем: manifest — единственный источник,
  реестр E3 будет индексировать его.

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `datasets/beir.py` | строгий ридер: issues с file:line вместо тихих `continue` |
| `datasets/validation.py` (новый) | `DatasetIssue`, `DatasetValidationError`, правила п. 1 |
| `datasets/fingerprint.py` (новый) | канонические хэши corpus/eval, `profile_sha256` |
| `datasets/sources.py` | provenance, атомарная материализация, `source.json`, `fillna` |
| `bench/dataset.py` | `load_dataset(cfg)` → items + identity + warnings |
| `indexing/build_index.py`, `indexing/qdrant_indexer.py` | валидация до сборки, `build_id`, building → complete, `points`, probe |
| `indexing/metadata.py` | контракт v2 и проверки совместимости (`IndexCompatibilityError(issues)`) |
| `stores/qdrant_store_async.py` | `count(exact=True)` |
| `bench/preflight.py` (новый, заменяет `bench/validate_dataset.py`) | `preflight(cfg)` |
| `bench/run_scenario.py`, `bench/artifacts.py` | preflight до run dir, `manifest.json` |
| `cli/bench.py`, `cli/dataset.py`, `cli/index.py` | отчёты и диагностика, `dataset validate`, exit codes |
| `README.md`, `docs/config-reference.yaml`, `docs/architecure-and-current-state.md` | troubleshooting, правило сравнимости, новый `bench validate` |

Конфиги сценариев и профилей не меняются.

## Тесты (критерий приёмки)

Unit, без Qdrant и torch (фейковый Qdrant-клиент и embedder, как в `test_rerank`):

1. Каждый код из п. 1 на синтетическом датасете: правильный уровень, file:line в
   сообщении, собираются все нарушения.
2. demo проходит без ошибок и предупреждений; `grade ≤ 0` к неизвестному документу —
   warning, не error.
3. Fingerprint корпуса не меняется от порядка ключей, пробелов, лишних полей и `.gz`;
   меняется от текста и порядка. Eval-fingerprint меняется от split, `limit` и grade.
4. HF-кэш без `source.json` материализуется заново; null из parquet превращается в
   `""` (фейковые snapshot-директории, без сети).
5. Сборка: `building` → `complete` + `points`; исключение посреди upsert оставляет
   `building`, и preflight его отклоняет.
6. Preflight: каждое расхождение п. 5 (schema, status, count, build_id, mode/count,
   corpus, model, probe) даёт свою строку; несколько расхождений — все сразу.
7. `run_scenario`: ошибка preflight → run dir не создан; успех → `manifest.json` с
   секциями `dataset`/`index`.
8. `tests/test_configs.py` проходит без изменений конфигов.

E2E на этой ноде не запускается (нет Qdrant и torch, см. ограничения в ragglet-16). В
PR приложу прогон preflight через фейковый harness и попрошу smoke на demo локально:
`index build` → `bench validate` → `bench run`, затем правка одного документа в corpus →
`bench run` отклоняется.

## Фазы (отдельными коммитами в этот же PR)

1. Валидация датасета, fingerprints, `dataset validate`, исправления HF-кэша.
2. Метаданные индекса v2: двухфазная запись, `profile_sha256`, probe, валидация в
   `index build`.
3. Preflight в `bench run`/`bench validate` (без скролла), `manifest.json`, документация.

## Сложности и риски

- **Что считать идентичностью.** Хэш байтов файла ломается от gzip, порядка ключей и
  parquet→jsonl. Отсюда канонический content hash (п. 2) и сознательное решение
  про порядок документов.
- **Стоимость на больших корпусах.** Bench теперь читает корпус целиком: на BEIR-small
  это миллисекунды, на MS MARCO — около минуты. Приемлемо, кэш потом.
- **Версия модели без сети не достаётся.** Probe-вектор решает это поведенчески, но
  порог (0.999) — эвристика.
- **Multi-target.** Согласованность `build_id` и счётчиков по шардам и репликам, а
  также сценарий в другом mode, чем индекс.
- **Cutoff.** Все существующие коллекции придётся пересобрать. Старые run dir без
  `manifest.json` остаются как есть.

## Вне скоупа

- Окружение прогона (commit, lockfile, железо, версии сервисов, статус run) — D2
  (ragglet-23), расширяет тот же `manifest.json`.
- Реестр запусков и поиск по fingerprint — E3 (ragglet-30); UI сравнения — F3.
- Pin ревизии HF в конфиге; кэш fingerprint для больших корпусов; DVC-подобный реестр
  версий датасетов.
- Ключ Redis embedding-кэша не учитывает версию модели (`model + normalize + text`):
  после смены весов из кэша придут старые векторы запросов. Это D1 (ragglet-22).

## Вопросы на ревью

1. Переводить ли `scripts/analyze_article_runs.py` на `manifest.json` сейчас, вместе с
   проверкой «в отчёте только прогоны с равными corpus/eval sha»? Или оставить статейный
   `capture-manifest` как есть: его прогоны сделаны до этого контракта.
2. Pin ревизии для `source: hf` (`dataset.revision`, плюс отдельная ревизия у `-qrels`
   репозитория) — нужен в этой задаче или хватит записи revision в manifest и
   fingerprint-проверки?
