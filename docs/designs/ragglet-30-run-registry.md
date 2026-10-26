# ragglet-30 — Реестр прогонов и артефакты

**Коротко.** Запись прогона — это файл `run.json` в его же каталоге
`results/runs/<id>/`, рядом с артефактами ядра. Отдельной базы нет. При старте сервер
сканирует каталог прогонов и держит в памяти индекс: записи и колонки сводки для списка.
Каталог без `run.json` (прогоны CLI до ragglet-37 и все старые) становится
импортированной записью, которую реестр выводит из артефактов и никуда не пишет.
Метаданные (запись, KB) отделены от тяжёлых результатов (per-query, MB): список и опрос
статуса читают только индекс, а per-query читается, когда открыта вкладка «Запросы». API
получает список прогонов и сборок с фильтрами, сортировкой и колонками сводки,
наполненные `?include=summary,manifest`, per-query с фильтрами и рангом первого
релевантного, деталь запроса с оценками из qrels и текстами чанков из Qdrant, а также
файлы и zip каталога. Адрес всего этого — id прогона. Это имя каталога, и оно не меняется
ни со временем, ни при переносе каталога на другую машину.

## Пример, на котором всё объясняется

Сейчас в `results/runs` лежат 55 каталогов, все сделаны CLI. В 44 из них есть
`summary.aggregate.csv`, остальные 11 упали или были прерваны. Каталоги трёх поколений
формата: февральские содержат только CSV и не имеют manifest, июньские — без manifest и
без nDCG, октябрьские — с manifest.

1. **Старт.** `ragglet serve`. Реестр сканирует `results/runs`. `run.json` нет ни в одном
   каталоге, поэтому все 55 записей импортированные (`origin: "import"`): 44 со
   статусом `completed` и 11 со статусом `failed` и ошибкой `RunIncomplete`. На этой
   машине скан занял 30 мс, 0,5 мс на каталог. В каталоги реестр ничего не пишет.
2. **Список.** `GET /v1/runs?scenario=nfcorpus_hf&k=10` возвращает 8 строк. Верхняя:

   ```json
   {"id": "nfcorpus_hf_20261024T180917Z", "scenario": "nfcorpus_hf", "scenario_id": null,
    "origin": "import", "state": "completed", "started_at": "2026-10-24T18:09:17Z",
    "finished_at": "2026-10-24T18:09:21Z", "duration_s": 4.2,
    "dataset": "hf:BeIR/nfcorpus@b5026a0e96e8+a451b3b26d3a", "collection": "nfcorpus_v1",
    "tags": ["beir", "hf", "nfcorpus"], "repeats": 1, "ndcg": 0.315, "recall": 0.150,
    "p50_ms": 152.5, "p95_ms": 232.1, "qps": 95.4, "timeout_rate": 0.0}
   ```

   Числа взяты из `summary.aggregate.json` (`doc_ndcg@10_mean_mean_over_repeats` и т.д.).
   У июньского `s5_distributed_like_20260616T195919Z` поля `ndcg`, `recall` и `qps`
   равны `null`: в его summary таких колонок нет. `recall@10_mean` (0,968) старого
   поколения реестр в `recall` не переименовывает: по ragglet-16 это нынешний `hit@10`.
   Пересчётом реестр не занимается.
3. **Прогон через API.** `POST /v1/runs` со сценарием `nfcorpus_rerank` → 202, id
   `nfcorpus_rerank_20261125T101540Z`. Реестр создаёт каталог с одним файлом `run.json`
   (`state: queued`). Воркер 29 вызывает ядро с этим `run_id`, и ядро кладёт рядом
   snapshot, manifest, per-query и summary. Менеджер сообщает каждый переход в
   `Registry.update`, и `run.json` перезаписывается.
4. **Рестарт.** Сервер перезапустили, пока шёл второй повтор из трёх. При старте у
   записи `running`, менеджер её не знает, `summary.aggregate.csv` нет. Запись
   становится `failed` с ошибкой `ServerRestart`. Повтор 0 успел записаться, и
   `GET …/queries?repeat=0` его отдаёт.
5. **Запросы.** `GET /v1/runs/nfcorpus_hf_20261024T180917Z/queries?miss_at=10` даёт 101
   запрос из 323, ровно столько же, сколько строк с `doc_hit@10 = 0`. У каждой строки
   сервер посчитал `first_relevant_rank` из `final_doc_ids` и `truth_ids`; медиана — 1,
   максимум — 21.
6. **Запрос.** `GET …/queries/PLAIN-307` («Vitamin D: Shedding some light on the new
   recommendations»). Три релевантных документа, у всех оценка 2 по qrels датасета,
   найдены на doc-рангах 1, 5 и 12. В выдаче чанки 1, 3 и 4 — это один документ
   `MED-714`, поэтому у них один `doc_rank` 1. Тексты чанков взяты из payload
   `nfcorpus_v1`: `built_at` коллекции совпадает с `manifest.index.built_at`. После
   пересборки коллекции тексты недоступны (`index.rebuilt`), а оценки и заголовки
   документов остаются: они из датасета.
7. **Файлы.** `GET …/artifacts` — девять файлов с размерами, `GET …/artifacts.zip` —
   весь каталог одним архивом.
8. **Перенос.** Каталог `nfcorpus_rerank_…` скопировали на другой стенд. Там он виден с
   тем же id, состоянием, тегами и ссылкой на сценарий: всё это лежит в `run.json`
   внутри каталога.

## Решение

### 1. Где живут записи: `run.json` в каталоге прогона

| Вариант | Плюсы | Минусы |
|---|---|---|
| **`run.json` в каталоге + индекс в памяти** (выбран) | каталог — единица переноса целиком (рамка 34); один источник правды; CLI-прогоны попадают в реестр тем же сканом | скан при старте; фильтры в Python |
| SQLite в корне артефактов | фильтры и сортировка на SQL | запись отделена от каталога и теряется при переносе; каталоги от CLI и удалённые руками нужно синхронизировать с базой; миграции схемы |
| `run.json` + SQLite как индекс | быстрый старт на десятках тысяч | два источника правды и их расхождения |
| только артефакты, без записи | новых файлов нет | в артефактах нет состояния, времён, ошибки, тегов, ссылки на сценарий; прогон в очереди без каталога не виден |

Почему так. Пользователь один, прогонов тысячи максимум. Скан стоит 0,5 мс на каталог
(замер на 55 нынешних), тысяча прогонов — полсекунды при старте, в потоке. Фильтр и
сортировка тысячи записей в памяти занимают микросекунды. SQL ничего не ускорит, зато
добавит второй источник правды: CLI до 37 пишет каталоги в обход API, а каталог можно
удалить или принести руками.

**`run.json`** — это `RunRecord` из протокола 28, сериализованный через
`pydantic.TypeAdapter`. Три поля в файл не попадают: `run_dir` (всегда
`artifacts_dir / id`), `metrics` (производное, п. 5) и `origin`.

```jsonc
// results/runs/nfcorpus_rerank_20261125T101540Z/run.json
{
  "format": 1,                                  // версия файла записи; незнакомая — см. ниже
  "id": "nfcorpus_rerank_20261125T101540Z",
  "name": "nfcorpus_rerank",
  "scenario_id": "nfcorpus_rerank",             // ссылка на библиотеку из запроса
  "etag": "c07d5a1e9b20f3d4",                   // версия текста сценария при постановке
  "tags": ["rerank"],
  "dataset": "hf:BeIR/nfcorpus@b5026a0e96e8+a451b3b26d3a",
  "collection": "nfcorpus_v1",
  "plan": {"repeats": 3, "queries_per_repeat": 323, "total": 969, "load": {"type": "closed", "concurrency": 16}},
  "warnings": [{"code": "index.not_green", "severity": "warning", "path": null,
                "where": "target default (nfcorpus_v1)", "message": "collection status is yellow: …"}],
  "state": "completed",
  "enqueued_at": "2026-11-25T10:15:40Z",
  "started_at": "2026-11-25T10:16:52Z",
  "finished_at": "2026-11-25T10:22:55Z",
  "failure": null,
  "final_progress": {"stage": "queries", "done": 969, "total": 969, "first_at": "…", "updated_at": "…"},
  "scenario": {"version": 1, "name": "nfcorpus_rerank", "…": "…"}   // cfg.model_dump(), как snapshot
}
```

- **Каталог создаёт реестр при постановке**, а не ядро при старте: записи `queued` нужно
  где-то жить. `create_run` резервирует id через `mkdir(exist_ok=False)`: если каталог
  занят (CLI, другая постановка), берётся следующий суффикс `-2`, `-3`. Такая
  резервация атомарна, в отличие от проверки «есть ли» с последующим созданием. Ядро
  принимает уже созданный каталог (п. 11). Прогон, отменённый в очереди, оставляет
  каталог с одним `run.json`. Это и есть его запись.
- **Пишет только сервер.** Запись атомарна: временный `.run.json.<uuid>.tmp` и
  `os.replace`, под `anyio.Lock` реестра. Сервер — один процесс (28), поэтому
  межпроцессных блокировок нет. Ядро, воркер 29 и CLI `run.json` не трогают: воркер
  сообщает переходы менеджеру, менеджер в процессе сервера зовёт `Registry.update`.
- **Битый или незнакомый `run.json`** (ручная правка, `format` из будущего) не
  перезаписывается. Сервер пишет предупреждение в свой лог, а каталог импортируется как
  каталог без записи (п. 3).
- **Сборки.** Каталога у сборки нет, поэтому её запись — файл
  `results/builds/<id>.json` того же устройства (`BuildRecord`). Новая настройка
  `builds_dir: Path = Path("results/builds")` (`RAGGLET_BUILDS_DIR`). Сборки CLI до 37 в
  реестр не попадают: от них не остаётся файлов. Их след — метаданные коллекции, их
  показывает `GET /v1/indexes`.

Индекс в памяти: `dict[id, RunRecord]` и такой же для сборок. Его строит скан при
первом обращении к реестру. Все методы `FileRegistry` асинхронные, файловый ввод-вывод
уходит в `anyio.to_thread`.

### 2. id и связи

id прогона — `<name>_<UTC постановки>[-N]` (решено в 28). Это имя каталога. id
стабилен: он не переиспользуется (резервация через `mkdir`), не меняется со временем и
переезжает вместе с каталогом. У импортированного каталога id — имя каталога как есть.

| Связь | Поле записи | Зачем |
|---|---|---|
| сценарий библиотеки | `scenario_id`, `etag` | ссылка «сценарий»; пометка «изменился после прогона» (п. 6) |
| конфиг прогона | `scenario` = `scenario.snapshot.json` | неизменяем; «Повторить», «Открыть как сценарий» |
| датасет | `dataset` (label); `manifest.dataset` (ревизия, хэши) | фильтр; оценки qrels в детали запроса (п. 8) |
| индекс | `collection` (primary); `manifest.index` (`profile_id`, `built_at`) | фильтр; тексты чанков, пока коллекция той же сборки (п. 8) |
| история сценария | `name`, `enqueued_at` | `GET /v1/runs?scenario=nfcorpus_hf` — все прогоны сценария во времени, включая CLI |

У сборки — `profile_id`, `etag`, `dataset`, `targets`, `result`.

### 3. Каталоги без записи: импорт

Каталог в `artifacts_dir` без `run.json`, но со `scenario.snapshot.json`, считается
прогоном CLI и импортируется. Каталог без snapshot прогоном не считается, как и имена
на `.`. Импортированная запись нигде не сохраняется: при каждом скане она выводится
заново, поэтому смена правил вывода не требует миграции.

| Поле | Откуда |
|---|---|
| `id` | имя каталога |
| `name` | `snapshot.name`, иначе часть id до `_<время>` |
| `scenario` | snapshot как dict, без валидации: 46 из 55 нынешних snapshot не проходят текущую схему (`dataset.query_field` и т.п.) |
| `scenario_id`, `etag`, `plan` | `null`: у CLI-прогона ссылки на библиотеку, текста сценария и плана preflight нет |
| `tags` | `[]`; теги сценария лежат в snapshot, фильтр `tag` их учитывает |
| `dataset` | `describe_dataset(source или "local", id, manifest.dataset.revision или snapshot.dataset.revision)` |
| `collection` | `snapshot.storage.backend.collection`: primary по правилу `build_storage_layout` |
| `enqueued_at`, `started_at` | время из id. У CLI это момент создания каталога, после загрузки моделей; если имя не по шаблону — mtime snapshot |
| `state` | `completed`, если есть `summary.aggregate.csv` (его ядро пишет последним); иначе `failed` |
| `failure` | у неполного: `RunIncomplete`, «no summary.aggregate.csv: the run crashed, was killed or is still running from the CLI» |
| `finished_at` | mtime `summary.aggregate.csv`; у неполного — `null` |
| `warnings` | `[]` |
| `origin` | `"import"` (у записей из `run.json` — `"api"`) |

- **Почему `failed`, а не новое состояние `incomplete`.** `TaskState` — состояние задачи
  менеджера (файл 29), и UI уже различает пять значений. С точки зрения реестра прогон
  не завершился, и `error.message` говорит прямо, что известно и что нет. Если CLI-прогон
  ещё идёт, при следующем скане он станет `completed`: неполные импортированные каталоги
  перепроверяются на каждом скане (один `stat`).
- **Длительность по mtime.** Если каталог скопировали без сохранения времён (`cp` без
  `-p`), время конца будет неверным. Импорт — переходный путь до 37, у прогонов через API
  времена берутся из записи.
- **Когда скан.** Полный — при первом обращении к реестру после старта. Перед каждым
  `list_runs` — `scandir` каталога (миллисекунды на тысячах): новые имена
  импортируются, исчезнувшие удаляются из индекса, неполные импортированные
  перепроверяются. `get_run` с незнакомым id ищет такое имя в листинге `artifacts_dir` и
  импортирует этот один каталог. Путь из id запроса напрямую не строится.

### 4. Восстановление после рестарта

В `create_app` появляется `lifespan`. При старте он вызывает
`recover(registry, jobs)` из `registry/recovery.py`. Функция пользуется только методами
протоколов и поэтому работает с любой реализацией. Для каждой записи `queued` или
`running` (прогоны и сборки), которой `jobs.status(ref)` не знает:

| Запись | Признак | Становится |
|---|---|---|
| прогон | есть `summary.aggregate.csv` | `completed`, `finished_at` = mtime файла: воркер успел закончить, а переход до сервера не дошёл |
| прогон `running` | нет `summary.aggregate.csv` | `failed`, `ServerRestart`: «server restarted while the run was running; artifacts of finished repeats are kept» |
| прогон `queued` | | `failed`, `ServerRestart`: «server restarted before the run started; use Repeat to run it» |
| сборка `running` | | `failed`, `ServerRestart`: «… the collection may be left without metadata: rebuild it» |
| сборка `queued` | | `failed`, `ServerRestart` |

`finished_at` у упавших — время восстановления. Запись меняется через обычный `update`,
поэтому `run.json` обновляется так же, как при любом переходе. Сами задачи заново не
ставятся (см. «Рассмотренные альтернативы»). `TestClient` без `with` lifespan не
запускает, поэтому тесты 28 это не затрагивает.

### 5. Список прогонов и сборок

`GET /v1/runs` → `RunList`:

| Параметр | Поле `RunQuery` | Смысл |
|---|---|---|
| `state` | `state` (есть) | одно состояние |
| `scenario` | `name` (есть) | имя сценария, то есть история сценария |
| `scenario_id` | `scenario_id` (есть) | ссылка на библиотеку |
| `dataset` | `dataset` (новое) | `hf:BeIR/nfcorpus` совпадает с label без ревизии и с любой ревизией (`hf:BeIR/nfcorpus@…`) |
| `collection` | `collection` (новое) | primary-коллекция, точное совпадение |
| `tag` | `tag` (есть) | в тегах записи или сценария |
| `since`, `until` | `since`, `until` (есть) | период по `enqueued_at`, UTC ISO |
| `q` | `text` (новое) | подстрока без учёта регистра в id, имени, описании сценария, тегах, датасете, коллекции |
| `sort` | `sort` (новое) | `enqueued_at`, `started_at`, `finished_at`, `duration`, `scenario`, `state`, `dataset`, `collection`, `ndcg`, `recall`, `p50`, `p95`, `qps`, `timeout_rate`; `-` в начале — по убыванию; по умолчанию `-enqueued_at`; `null` всегда в конце |
| `k` | `k` (новое) | k для `ndcg` и `recall`, по умолчанию 10 |
| `offset`, `limit` | есть | `limit` ≤ 500, по умолчанию 50 |

```python
class RunItem(BaseModel):
    id: str
    scenario: str                          # name
    scenario_id: str | None
    origin: Literal["api", "import"]
    state: TaskState
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_s: float | None               # finished_at − started_at
    dataset: str
    collection: str
    tags: list[str]                        # теги записи и сценария
    repeats: int | None                    # из summary.aggregate
    ndcg: float | None                     # doc_ndcg@k, среднее по повторам
    recall: float | None                   # doc_recall@k
    p50_ms: float | None                   # lat_total_p50
    p95_ms: float | None                   # lat_total_p95
    qps: float | None                      # throughput_qps
    timeout_rate: float | None

class RunList(BaseModel):
    k: int
    total: int
    items: list[RunItem]
```

**Колонки сводки.** Когда реестр видит прогон завершённым (скан или `update` в
`completed`), он один раз читает `summary.aggregate.json` и держит в памяти
`RunMetrics(repeats, means)`, где `means[col] = aggregate[f"{col}_mean_over_repeats"]`
для всех числовых колонок, без NaN. Колонка списка — это ключ `means`:

| Колонка | Ключ | Где его нет |
|---|---|---|
| `ndcg` | `doc_ndcg@{k}_mean` | до ragglet-21; `k` нет в `metrics.quality.k` прогона |
| `recall` | `doc_recall@{k}_mean` | до ragglet-16: там `recall@{k}_mean`, а это по ragglet-16 нынешний `hit@k` |
| `p50_ms`, `p95_ms` | `lat_total_p50`, `lat_total_p95` | 50 или 95 нет в `metrics.percentiles` |
| `qps` | `throughput_qps` | до ragglet-19 |
| `timeout_rate` | `timeout_rate` | февральские прогоны |

Нет ключа — `null`, в UI «—». Ничего не пересчитывается и не переименовывается: у
старых поколений метрики определены иначе (старый `recall@k` — это `hit@k`), и в
ragglet-19 старые артефакты тоже не мигрировали. Почему в памяти, а не в `run.json`: у прогонов API и CLI
один путь, а при изменении набора колонок не нужно мигрировать файлы. Тысяча
aggregate — полсекунды при старте.

Активные прогоны и так стоят сверху при сортировке по умолчанию: менеджер выполняет
очередь по порядку постановки. При другой сортировке UI закрепляет их отдельным
запросом `state=running` и `state=queued`. Колонки списка не накладывают прогресс
менеджера: он есть в `GET /v1/tasks`.

`GET /v1/builds` → `BuildList {total, items}`. Строка `BuildItem`: `id`, `profile`,
`profile_id`, `state`, времена и `duration_s`, `dataset`,
`targets: list[CollectionRef]`, `documents` (из `result`). Фильтры: `state`,
`profile_id`, `dataset`, `collection` (среди целей), `since`, `until`, `q`; сортировка
по временам, `duration`, `profile`, `state`; пагинация как у прогонов. Для этого в
`BuildQuery` добавляются `dataset`, `collection`, `since`, `until`, `text`, `sort`.

`RunBrief` (последний прогон сценария в `GET /v1/scenarios`) получает
`ndcg_at_10: float | None`, как просит ragglet-15. «Недавние» на Стенде — это
`GET /v1/runs?limit=5`.

### 6. Деталь прогона: `?include=`, «сценарий изменился»

- `include=summary` → `RunSummaryOut(repeats=summary.json, aggregate=summary.aggregate.json)`,
  файлы как есть. Нет `summary.json` (прогон идёт, упал, в очереди) — `summary: null`.
  Нет только aggregate (февральский `baseline_qdrant_20260221T134757Z`) — `aggregate: {}`.
  UI разбирает суффиксы `_mean/_std/_min/_max_over_repeats` одним правилом.
- `include=manifest` → `manifest.json` как dict; у прогонов без manifest — `null`.
- NaN и ±inf в JSON-артефактах заменяются на `null` при чтении: `JSONResponse` Starlette
  NaN не сериализует.
- `RunStatus` получает два поля, оба считаются без чтения артефактов, поэтому опрос
  раз в секунду остаётся дешёвым:

  ```python
  origin: Literal["api", "import"]
  scenario_file: Literal["unchanged", "changed", "deleted"] | None
  ```

  `scenario_file` сравнивает `etag` записи с `etag` текущего файла
  `scenarios/<scenario_id>.yaml` (`ConfigLibrary.find`). Это чтение и хэш одного YAML
  на пару КБ. `null` — у прогона нет `scenario_id` (импорт или запуск без ссылки). Для
  вкладки «Конфиг» и шапки этого хватает: `changed` — пометка «сценарий изменился после
  прогона», `deleted` — «сценарий удалён, snapshot остался».
- У импортированных `etag` и `plan` равны `null`. Поэтому `RunRecord.etag`,
  `RunRecord.plan`, `RunStatus.etag` и `RunStatus.plan` становятся Optional (п. 10).

### 7. Запросы прогона: `GET /v1/runs/{id}/queries`

| Параметр | Смысл | Если в файле нет колонки |
|---|---|---|
| `repeat` | номер повтора, по умолчанию 0 | — |
| `outcome` | `ok` / `timeout` / `error` | 422 `request.filter`: у прогонов до ragglet-19 есть `timed_out`, а не `outcome` |
| `miss_at` | промах при k: `first_relevant_rank` пуст или больше k | 422, если нет `final_doc_ids` (февральские и июньские прогоны) |
| `min_total_ms` | `lat_total_ms ≥ X` | — |
| `q` | подстрока в `qid` или `query`, без учёта регистра | — |
| `sort` | любая скалярная колонка файла или `first_relevant_rank`, `-` — по убыванию; по умолчанию порядок файла (порядок выполнения повтора); `null` в конце | 422 `request.sort` |
| `offset`, `limit` | `limit` ≤ 1000, по умолчанию 100 | — |

```python
class QueryPageOut(BaseModel):
    repeat: int
    repeats: list[int]                     # какие повторы записаны: у упавшего — «частичные результаты»
    source: str                            # "per_query.repeat0.parquet"
    columns: list[str]                     # колонки строки: у старых поколений их меньше
    total: int
    items: list[dict[str, Any]]
```

- **Файл.** Если есть parquet, читается он, иначе CSV. В CSV списки записаны
  Python-repr и разбираются `ast.literal_eval` по фиксированному набору колонок
  `LIST_COLUMNS` (`truth_ids`, `retrieved_ids`, `retrieved_doc_ids`, `reranked_ids`,
  `context_ids`, `final_ids`, `final_doc_ids`, `rerank_scores`), а не угадыванием по
  первой `[`: текст запроса тоже может начинаться со скобки. `qid`, `query`, `outcome`,
  `timeout_stage` читаются строками, иначе `qid` вида `0042` стал бы числом 42. Новая
  колонка-список в CSV требует строки в `LIST_COLUMNS`, parquet — нет.
- **Строка списка** — все скалярные колонки файла плюс `first_relevant_rank`. Списки и
  длинные тексты (`context_text`, `prompt_text`, `answer`, `raw_generation_response`)
  отдаёт деталь запроса. NaN → `null` (через `DataFrame.to_json`, он же превращает
  массивы parquet в списки).
- **`first_relevant_rank`** — позиция первого документа из `truth_ids` в
  `final_doc_ids`, с 1, или `null`. Это ровно то условие, по которому считается
  `doc_hit@k`: на nfcorpus `miss_at=10` даёт те же 101 строку, что `doc_hit@10 = 0`. У
  прогонов без `final_doc_ids` колонки нет.
- **Что в строке есть.** Отметки кэша `emb_cache` / `ret_cache` (`hit`, `miss`, пусто)
  пишутся с ragglet-22 и отдаются как обычные колонки; показывать ли их — решение 31.
  Чего нет и что API не обещает: score retrieval, источник (`vector` / `keyword`).
- **Стоимость.** nfcorpus, 323 строки: parquet читается за 5 мс, CSV со списками — за
  70 мс. Разобранный файл хранится в LRU на 4 файла, ключ — `(путь, mtime, размер)`.
  Фильтр, сортировка и страница считаются по нему, повторное чтение файла не нужно.
- **Нет файлов per-query** (прогон в очереди, ещё не закончил повтор,
  `save_per_query: false`) → 404 `run.no_per_query`. Нет такого повтора → 404
  `run.repeat_not_found`, в сообщении перечислены записанные повторы.

### 8. Деталь запроса: `GET /v1/runs/{id}/queries/{qid}?repeat=0`

```python
class TruthDoc(BaseModel):
    doc_id: str
    grade: int | None                      # из qrels датасета; None — см. grades
    title: str | None                      # из корпуса датасета
    doc_rank: int | None                   # ранг в final_doc_ids; None — не найден

class ResultChunk(BaseModel):
    rank: int                              # позиция в final_ids, с 1
    chunk_id: str
    doc_id: str                            # из chunk_id «<doc_id>#<chunk_index>» — контракт сборки
    doc_rank: int                          # ранг документа в final_doc_ids: по нему считаются doc-метрики
    relevant: bool                         # doc_id в truth_ids строки
    grade: int | None
    rerank_score: float | None             # по chunk_id из reranked_ids/rerank_scores
    in_context: bool | None                # None — context выключен
    title: str | None                      # payload Qdrant, иначе заголовок документа из корпуса
    text: str | None                       # payload Qdrant; None — см. texts

class Availability(BaseModel):
    available: bool
    code: str | None                       # почему нет
    message: str | None

class QueryDetail(BaseModel):
    qid: str
    repeat: int
    query: str
    row: dict[str, Any]                    # строка per-query целиком: задержки, метрики, контекст, ответ
    truth: list[TruthDoc]
    results: list[ResultChunk]
    grades: Availability                   # оценки и заголовки из датасета
    texts: Availability                    # тексты чанков из Qdrant
```

**Оценки и заголовки — из датасета прогона.** Правила:

- Источник ищется **только локально**. Для этого появляется
  `resolve_dataset(source, id, revision, offline=True)`: для `hf` это
  `snapshot_download(local_files_only=True)` с ревизией из `manifest.dataset.revision`
  (для BeIR она составная, `<data>+<qrels>`), для `local` — путь `dataset.id`
  относительно рабочего каталога сервера. Если датасета нет в кэше, получаем
  `grades.code = "dataset.not_local"`. Сеть по клику на строку не трогаем.
- Затем `parse_dataset(path, [split])` и `eval_items(split, limit)`, и хэши сверяются с
  `manifest.dataset`. Не совпал `eval_sha256` — `dataset.changed`, оценок нет. Не совпал
  `corpus_sha256` — заголовков нет. Сверка бесплатная: `parse_dataset` считает хэши всё
  равно. Для `hf` её и так гарантирует ревизия, для `local` — только она.
- Нет manifest (прогоны до ragglet-17) — `run.no_manifest`. Сверить не с чем, поэтому
  оценок нет. `relevant` и `truth` по-прежнему берутся из `truth_ids` строки.
- Разобранный датасет кэшируется: qrels split и словарь `doc_id → title` (второй проход
  по корпусу), LRU на 2 датасета, загрузка в потоке. Для nfcorpus (3,6 тыс. документов)
  это доли секунды при первом клике.

**Тексты чанков — из Qdrant, пока коллекция той же сборки.**

- Цели берутся из snapshot: `ScenarioConfig.model_validate(record.scenario)` и
  `build_storage_layout`. Если snapshot не проходит текущую схему —
  `run.snapshot_outdated`.
- По каждой цели вызывается `collection_info()`, и `ragglet.built_at` и
  `index_profile_id` сравниваются с `manifest.index`. Причины недоступности:
  `run.no_manifest`, `index.missing`, `index.no_metadata` (идёт пересборка или
  коллекция чужая), `index.rebuilt` (в сообщении оба `built_at`), `index.unreachable`.
- Чанки читаются новым `AsyncQdrantStore.payloads(chunk_ids)`: `client.retrieve` по
  `uuid5(NAMESPACE_URL, chunk_id)`, без векторов. В режиме `sharded` чанк лежит в одной
  из целей, поэтому опрашиваются все цели той же сборки. Таймаут — `probe_timeout_s`.
- Endpoint берётся из snapshot прогона, а не из `settings.qdrant_endpoints`: прогон шёл
  именно туда.

Ни одна причина недоступности не даёт ошибку ответа: деталь всегда 200, а `grades` и
`texts` говорят, чего в ней нет. 404 — только если нет прогона, повтора или `qid`
(`query.not_found`).

### 9. Артефакты

| Endpoint | Ответ |
|---|---|
| `GET /v1/runs/{id}/artifacts` | `list[ArtifactOut{name, size, modified_at}]` — файлы каталога, без подкаталогов и имён на `.` (временные файлы атомарной записи), включая `run.json` и лог 29, если он там |
| `GET /v1/runs/{id}/artifacts/{name}` | файл как есть (`FileResponse`), тип по расширению: `json`, `yaml`, `text/csv`, `application/vnd.apache.parquet` |
| `GET /v1/runs/{id}/artifacts.zip` | весь каталог: `<id>.zip` с записями `<id>/<файл>`, `ZIP_DEFLATED`; собирается в потоке во временный файл (каталоги — единицы и десятки МБ: все 55 нынешних весят 42 МБ) и отдаётся потоком |

- Путь к файлу строится только из имён, которые вернул листинг каталога, а id — только
  из индекса реестра. Поэтому `..` и абсолютные пути не проходят. Неизвестное имя —
  404 `artifact.not_found`.
- У идущего прогона отдаётся то, что уже записано. Ядро пишет файлы атомарно (п. 11),
  поэтому каждый файл в листинге целый.
- `Artifact` в протоколе получает `modified_at`.

### 10. Интерфейс: что меняется

`registry/protocol.py` (только расширения, смысл существующего не меняется):

```python
@dataclass(frozen=True)
class RunMetrics:
    repeats: int | None
    means: dict[str, float]                # колонка summary -> среднее по повторам; без NaN

@dataclass(frozen=True)
class RunRecord:
    ...
    etag: str | None                       # None — импортирован: текста сценария не было
    plan: RunPlan | None                   # None — импортирован: preflight при постановке не было
    ...
    origin: Literal["api", "import"] = "api"   # не пишется в run.json
    metrics: RunMetrics | None = None          # производное из summary.aggregate.json; не пишется

@dataclass(frozen=True)
class RunQuery:                            # + dataset, collection, text, sort="-enqueued_at", k=10
class BuildQuery:                          # + dataset, collection, since, until, text, sort
class QueryFilter:                         # + miss_at, min_total_ms, text, sort

@dataclass(frozen=True)
class QueryPage(Page[dict[str, Any]]):     # queries() возвращает его
    repeat: int
    repeats: list[int]
    source: str
    columns: list[str]

@dataclass(frozen=True)
class Artifact:
    name: str
    size: int
    modified_at: datetime                  # новое

class Registry(Protocol):
    ...
    async def query_row(self, run_id: str, repeat: int, qid: str) -> dict[str, Any] | None: ...  # новое
```

`InMemoryRegistry` остаётся тестовым двойником для тестов 28: новые поля фильтров он
понимает минимально, `query_row` возвращает `None`. Деталь запроса собирает отдельный
`QueryInspector` (`registry/inspect.py`: датасет и Qdrant). Роутер создаёт его лениво в
`Services`. Реестр отвечает за хранение, а не за походы в Qdrant.

`api/schemas.py`: `RunItem`, `RunList`, `BuildItem`, `BuildList`, `QueryPageOut`,
`QueryDetail` и вложенные модели, `ArtifactOut`; `RunStatus.origin`,
`RunStatus.scenario_file`, Optional `RunStatus.etag` и `plan`; `RunBrief.ndcg_at_10`.
`create_app`: по умолчанию `FileRegistry(settings.artifacts_dir, settings.builds_dir)`
и `lifespan` с `recover`. Новые endpoints: `GET /v1/runs` в `routers/runs.py`,
`GET /v1/builds` в `routers/builds.py`, запросы и артефакты в новом
`routers/results.py`.

Новые коды проблем: `run.no_per_query`, `run.repeat_not_found`, `query.not_found`,
`artifact.not_found` (404); `request.filter`, `request.sort` (422).

### 11. Доработки ядра

1. **Каталог прогона.** С `run_id` каталог может уже существовать (его создал реестр),
   но в нём не должно быть `scenario.snapshot.json`. Иначе `FileExistsError`, то есть
   защита из 28 от смешения артефактов двух прогонов сохраняется. Без `run_id` (CLI до
   37) имя получает тот же суффикс `-2`, `-3` через `mkdir(exist_ok=False)`. Сейчас два
   CLI-прогона в одну секунду пишут в один каталог, а CLI может попасть в каталог,
   зарезервированный API. `unique_id` переезжает из `registry/memory.py` в
   `bench/artifacts.py`.
2. **Атомарная запись артефактов.** snapshot, manifest, per-query, `summary.*`
   пишутся во временный `.<имя>.<uuid>.tmp` и переименовываются `os.replace`. Тогда
   `summary.aggregate.csv` на диске — честный признак завершения, а чтение per-query
   идущего прогона не видит полфайла.
3. **`resolve_dataset(…, offline=False)`**: при `offline=True` у `hf` —
   `local_files_only=True`, нет в кэше — `None` вместо скачивания.
4. **`AsyncQdrantStore.payloads(chunk_ids) -> dict[str, dict]`** (п. 8).

CLI это не меняет, кроме суффикса имени при совпадении секунды.

## Требования к ragglet-29

Файлы 29 эта задача не трогает, в `jobs/protocol.py` правок не нужно. Нужно следующее:

1. **Переходы.** Терминальный `update` несёт `finished_at`, у прогона — `final_progress`.
   Всё это уже есть в `TaskChange`. `completed` ставится только после того, как
   `run_scenario` вернул результат.
2. **Каталог.** Воркер вызывает `run_scenario(cfg, run_id=record.id)`. Каталог к этому
   моменту существует, в нём лежит `run.json`. После фазы 1 этой задачи ядро это
   допускает.
3. **Lifespan.** В `create_app` реестр добавляет `lifespan` с `recover(registry, jobs)`.
   Если 29 тоже поднимает воркер в lifespan, порядок такой: сначала менеджер, потом
   `recover`, а при остановке менеджер гасит процессы задач. Кто из нас вольётся вторым,
   тот и объединяет lifespan.
4. **Лог (предложение).** Если лог задачи пишется в файл, удобно класть его в каталог
   прогона (`results/runs/<id>/task.log`) и рядом с записью сборки
   (`results/builds/<id>.log`). Тогда лог переживает рестарт, переезжает с каталогом и
   сам попадает во «Файлы» и в zip. Решает 29.
5. **Процессы задач умирают вместе с сервером.** Если процесс задачи переживёт рестарт
   и допишет каталог, запись к этому времени уже будет `failed`, и правило «есть
   aggregate → `completed`» из п. 4 не сработает.

## Расхождения с ragglet-15 и ragglet-28

| Было | Здесь | Почему |
|---|---|---|
| 28: каталог прогона создаёт ядро при старте | реестр при постановке, с `run.json`; ядро принимает готовый каталог без snapshot | записи `queued` нужно жить в каталоге, чтобы он оставался единицей переноса; резервация id через `mkdir` атомарна |
| 28: `RunStatus.etag: str`, `plan: RunPlanOut` | Optional | у импортированных CLI-прогонов нет ни текста сценария, ни плана |
| 15: «сценарий изменился после прогона — по `updated_at` файла» | по `etag` (как решено в 28), поле `scenario_file` с `deleted` | mtime меняется без правки |
| 15: `GET /v1/runs?…&from&to&page` | `since`, `until`, `offset`, `limit` | имена из `RunQuery` 28 |
| 15: `GET /v1/runs/{id}` отдаёт `partial` и список артефактов | `repeats` в ответе `queries` и `GET …/artifacts` | опрос раз в секунду не читает каталог |
| 15: «флаг попадания кэша у запроса UI не обещает» | `emb_cache`, `ret_cache` есть в строке с ragglet-22 и отдаются как колонки | показ — решение 31 |
| 15: состояние прогона по каталогу неразличимо | у импортированных — `completed` по `summary.aggregate.csv`, иначе `failed` / `RunIncomplete` с честным сообщением | ничего не выдумываем и не добавляем состояние |

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `registry/protocol.py` | поля и `query_row` (п. 10) |
| `registry/files.py` (новый) | `FileRegistry`: `run.json`, записи сборок, скан, индекс, список, summary, manifest, артефакты |
| `registry/importer.py` (новый) | запись из каталога без `run.json` |
| `registry/metrics.py` (новый) | `RunMetrics` из aggregate, колонки списка, сортировка |
| `registry/per_query.py` (новый) | чтение parquet/CSV, LRU, фильтры, `first_relevant_rank` |
| `registry/inspect.py` (новый) | `QueryInspector`: датасет, qrels, заголовки, тексты из Qdrant |
| `registry/recovery.py` (новый) | `recover(registry, jobs)` |
| `registry/memory.py` | новые поля и методы минимально; `unique_id` — из `bench/artifacts.py` |
| `api/routers/runs.py`, `builds.py` | `GET /v1/runs`, `GET /v1/builds`; `origin`, `scenario_file`, Optional `etag` и `plan` |
| `api/routers/results.py` (новый) | запросы, деталь запроса, артефакты, zip |
| `api/routers/library.py` | `RunBrief.ndcg_at_10` |
| `api/schemas.py`, `api/app.py`, `api/settings.py`, `api/deps.py` | точечно: модели; `FileRegistry` по умолчанию и `lifespan`; `builds_dir`; `QueryInspector` в `Services` |
| `bench/artifacts.py`, `bench/run_scenario.py` | каталог с `run_id`, суффикс без него, атомарная запись |
| `datasets/sources.py` | `offline` |
| `stores/qdrant_store_async.py` | `payloads()` |
| `README.md` | реестр, `RAGGLET_BUILDS_DIR`, новые endpoints |

`pyproject.toml` не меняется: pandas, pyarrow и pydantic уже в зависимостях.

## Тесты (критерий приёмки)

Без Qdrant, Redis и torch, на `tmp_path`. Фикстура `make_run_dir(path, *, repeats,
formats, complete, manifest, generation)` в `tests/registry_support.py` собирает
синтетический каталог: per-query на трёх запросах `datasets/demo` (parquet и/или CSV со
списками в repr и NaN у незавершённых стадий), один таймаут, `summary.json`,
`summary.aggregate.json/csv`, manifest с `built_at` из `index_meta()`. Поколение
`"feb"` — только CSV, `recall@k`, без `outcome`, `final_doc_ids` и manifest.

1. **Запись и чтение.** `create_run` создаёт каталог и `run.json`. `update` его
   перезаписывает, временных файлов не остаётся. Новый `FileRegistry` на том же каталоге
   читает те же записи. Три постановки в одну секунду дают `…Z`, `…Z-2`, `…Z-3`, и
   занятый CLI каталог тоже пропускается. Битый `run.json` не перезаписывается, каталог
   импортируется. Сборки — то же в `builds_dir`.
2. **Импорт.** Полный каталог без `run.json` — `completed`, `origin: import`, `dataset`
   и `collection` из snapshot и manifest, `finished_at` = mtime aggregate. Без aggregate
   — `failed` / `RunIncomplete`; после появления aggregate следующий список даёт
   `completed`. Snapshot, не проходящий текущую схему, импортируется. Каталог без
   snapshot и имя на `.` не видны. Удалённый каталог исчезает из списка. В
   импортированные каталоги ничего не пишется.
3. **Рестарт.** Записи `queued` и `running`, неизвестные менеджеру, становятся
   `failed` / `ServerRestart`. `running` с aggregate на диске — `completed`. Задачи,
   которые менеджер знает, не трогаются. Проверяется через `with TestClient(app)`.
4. **Список.** Колонки при `k=10` и `k=5`; `null` у поколения `"feb"` и при отсутствии
   `k`. Фильтры `state`, `scenario`, `scenario_id`, `dataset` (с ревизией и без),
   `collection`, `tag` (тег сценария), период, `q`. Сортировка по `ndcg` по убыванию с
   `null` в конце; `total` и страницы. `GET /v1/builds` с фильтрами.
   `last_run.ndcg_at_10` в списке сценариев.
5. **Деталь прогона.** `include=summary,manifest` наполнены; у упавшего `summary: null`.
   NaN в aggregate отдаётся как `null`. `scenario_file`: `unchanged` → после `PUT`
   `changed` → после `DELETE` `deleted`; у импорта `null`.
6. **Per-query.** Parquet и CSV дают одинаковые строки (списки, NaN → `null`, `qid`
   `0042` — строка). `outcome=timeout`, `miss_at`, `min_total_ms`, `q`, сортировка по
   `first_relevant_rank` и `-lat_total_ms`, страницы, `repeats`. `outcome` у поколения
   `"feb"` — 422 `request.filter`. Нет per-query — 404 `run.no_per_query`, нет повтора
   — 404 со списком.
7. **Деталь запроса** на `datasets/demo`: оценки из qrels и заголовки из корпуса,
   `doc_rank` против `rank` при двух чанках одного документа, `rerank_score` по
   `chunk_id`, `in_context`. Без Qdrant (фейк не отвечает) — `texts.code:
   index.unreachable`, ответ 200. Фейк с другим `built_at` — `index.rebuilt`. С тем же
   `built_at` и фейковыми payload — тексты на месте. Изменённый qrels во временной копии
   датасета — `dataset.changed`. Без manifest — `run.no_manifest`.
8. **Артефакты.** Листинг без временных файлов; файл отдаётся байт в байт; `..%2Fx` и
   неизвестное имя — 404; в zip все файлы под `<id>/`.
9. **Ядро.** `run_id` на каталоге только с `run.json` работает, на каталоге со
   snapshot — `FileExistsError`. Два CLI-прогона в одну секунду получают разные каталоги.
   Артефакты пишутся без временных файлов в итоге.
10. Существующие тесты, включая тесты API 28 на `InMemoryRegistry`, проходят без правок.

Ручная проверка: `ragglet serve` на настоящем `results/runs` (55 каталогов трёх
поколений), список с колонками, per-query и деталь запроса nfcorpus при живом Qdrant,
zip.

## Фазы (отдельными коммитами)

1. **Ядро**: каталог с `run_id` и суффикс без него, атомарная запись артефактов,
   `resolve_dataset(offline)`, `payloads()`.
2. **Реестр на файлах**: `run.json`, записи сборок, скан и импорт, `recover` в
   lifespan, `GET /v1/runs` и `GET /v1/builds` с колонками и фильтрами,
   `include=summary,manifest`, `origin`, `scenario_file`, `RunBrief.ndcg_at_10`,
   `FileRegistry` по умолчанию, `builds_dir`.
3. **Per-query**: список с фильтрами и рангом, деталь запроса с датасетом и Qdrant.
4. **Артефакты**: листинг, файл, zip; README.

## Вне скоупа

- **Удаление прогонов.** Действие необратимое, а для прогона в очереди или идущего
  непонятно, что удалять. Диск не давит: 55 прогонов занимают 42 МБ. Каталог можно
  удалить руками, и запись исчезнет из списка при следующем скане. В v2 вместе с
  удалением коллекций, если понадобится.
- Сравнение прогонов и Парето — 32 и 33. Задел: `compare_runs` работает по `run_dir`
  двух записей, классы различий уже есть в ядре.
- Очередь, процессы, отмена, лог — 29.
- Правка тегов и ссылки на сценарий после постановки.
- Импорт сборок CLI и сборок из метаданных коллекций; связь «прогон → сборка индекса»
  по `built_at`.
- Пересчёт и переименование метрик старых поколений, версия формата артефактов:
  читатель смотрит на наличие колонок.
- Скачивание датасета ради детали запроса, загрузка датасетов, удаление коллекций.
- Индекс заголовков на диске для корпусов в миллионы документов. Сейчас словарь
  `doc_id → title` в памяти, на корпусах BEIR размера nfcorpus и scifact этого
  достаточно.
- Перенос прогонов между стендами через API (загрузка zip). Перенос каталога руками
  работает и так.

## Рассмотренные альтернативы

- **SQLite или SQLite-индекс рядом с `run.json`** — см. таблицу в п. 1: второй источник
  правды без выигрыша на тысячах записей.
- **Записи вне каталога прогона** (`results/registry/<id>.json`). Ядро не пришлось бы
  трогать, но при переносе каталога запись теряется (рамка 34), а записи и каталоги
  придётся сверять.
- **Писать `run.json` и в импортированные каталоги.** Тогда следующий старт будет быстрее,
  но реестр начнёт менять каталоги, которые писал не он, а запись идущего CLI-прогона
  застынет в состоянии на момент импорта.
- **Колонки сводки в `run.json`.** Это кэш, который придётся мигрировать при каждом
  изменении набора колонок, и у CLI-прогонов его всё равно не будет.
- **Новое состояние `incomplete` для импортированных.** Пришлось бы менять
  `TaskState`, который принадлежит менеджеру (29), и учить UI шестому состоянию. Сейчас
  то же сообщает `failed` с `RunIncomplete`.
- **После рестарта ставить `queued` обратно в очередь.** Серия на ночь пережила бы
  рестарт, но за это время мог смениться код сервера, и прогон, поставленный как
  проверенный, пошёл бы непроверенным. Кнопка «Повторить» делает то же явно. Если это
  станет нужно, задачу может взять 29: всё необходимое есть в записи.
- **Тексты чанков из корпуса** (заново нарезать документ по `manifest.index.chunking`).
  Работало бы и после пересборки, но код нарезки мог измениться между сборкой и сегодня,
  и тогда текст тихо отличался бы от того, что видел прогон. «Текст недоступен» честнее.
- **Угадывать списки в CSV по первой `[`.** Текст запроса тоже может начинаться со
  скобки.

## Открытые вопросы

1. **Связывать ли импортированные прогоны со сценарием библиотеки по имени?** Сейчас у
   них `scenario_id: null`, поэтому в «последний прогон» списка сценариев CLI-прогоны не
   попадают (8 прогонов `nfcorpus_hf` там не видны). Можно ставить
   `scenario_id = name`, если такой файл есть в библиотеке. Но CLI мог запускать и
   `configs/experiments/…` с тем же `name`, тогда ссылка будет ложной. Предлагаю
   оставить `null`: историю сценария даёт фильтр `scenario=`, а после 37 все прогоны
   идут через API со ссылкой.
2. **Время конца импортированных прогонов по mtime.** Это даёт длительность в списке,
   но после копирования без `-p` она неверна. Альтернатива: `duration_s = null` у всех
   импортированных.
3. **`builds_dir` по умолчанию `results/builds`.** Подходит ли, или сборки держать в
   `results/runs/.builds`, чтобы для переноса хватало одного каталога?
