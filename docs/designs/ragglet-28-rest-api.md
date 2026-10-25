# ragglet-28 — REST API сценариев и экспериментов

**Коротко.** FastAPI-сервер (`ragglet serve`) — единственный вход в стенд по рамке
ragglet-38. Он отдаёт библиотеку сценариев и профилей индексов (YAML-файлы в `configs/`),
схему конфигов для формы UI, состояние стенда, коллекции Qdrant и датасеты. Прогон и
сборку он синхронно проверяет (dry-run) и ставит в очередь (202 и id). Логики бенчмарка
в API нет: проверки и выполнение — функции ядра `run_scenario` и `build_index`, а API
переводит HTTP в их вызовы, а их результаты и ошибки — в JSON. Если UI нужна проверка,
которой у ядра нет, она добавляется в preflight ядра, а не в API, и CLI получает её тоже.
Ошибки во всех ответах — один формат `{title, hint, issues[{code, severity, path, where,
message}]}`, где `path` — путь поля конфига. Выполнение задач (29) и хранение прогонов
(30) здесь представлены интерфейсами `JobManager` и `Registry` (`typing.Protocol`) с
in-memory заглушками. После этой задачи 29 и 30 идут параллельно, каждая заменяет свою
заглушку, не трогая endpoints.

## Пример, на котором всё объясняется

История из ragglet-15: исследователь в UI делает `nfcorpus_rerank` из `nfcorpus_hf`,
проверяет и запускает. Те же запросы шлют CLI (ragglet-37) и curl.

1. **Открыть.** `GET /v1/scenarios/nfcorpus_hf` →
   `{"id": "nfcorpus_hf", "yaml": "# Пример бенчмарка…\nversion: 1\nname: \"nfcorpus_hf\"\n…",
   "updated_at": "2026-10-24T17:58:02Z", "etag": "9f2c41d07a1be355", "valid": true, "problem": null}`.
   `yaml` — текст файла как есть, с комментариями.
2. **Дублировать.** UI меняет в тексте `name` и шлёт `POST /v1/scenarios {"yaml": "…"}` →
   201, появляется файл `configs/scenarios/nfcorpus_rerank.yaml`. Отдельной ручки
   «дублировать» нет: копия — это создание из YAML.
3. **Сохранить с ошибкой.** Включён реранк с `output_top_n: 10`, а в `metrics.quality.k`
   осталось 30. `PUT /v1/scenarios/nfcorpus_rerank {"yaml": "…", "etag": "51e0…"}` → 200:

   ```json
   {"id": "nfcorpus_rerank", "valid": false, "etag": "c07d…", "problem": {
     "title": "Scenario config is invalid", "hint": null, "issues": [
       {"code": "config.value_error", "severity": "error", "path": "metrics.quality.k", "where": null,
        "message": "metrics.quality.k=30 exceeds final list depth 10 (rerank.output_top_n)"}]}}
   ```

   Файл сохранён как черновик: работа не теряется, а запустить его нельзя.
4. **Конфликт.** Тем временем файл поправили в IDE. Следующий `PUT` с прежним `etag` →
   409 `library.conflict`, UI предлагает перезагрузить.
5. **Проверить.** После правки UI шлёт только что сохранённый текст:
   `POST /v1/runs {"scenario": "…", "scenario_id": "nfcorpus_rerank", "dry_run": true}`
   → 200: датасет `hf:BeIR/nfcorpus@b5026a0e…`, 323 запроса × 1 повтор, concurrency 16,
   цели storage со статусом и `points` и одно предупреждение `index.not_green` с
   `path: null` и `where: "target default (nfcorpus_v1)"`. Это `run_scenario(cfg,
   dry_run=True)`, вызванный в отдельном потоке. `scenario_id` — только ссылка на
   библиотеку для реестра, файл сервер не читает.
6. **Запустить.** `POST /v1/runs {"scenario": "…", "scenario_id": "nfcorpus_rerank"}` → 202
   `{"id": "nfcorpus_rerank_20261125T101540Z", "state": "queued", "position": 1, "warnings": […]}`
   и заголовок `Location: /v1/runs/nfcorpus_rerank_20261125T101540Z`. Сервер ещё раз
   сделал preflight, реестр выдал id (оно же имя каталога прогона), менеджер задач
   поставил задачу за идущей сборкой `demo_sharded`. Если тот же сценарий поставить ещё
   раз в ту же секунду, id второго прогона — `nfcorpus_rerank_20261125T101540Z-2`.
7. **Следить.** `GET /v1/runs/nfcorpus_rerank_20261125T101540Z` раз в секунду: `queued,
   position 1` → `running, phase preparing` → `running, phase progress, progress {stage:
   queries, done: 120, total: 323}` → `completed`. С `?include=summary` приходит агрегат
   по повторам.
8. **То же из терминала.**
   `jq -n --rawfile s nfcorpus_rerank.yaml '{scenario: $s, dry_run: true}' | curl -X POST
   localhost:8000/v1/runs -H 'content-type: application/json' -d @-` даёт тот же отчёт:
   файл клиента уходит содержимым. CLI из ragglet-37 шлёт тот же запрос через httpx.

Ни на одном шаге API не решал сам, валиден ли конфиг и совместим ли индекс. На шаге 3
это решила схема `ScenarioConfig`, на шаге 5 — preflight ядра.

## Решение

### 1. Принципы контракта

- Всё под `/v1`. OpenAPI лежит в `/v1/openapi.json`, Swagger UI — в `/v1/docs`: корень
  `/` остаётся за UI.
- Конфиг передаётся только содержимым: YAML-текстом в поле `scenario` / `profile`, как
  требует рамка 38. Рядом может стоять `scenario_id` / `profile_id`, но это ссылка на
  сценарий библиотеки для реестра, а не способ передать конфиг: файл по нему сервер не
  читает.
- **API не проверяет конфиг сам.** Ошибки дают только схема (pydantic) и ядро
  (`ValidationError`). К результатам ядра API добавляет данные (состояние коллекций, кто
  их использует), но не новые ошибки. Если нужна новая проверка, она идёт в preflight
  ядра (п. 10).
- Долгое (прогон, сборка) отвечает 202 и id. Проверка (dry-run) синхронна: 200 или 422.
- Любой ответ не 2xx — `Problem` (п. 3), включая 404, 409 и 500.
- Времена — UTC ISO 8601 (`2026-11-25T10:15:40Z`).
- Сервер — один процесс uvicorn с одним worker: заглушки, блокировки библиотеки и
  потом менеджер задач живут в памяти этого процесса.

### 2. Endpoints

| Метод и путь | Запрос → ответ | Реализует |
|---|---|---|
| `GET /v1/meta/scenario` | → `ScenarioMeta` | 28 |
| `GET /v1/meta/index-profile` | → `IndexProfileMeta` | 28 |
| `GET /v1/stand` | → `StandStatus` | 28 |
| `GET /v1/scenarios` | → `list[ScenarioItem]` | 28 (`last_run` — из реестра) |
| `POST /v1/scenarios` | `ScenarioCreate` → 201 `ConfigDocument` | 28 |
| `GET /v1/scenarios/{id}` | → `ConfigDocument` | 28 |
| `PUT /v1/scenarios/{id}` | `ConfigWrite` → `ConfigDocument` (201, если файл создан) | 28 |
| `DELETE /v1/scenarios/{id}` | `?etag=` → 204 | 28 |
| `/v1/index-profiles`, `/v1/index-profiles/{id}` | то же для профилей, строки списка — `IndexProfileItem` | 28 |
| `GET /v1/indexes` | → `IndexList` | 28 |
| `GET /v1/datasets` | → `DatasetList` | 28 |
| `POST /v1/runs` | `RunRequest`: с `dry_run` → 200 `RunCheck`, иначе 202 `RunAccepted` | 28 поверх `JobManager` и `Registry` |
| `GET /v1/runs/{id}` | `?include=summary,manifest,scenario` → `RunStatus` | 28 поверх заглушек; `summary` и `manifest` наполняет 30 |
| `POST /v1/runs/{id}/cancel` | → `RunStatus` | 28 поверх `JobManager` |
| `GET /v1/runs/{id}/log` | `?offset=` → `LogChunkOut` | 28 поверх `JobManager` |
| `POST /v1/builds` | `BuildRequest`: 200 `BuildCheck` или 202 `BuildAccepted` | 28 поверх заглушек |
| `GET /v1/builds/{id}`, `POST …/cancel`, `GET …/log` | как у прогонов | 28 поверх заглушек |
| `GET /v1/tasks` | → `list[TaskItem]`: текущая задача и очередь по порядку | 28 поверх `JobManager` |
| `GET /v1/runs`, `GET /v1/builds` (списки), `/v1/runs/{id}/queries…`, `/v1/runs/{id}/artifacts…` | | 30; здесь только методы `Registry` |

Схемы ниже — pydantic-модели из `api/schemas.py`, они же схемы FastAPI и OpenAPI.
Генерировать ли по ним TS-клиент для UI, решается в 31.

### 3. Формат проблем

```python
class ProblemIssue(BaseModel):
    code: str                              # "config.value_error", "index.not_green", "library.conflict"
    severity: Literal["error", "warning"]
    path: str | None = None                # поле конфига: "retrieval.sources[1].params"
    where: str | None = None               # место вне конфига: "qrels/test.tsv:7", "target default (nfcorpus_v1)"
    message: str

class Problem(BaseModel):
    title: str
    hint: str | None = None
    issues: list[ProblemIssue]
```

Тело любого ответа не 2xx — `Problem`, предупреждения в 200-ответах —
`list[ProblemIssue]`. Источники сводятся к нему так:

| Источник | `code` | `path` | `where` | `title` | HTTP |
|---|---|---|---|---|---|
| `pydantic.ValidationError` конфига | `config.<type>`: `config.missing`, `config.extra_forbidden`, `config.value_error`, … | из `loc`, см. ниже | null | `Scenario config is invalid` / `Index profile config is invalid` | 422 |
| `yaml.YAMLError` | `config.yaml` | null | `line 3, column 2` (из `problem_mark`, считая с единицы) | то же | 422 |
| корень YAML не mapping, пустой текст | `config.yaml` | null | null | то же | 422 |
| `ragglet.core.issues.ValidationError` | `Issue.code` | `Issue.path` (новое поле, п. 10) | `Issue.where` | `title` и `hint` исключения | 422 |
| ошибка тела запроса (`RequestValidationError` FastAPI) | `request.<type>` | `loc` без `body` | null | `Request is invalid` | 422 |
| `scenario_id` / `profile_id` не равен `name` / `id` в содержимом | `request.library_ref` | `scenario_id` / `profile_id` | null | `Request is invalid` | 422 |
| нет сценария, прогона, сборки | `library.not_found`, `run.not_found`, `build.not_found` | null | `scenarios/x` | | 404 |
| конфликт версий, отмена завершённой задачи | `library.conflict`, `task.finished` | null | | | 409 |
| любое другое исключение | `server.error` | null | null | `Internal error`; в `message` — `Type: message`, трейсбек — в лог сервера | 500 |

**`loc` → `path`.** Строковые части соединяются точкой, индексы списков идут в скобках:
`('retrieval', 'sources', 1, 'params')` → `retrieval.sources[1].params`. В `loc` pydantic
стоят алиасы, поэтому получается `retrieval.async`, как в YAML, а не `async_enabled`.
`message` — это `msg` без префикса `Value error, `.

**Model-валидаторы.** Проверено на pydantic 2.12.5 из `uv.lock`: у model-валидатора
корня `loc` пустой, у вложенной модели — путь этой модели (`('rerank',)`), а в
`ctx["error"]` лежит сам брошенный экземпляр исключения. Поэтому путь объявляет
валидатор, а не API по тексту сообщения. В `config/errors.py` появляется

```python
class FieldError(ValueError):
    """Ошибка model-валидатора, относящаяся к полю; path — от модели валидатора."""

    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path
```

и валидаторы бросают её вместо `ValueError`, а API склеивает `loc` с `FieldError.path`.
Сопоставление:

| Валидатор | `loc` | `FieldError.path` | Итоговый `path` |
|---|---|---|---|
| `ScenarioConfig._generation_requires_context` | `()` | `generation.enabled` | `generation.enabled` |
| `ScenarioConfig._quality_k_within_final_depth` | `()` | `metrics.quality.k` | `metrics.quality.k` |
| `RerankConfig._model_required_if_enabled` | `rerank` | `model` | `rerank.model` |
| `GenerationConfig._model_required_if_enabled` | `generation` | `model` | `generation.model` |
| `CacheConfig._at_least_one_layer_if_enabled` | `cache` | `enabled` | `cache.enabled` |
| `RetrievalConfig._single_keyword_not_supported` | `retrieval` | `sources` | `retrieval.sources` |
| `StorageConfig._validate_storage_shape` (сценарий) | `storage` | `backends` | `storage.backends` |
| `DatasetConfig._known_source` → `check_source` | `dataset` | `source` или `revision` | `dataset.source` / `dataset.revision` |
| `LoadConfig._validate_params` | `run.load` | `type` для неизвестной модели | `run.load.type`; ошибки параметров pydantic уже даёт по полям: `run.load.concurrency` |
| `ChunkingConfig._validate_sizes` (профиль) | `chunking` | `chunk_size` / `chunk_overlap` | `chunking.chunk_overlap` и т.п. |
| `StorageConfig._validate_storage_shape` (профиль) | `storage` | `endpoint` / `backends` | `storage.endpoint` / `storage.backends` |

`FieldError` — подкласс `ValueError`, поэтому `load_scenario` и тексты ошибок в CLI не
меняются. Если валидатор бросит обычный `ValueError`, `path` будет путём модели
(`rerank`) или null у корня, и UI покажет ошибку в секции или в общей панели. Выглядит
хуже, но ошибка не теряется. Тест обходит все model-валидаторы обеих схем и требует
`FieldError`.

### 4. Библиотека сценариев и профилей

Как решено в ragglet-15, это файлы `<id>.yaml` в `settings.scenarios_dir`
(`configs/scenarios`) и `settings.index_profiles_dir` (`configs/indexes`). id — имя
файла без расширения, оно же `name` сценария или `id` профиля. Оба каталога обслуживает
одна реализация `ConfigLibrary(dir, model, id_field)`. `configs/experiments/**` в
библиотеку не входит.

```python
ConfigId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

class ConfigDocument(BaseModel):
    id: str
    yaml: str                              # текст файла как есть, с комментариями
    updated_at: datetime                   # mtime файла
    etag: str                              # sha256 текста, первые 16 hex
    valid: bool
    problem: Problem | None                # почему невалиден (черновик)

class ConfigWrite(BaseModel):              # PUT /v1/scenarios/{id}
    yaml: str
    etag: str | None                       # версия, с которой начата правка; None — файла ещё нет

class ScenarioCreate(BaseModel):           # POST /v1/scenarios: ровно одно из yaml / from_run
    yaml: str | None = None                # id берётся из name
    from_run: str | None = None            # id прогона: берётся его snapshot
    name: ConfigId | None = None           # обязательно вместе с from_run

class CollectionRef(BaseModel):
    endpoint: str                          # "http://localhost:6333"
    collection: str

class ScenarioItem(BaseModel):
    id: str
    updated_at: datetime
    etag: str
    valid: bool
    description: str | None
    tags: list[str]
    dataset: str | None                    # "hf:BeIR/nfcorpus"; None у черновика без dataset
    collections: list[CollectionRef]       # цели storage
    pipeline: Pipeline | None              # чипы: strategy, sources, rerank, context, generation, cache, concurrency, repeats
    last_run: RunBrief | None              # Registry.list_runs(scenario_id=id, limit=1)

class IndexProfileItem(BaseModel):
    id: str
    updated_at: datetime
    etag: str
    valid: bool
    dataset: str | None
    chunking: str | None                   # "recursive 800/120"
    embedding_model: str | None
    targets: list[CollectionRef]
    last_build: BuildBrief | None
```

`Pipeline` — плоская модель из полей в комментарии; у черновика она `None`. `RunBrief` и
`BuildBrief` — `id`, `state`, `enqueued_at`. Колонки сводки (nDCG@10 и т.п.) добавляет 30
вместе со списком прогонов.

Правила:

- **Версия файла — `etag`, а не `updated_at`.** `PUT` и `DELETE` принимают `etag` версии,
  с которой начата правка. Если файл на диске уже другой — 409 `library.conflict`, в
  сообщении текущие `updated_at` и `etag`. `PUT` с `etag: null` создаёт файл и даёт 409,
  если файл уже есть. `DELETE` без `etag` удаляет безусловно. `etag` считается по
  содержимому, потому что mtime хуже: на части ФС две записи в одну секунду дают
  одинаковый mtime, а `git checkout` и `touch` меняют его без правки. `updated_at`
  остаётся для показа и для пометки «сценарий изменился после прогона».
- **Черновики разрешены на уровне схемы, но не синтаксиса.** Сохраняется любой текст,
  который разбирается как YAML-mapping, если строковое `name` (у профиля `id`) в нём
  равно id в пути. Иначе 422 `library.name_mismatch` с `path: "name"`. Ошибки схемы
  записи не мешают: ответ 200 с `valid: false` и `problem`. Синтаксически битый YAML —
  422 `config.yaml`, файл не трогается. Почему так. Файл в библиотеке и так бывает
  невалидным, его правят в IDE, поэтому все читатели — список, dry-run, CLI — уже обязаны
  с этим справляться. Запрет сохранять такого инварианта не даёт, а работу теряет: новый
  пустой сценарий невалиден, пока не заполнены `name`, `dataset` и `storage`. Битый YAML
  же не прочтут ни форма, ни CLI, и по нему нельзя проверить, что `name` совпадает с id.
  Запустить черновик нельзя: `POST /v1/runs` отклонит его той же 422.
- **Создание.** Из YAML: `POST {yaml}` → 201, 409 если id занят. Из snapshot прогона:
  `POST {from_run, name}`. Сервер берёт `scenario` из записи реестра, ставит `name`,
  убирает `run.artifacts_dir` и пишет `yaml.safe_dump(sort_keys=False)`: snapshot и так
  машинный дамп без комментариев. Дублирование и пустой сценарий делает клиент: UI
  меняет `name` в тексте (комментарии сохраняет его YAML-редактор) или собирает текст из
  `meta.defaults` и шлёт `POST {yaml}`. Переименования нет: это создание под новым
  именем и удаление старого.
- **Список** читает все `*.yaml` каталога при каждом запросе, их десятки. Невалидный файл
  — строка с `valid: false`. Файл, у которого `name` не совпадает с именем файла
  (переименовали в IDE), тоже `valid: false`, с `library.name_mismatch`.
- **Запись атомарная**: временный файл в том же каталоге и `os.replace`. Проверка `etag`
  и запись идут под `threading.Lock` библиотеки. Обработчики библиотеки синхронные,
  FastAPI выполняет их в пуле потоков.
- id в пути проверяется шаблоном `ConfigId`, поэтому `../x` даёт 422 до обращения к диску.

### 5. Схема и умолчания: `GET /v1/meta/…`

```python
class DatasetSourceMeta(BaseModel):
    id: str                                # local | hf
    supports_revision: bool

class ScenarioMeta(BaseModel):
    json_schema: dict[str, Any]            # ScenarioConfig.model_json_schema()
    defaults: dict[str, Any]               # умолчания всех полей, включая default_factory
    load_models: dict[str, dict[str, Any]] # run.load.type -> JSON Schema параметров модели
    dataset_sources: list[DatasetSourceMeta]
    storage_backends: list[str]            # реализованные: ["qdrant"]
    retrieval_strategies: list[str]        # реализованные: ["single", "fanout_merge"]
    metrics: dict[Literal["doc", "chunk"], list[str]]
    latency_steps: list[str]               # стадии, для которых ядро пишет lat_*_ms

class IndexProfileMeta(BaseModel):
    json_schema: dict[str, Any]
    defaults: dict[str, Any]
    dataset_sources: list[DatasetSourceMeta]
    storage_backends: list[str]
```

Чего нет в `model_json_schema()` и откуда это берётся (проверено на текущей схеме):

| Чего нет | Почему | Поле meta | Источник |
|---|---|---|---|
| `run.load.concurrency` | `LoadConfig` с `extra="allow"`, в схеме только `type` | `load_models` | `{t: m.Params.model_json_schema() for t, m in LOAD_MODELS.items()}` |
| значения `run.load.type` | объявлен как `str` | ключи `load_models` | `LOAD_MODELS` |
| значения `dataset.source` | объявлен как `str` | `dataset_sources` | `SOURCES` и их `supports_revision` |
| что реализовано из `storage.backend.type` | `Literal` шире реализации | `storage_backends` | `IMPLEMENTED_BACKENDS` |
| что реализовано из `retrieval.strategy` | `cascade` есть в `Literal`, а в рантайме `NotImplementedError` | `retrieval_strategies` | новая `IMPLEMENTED_STRATEGIES`, её же проверяет preflight (п. 10) |
| метрики по уровням | `doc` и `chunk` — `list[str]` | `metrics` | `METRICS`; для `chunk` — без `needs_ideal`, через новую `metrics_for_level()`, которой пользуется и валидатор |
| допустимые `latency_steps` | `list[str]`, неизвестные тихо пропускаются | `latency_steps` | новая `LATENCY_STEPS = STAGES + ("total",)` из `query_loop`; `merge` есть в `summarize`, но ядро его не пишет |
| умолчания `default_factory` | pydantic не вызывает фабрики при генерации схемы: у `metrics.quality.k` в схеме нет `default` | `defaults` | рекурсивный обход полей: `default` или вызов фабрики; обязательные поля пропускаются |

Правило одно: список «что реализовано» живёт в ядре одной константой, и её читают и
проверка, и meta. Своих списков у API нет.

### 6. Стенд: `GET /v1/stand`

```python
class ServiceStatus(BaseModel):
    url: str
    available: bool
    version: str | None
    error: str | None                      # "ConnectError: [Errno 61] Connection refused"
    collections: int | None = None         # только у Qdrant

class StandStatus(BaseModel):
    qdrant: list[ServiceStatus]            # по settings.qdrant_endpoints
    redis: ServiceStatus | None            # settings.redis_url; None — не задан
    compute: dict[str, Any]                # {"device": "mps", "accelerator": null, "torch_threads": 12}
    system: dict[str, Any]                 # system_info(): platform, cpu, cpus, ram_bytes
    commit: str | None
    python: str
    env: dict[str, bool]                   # переменная из generation.api_key_env -> задана ли
    current: TaskItem | None               # из JobManager.queue()
    queued: int
```

- Сервисы опрашиваются параллельно, каждый с таймаутом `settings.probe_timeout_s` (2 с).
  Недоступный сервис — это `available: false` и текст ошибки, а ответ всё равно 200:
  стенд отвечает, а не падает. У Qdrant берутся `client.info()` и число коллекций, у
  Redis — `INFO server`, как в `describe()`.
- `commit`, `python`, `system` — те же функции `ENV_FACTS`, что пишут `manifest.env`.
- `compute` показывает, где будет считать модель, не загружая её:
  `sentence_transformers.util.get_device_name()` (та же логика cuda → mps → cpu, что при
  загрузке) и `torch.get_num_threads()`. Это импорт torch в процессе сервера, около 1–2 с,
  поэтому значение считается один раз, лениво и в потоке. Фактическое устройство прогона
  по-прежнему пишет `manifest.env.compute`. Функция — `predicted_compute()` в
  `core/environment.py`.
- `env` — имена из `generation.api_key_env` всех валидных сценариев библиотеки плюс
  `OPENAI_API_KEY`. Значение — только `bool`, сами значения не отдаются никогда.
  Проверяется окружение процесса сервера, потому что задачи 29 наследуют его.

### 7. Коллекции и датасеты

`GET /v1/indexes` собирает коллекции со всех `settings.qdrant_endpoints`, параллельно и с
тем же таймаутом:

```python
class IndexList(BaseModel):
    endpoints: list[ServiceStatus]
    collections: list[CollectionItem]

class CollectionItem(BaseModel):
    endpoint: str
    name: str
    state: Literal["ready", "optimizing", "incomplete", "foreign"]
    status: str                            # green / yellow / grey / red
    points: int | None
    indexed_vectors: int | None
    segments: int
    params: dict[str, Any] | None          # vector_size, distance, hnsw; None — не один безымянный вектор
    ragglet: dict[str, Any] | None         # метаданные ragglet целиком; None — их нет или они неполные
    used_by: list[str]                     # сценарии библиотеки, у которых коллекция среди целей storage
    profiles: list[str]                    # профили библиотеки, которые её собирают
```

- Данные берутся из `collection_info()` store, как в preflight. В store добавляется
  `list_collections()`, а `collection_info()` для коллекции, у которой не один
  безымянный вектор, отдаёт `params=None`, а не бросает `ValueError`: иначе одна чужая
  коллекция роняла бы весь список. Для preflight это ничего не меняет: у такой
  коллекции нет метаданных ragglet, и она и так получает `index.no_metadata`.
- `state` считает API по правилам ragglet-15. `ready` — метаданные полны
  (`extract_index_metadata` не `None`) и статус `green`. `optimizing` — полны, но не
  `green`. `incomplete` — ключ `ragglet` есть, но неполон, или метаданных нет совсем.
  `foreign` — ключа `ragglet` нет, а другие метаданные есть, или `params is None`.
- `used_by` и `profiles` — сопоставление по паре `(endpoint, collection)` с целями
  `build_storage_layout()` сценариев и профилей библиотеки. Endpoint сравнивается после
  нормализации: схема и хост в нижнем регистре, без `/` в конце. `localhost` и
  `127.0.0.1` считаются разными, DNS не разрешаем.
- Недоступный endpoint попадает в `endpoints` с `available: false`, коллекции остальных
  отдаются как обычно.

`GET /v1/datasets` — минимум для выбора в форме:

```python
class DatasetList(BaseModel):
    local: list[LocalDataset]              # каталоги с corpus.jsonl(.gz) под settings.dataset_roots
    hf: list[HfDataset]

class LocalDataset(BaseModel):
    id: str                                # как писать в dataset.id: "datasets/demo"
    splits: list[str]                      # по qrels/*.tsv

class HfDataset(BaseModel):
    id: str                                # "BeIR/nfcorpus"
    seen_in: list[str]                     # "scenario:nfcorpus_hf", "profile:nfcorpus_hf", "collection:http://localhost:6333/nfcorpus_v1"
```

Локальные датасеты ищутся обходом корней на глубину 2, без разбора файлов. `id` — путь
относительно рабочего каталога сервера, потому что `LocalSource` разрешает `dataset.id`
именно так. В `hf` попадают id из сценариев и профилей библиотеки с `source: hf` и из
метаданных коллекций с `dataset_source == "hf"`. Ревизии, размеры и проверка — не здесь:
это делает dry-run.

### 8. Dry-run прогона и сборки

`POST /v1/runs {…, "dry_run": true}` и `POST /v1/builds {…, "dry_run": true}` синхронны:
200 с отчётом или 422.

```python
class RunRequest(BaseModel):
    scenario: str                          # YAML-текст сценария: конфиг только содержимым
    scenario_id: ConfigId | None = None    # ссылка для реестра: из какого сценария библиотеки прогон
    dry_run: bool = False
    tags: list[str] = []                   # метки прогона в реестре, сверх scenario.tags

class RunPlan(BaseModel):                  # модель ответа для dataclass RunPlan из run_plan() ядра
    repeats: int
    queries_per_repeat: int
    total: int                             # то же total, что у progress("queries", …)
    load: dict[str, Any]                   # {"type": "closed", "concurrency": 16}

class RunCheck(BaseModel):
    scenario: str                          # name
    dataset: DatasetCheck                  # label, revision, split, queries, corpus_sha256, eval_sha256
    index: dict[str, Any]                  # метаданные ragglet primary target
    params: dict[str, Any]                 # параметры коллекции primary target
    targets: list[TargetCheck]             # по каждой цели: id, endpoint, collection, status, points, indexed_vectors
    plan: RunPlan
    warnings: list[ProblemIssue]

class BuildRequest(BaseModel):
    profile: str                           # YAML-текст профиля
    profile_id: ConfigId | None = None     # ссылка для реестра: из какого профиля библиотеки сборка
    dry_run: bool = False

class BuildCheck(BaseModel):
    profile: str                           # id профиля
    dataset: str                           # label
    corpus_sha256: str
    documents: int
    targets: list[BuildTargetCheck]
    warnings: list[ProblemIssue]

class BuildTargetCheck(BaseModel):
    id: str
    endpoint: str
    collection: str
    reachable: bool
    error: str | None
    exists: bool | None                    # None — endpoint недоступен
    status: str | None
    points: int | None
    built_at: str | None                   # из метаданных ragglet
    index_profile_id: str | None
    used_by: list[str]                     # сценарии библиотеки
```

- **`scenario_id` / `profile_id`** не заменяют содержимое: сервер по ним файл не читает,
  а только кладёт ссылку в запись реестра (страница прогона ведёт на сценарий,
  `last_run` в списке библиотеки ищется по ней). По правилу библиотеки id равен `name`
  сценария или `id` профиля, поэтому несовпадение — 422 `request.library_ref` с
  `path: "scenario_id"`. Существование файла не проверяется: сценарий могут удалить и
  позже, а ссылка в реестре останется. UI перед запуском и так сохраняет сценарий, текст
  у него на руках; «Запустить выбранные» сначала читает тексты через
  `GET /v1/scenarios/{id}`.
- **`RunCheck`** — это `Preflight` ядра плюс план. План считает ядро: новая
  `run_plan(cfg, n_queries)` в `bench/run_scenario.py`, ею же `run_scenario` считает
  `total` для прогресса. Для `targets` `Preflight` должен отдавать состояние всех целей,
  а не только `params` primary, — это поле `collections` (п. 10).
- **`BuildCheck`** — это `BuildResult` плюс то, чего ядро не знает по каждой цели: по
  рамке 38 dry-run сборки к Qdrant не подключается. API вызывает `collection_info()` по
  целям параллельно и с таймаутом и сопоставляет их с библиотекой, как в
  `GET /v1/indexes`. Это данные, а не проверки: недоступный endpoint даёт
  `reachable: false`, а не 422. Ошибкой он станет при старте сборки.
- **Event loop.** `build_index` синхронный и блокирующий. `run_scenario` асинхронный, но
  его preflight синхронно читает и хэширует датасет, а для `hf` при первом обращении ещё
  и скачивает его. В event loop сервера это остановило бы все запросы, включая опрос
  `GET /v1/tasks`. Поэтому оба вызова уходят в поток:

  ```python
  checks = anyio.CapacityLimiter(1)
  await anyio.to_thread.run_sync(partial(build_index, cfg, dry_run=True), limiter=checks)
  await anyio.to_thread.run_sync(anyio.run, partial(run_scenario, cfg, dry_run=True), limiter=checks)
  ```

  У прогона в потоке свой event loop, а клиенты Qdrant создаются внутри `run_scenario`,
  то есть в этом же loop. Лимит 1: проверки идут по одной, и две первые проверки одного
  `hf`-датасета не качают его одновременно. Обёртки лежат в `api/core_calls.py`, больше
  нигде API ядро не вызывает.
- **`run.artifacts_dir`** клиента сервер до preflight заменяет на
  `settings.artifacts_dir`: это серверный путь, и назначает его сервер (ragglet-15).
- **Известное ограничение: dry-run во время идущего прогона делит с ним CPU.** В v1
  ничего с этим не делаем. Dry-run лёгкий: чтение и хэш корпуса, для nfcorpus это доли
  секунды. Тяжёлое в нём только первое скачивание датасета с Hub, а к моменту прогона
  датасет уже в кэше после сборки индекса. Если это станет шумом в замерах на больших
  корпусах, dry-run можно будет ставить в очередь менеджера за текущей задачей.

### 9. Постановка, статус, отмена, лог

**Как `POST /v1/runs` вызывает реестр и менеджер:**

```python
async def post_run(req: RunRequest, response: Response, ...) -> RunCheck | RunAccepted:
    cfg = parse_scenario(req.scenario)                      # 422: config.yaml, config.*
    check_library_ref(req.scenario_id, cfg.name)            # 422: request.library_ref
    cfg = with_artifacts_dir(cfg, settings.artifacts_dir)
    checked = await check_run(cfg)                          # preflight №1 в потоке; 422: ValidationError
    if req.dry_run:
        response.status_code = 200
        return RunCheck.build(cfg, checked)
    record = await registry.create_run(NewRun.build(cfg, req, checked))  # id и run_dir
    status = await jobs.submit(TaskSpec(
        ref=TaskRef(TaskKind.RUN, record.id),
        name=cfg.name,
        config=cfg.model_dump(by_alias=True),
        run_dir=record.run_dir,
    ))
    response.headers["Location"] = f"/v1/runs/{record.id}"
    return RunAccepted(id=record.id, state=status.state, position=status.position,
                       warnings=issues_out(checked.warnings))
```

`POST /v1/builds` устроен так же, с `registry.create_build` и `TaskKind.BUILD`. Если
проверка не прошла, в реестр и очередь ничего не попадает. Второй preflight делает
воркер 29 при старте: `run_scenario` начинает с него, и `ValidationError` там превращает
задачу в `failed` с тем же списком проблем.

**id.** У прогона это `<name>_<UTC постановки>`, например
`nfcorpus_rerank_20261125T101540Z`, и это же имя каталога прогона (ragglet-15). У
сборки — `<id профиля>_<UTC постановки>`. id выдаёт `Registry.create_*`, и он
гарантирует уникальность: если id занят, к нему добавляется суффикс `-2`, `-3` и так
далее до свободного: `nfcorpus_rerank_20261125T101540Z-2`. Время в id всегда настоящее,
сдвигать его нельзя. Занят id бывает, когда тот же сценарий поставили дважды за секунду,
например двойным кликом по «Повторить». Решение ragglet-15 «два прогона в одну секунду
невозможны» верно для старта, но id выдаётся при постановке, а ставить можно сколько
угодно быстро. Полный формат: `<name>_<YYYYMMDDTHHMMSSZ>[-N]`.

У сущности один id. Задача менеджера адресуется `TaskRef(kind, id)` того же прогона или
сборки, своих id у задач нет. Пространства id прогонов и сборок разные (`nfcorpus_hf`
есть и сценарием, и профилем), поэтому в `TaskRef` есть `kind`.

Каталог прогона получает имя по id, а не по времени создания: в ядре появляется
`run_scenario(…, run_id=None)` (п. 10). Время в имени каталога теперь означает
постановку, а не старт, время старта хранит реестр.

**Интерфейс менеджера**, `jobs/protocol.py`. Типы — frozen dataclass, как результаты
ядра; `api/schemas.py` переводит их в модели ответов.

```python
class TaskKind(StrEnum):
    RUN = "run"
    BUILD = "build"

class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class TaskRef:
    kind: TaskKind
    id: str                                # id прогона или сборки из реестра

@dataclass(frozen=True)
class TaskSpec:
    ref: TaskRef
    name: str                              # сценарий или профиль: для очереди и шапки
    config: dict[str, Any]                 # проверенный конфиг; воркер валидирует его заново
    run_dir: Path | None = None            # у прогона: каталог, который создаст ядро

@dataclass(frozen=True)
class ProgressInfo:
    stage: str                             # queries | corpus
    done: int
    total: int
    first_at: datetime                     # первое событие: от него UI считает «осталось ≈»
    updated_at: datetime

@dataclass(frozen=True)
class TaskFailure:
    type: str                              # класс исключения
    message: str
    qid: str | None = None                 # из __notes__ ядра
    title: str | None = None               # у ValidationError второго preflight
    hint: str | None = None
    issues: tuple[Issue, ...] = ()

@dataclass(frozen=True)
class TaskStatus:
    ref: TaskRef
    name: str
    state: TaskState
    position: int | None                   # в очереди: 1 — следующая; иначе None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: ProgressInfo | None
    failure: TaskFailure | None

@dataclass(frozen=True)
class LogChunk:
    text: str
    next_offset: int
    complete: bool                         # задача завершена и лог дочитан

class JobManager(Protocol):
    async def submit(self, spec: TaskSpec) -> TaskStatus: ...
    async def status(self, ref: TaskRef) -> TaskStatus | None: ...   # None — менеджер не знает задачу
    async def cancel(self, ref: TaskRef) -> TaskStatus: ...          # queued — убрать; running — прервать
    async def log(self, ref: TaskRef, offset: int = 0, limit: int = 65536) -> LogChunk: ...
    async def queue(self) -> list[TaskStatus]: ...                   # текущая, затем ожидающие по порядку
```

**Интерфейс реестра**, `registry/protocol.py`:

```python
@dataclass(frozen=True)
class NewRun:
    name: str
    scenario_id: str | None                # ссылка на сценарий библиотеки из запроса
    scenario: dict[str, Any]               # cfg.model_dump(): ровно то, что ляжет в scenario.snapshot
    etag: str                              # sha256 текста сценария: для «изменился после прогона»
    tags: list[str]
    dataset: str                           # label
    collection: str                        # primary target
    plan: RunPlan
    warnings: list[Issue]                  # preflight при постановке

@dataclass(frozen=True)
class RunRecord:
    id: str
    run_dir: Path
    name: str
    scenario_id: str | None
    scenario: dict[str, Any]
    etag: str
    tags: list[str]
    dataset: str
    collection: str
    plan: RunPlan
    warnings: list[Issue]
    state: TaskState
    enqueued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: TaskFailure | None = None
    final_progress: ProgressInfo | None = None   # «отменён, выполнено 420 / 969»

@dataclass(frozen=True)
class TaskChange:                          # что меняет менеджер задач при переходах
    state: TaskState | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: TaskFailure | None = None
    final_progress: ProgressInfo | None = None
    result: dict[str, Any] | None = None   # у сборки: documents и записанные метаданные

class Registry(Protocol):
    async def create_run(self, new: NewRun) -> RunRecord: ...        # резервирует уникальный id
    async def create_build(self, new: NewBuild) -> BuildRecord: ...
    async def update(self, ref: TaskRef, change: TaskChange) -> None: ...
    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def get_build(self, build_id: str) -> BuildRecord | None: ...
    async def list_runs(self, query: RunQuery) -> Page[RunRecord]: ...
    async def list_builds(self, query: BuildQuery) -> Page[BuildRecord]: ...
    async def summary(self, run_id: str) -> RunSummary | None: ...   # summary.json и aggregate
    async def manifest(self, run_id: str) -> dict[str, Any] | None: ...
    async def artifacts(self, run_id: str) -> list[Artifact]: ...    # name, size
    async def queries(self, run_id: str, query: QueryFilter) -> Page[dict[str, Any]]: ...
```

`NewBuild` и `BuildRecord` устроены так же, только с `profile_id`, `profile`, `targets`
и `result` вместо плана. `Page` — это `items` и `total`. `RunQuery` (фильтры
`scenario_id`, `name`, `state`, `tag`, период, сортировка, страница), `QueryFilter`, `RunSummary` и строка списка с
колонками сводки здесь заданы минимально. 30 расширяет их полями, не меняя смысла
существующих.

**Кто что пишет.** Реестр хранит долговечное: запись, состояние, времена, ошибку, итог.
Менеджер — живое: позицию в очереди, прогресс, лог. При каждом переходе состояния
менеджер (29) сам вызывает `Registry.update`, а прогресс в реестр не пишет: он меняется
чаще раза в секунду. `GET /v1/runs/{id}` склеивает одно с другим: берёт запись реестра и,
если `jobs.status()` знает задачу, накладывает поверх её состояние, позицию и прогресс.

**Заглушки.** `InMemoryRegistry` держит записи в словарях и выдаёт id по правилу выше;
`summary` и `manifest` у него `None`, `artifacts` и `queries` пустые.
`InMemoryJobManager` ничего не выполняет: `submit` ставит задачу в очередь, а переходы
делают тестовые методы `start(ref)`, `report(ref, stage, done, total)` и
`finish(ref, state, failure=None)`. Они вызывают `Registry.update` так же, как будет
вызывать 29. Значит, между 28 и 29 `ragglet serve` принимает прогоны, но они так и
остаются `queued`, и в UI это видно. Так решено на ревью: CLI до ragglet-37 вызывает
ядро напрямую, а заглушка нужна тестам API и фронту (31), чтобы строить страницы прогона
и очереди до появления 29. `create_app(settings, jobs=None, registry=None)` подставляет
заглушки; 29 и 30 передают свои реализации, а endpoints и их тесты не меняются.

**Ответы:**

```python
class RunAccepted(BaseModel):              # 202, Location: /v1/runs/{id}
    id: str
    state: TaskState                       # queued
    position: int | None
    warnings: list[ProblemIssue]

class ProgressOut(BaseModel):
    stage: str
    done: int
    total: int
    first_at: datetime
    updated_at: datetime

class TaskErrorOut(BaseModel):
    type: str
    message: str
    qid: str | None
    problem: Problem | None                # второй preflight при старте не прошёл

class RunStatus(BaseModel):
    id: str
    scenario: str                          # name
    scenario_id: str | None                # ссылка на сценарий библиотеки, если её передали
    etag: str
    state: TaskState
    phase: Literal["preparing", "progress", "finishing"] | None   # только при running
    position: int | None
    progress: ProgressOut | None           # у завершённой — final_progress
    plan: RunPlan
    dataset: str
    collection: str
    tags: list[str]
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: TaskErrorOut | None
    warnings: list[ProblemIssue]           # preflight при постановке
    summary: RunSummary | None = None      # ?include=summary (30)
    manifest: dict[str, Any] | None = None # ?include=manifest (30)
    scenario_yaml: str | None = None       # ?include=scenario: snapshot для «Повторить» и «Открыть как сценарий»

class TaskItem(BaseModel):                 # GET /v1/tasks и StandStatus.current
    kind: TaskKind
    id: str
    name: str
    state: TaskState
    position: int | None
    progress: ProgressOut | None
    enqueued_at: datetime
    started_at: datetime | None

class LogChunkOut(BaseModel):
    text: str
    offset: int
    next_offset: int
    complete: bool
```

`BuildAccepted` и `BuildStatus` такие же, с `profile`, `profile_id`,
`targets: list[CollectionRef]` и `result` (число документов, записанные метаданные)
вместо плана и summary.

- Без `include` ответ лёгкий: только реестр и менеджер, артефакты не читаются. Именно его
  опрашивают раз в секунду.
- `phase` считает API из состояния и прогресса по таблице ragglet-15: `running` без
  прогресса — `preparing`, `done < total` — `progress`, `done == total` — `finishing`.
  Так CLI и UI показывают одно и то же.
- «Повторить» — это `include=scenario` и
  `POST /v1/runs {"scenario": scenario_yaml, "scenario_id": scenario_id}`: новый прогон
  из того же snapshot, переданного содержимым, с той же ссылкой на библиотеку.
- `cancel`: задача в очереди убирается и сразу становится `cancelled`; у `running`
  менеджер прерывает процесс, и ответ показывает текущее состояние (`running`, пока
  процесс не завершился). Отмена завершённой задачи — 409 `task.finished`.
- `log` — JSON со смещением: следующий запрос идёт с `?offset=` из `next_offset`
  прошлого ответа, не больше 64 КиБ за раз.
- id, которого не знают ни реестр, ни менеджер, — 404.

### 10. Доработки ядра

Всё ниже действует и в CLI: `ragglet bench run --dry-run` получает те же проверки с
кодом выхода 2. Рамка 38 разрешает переводить такие ошибки в `ValidationError` в 28.

1. **`Issue.path`.** `Issue(code, severity, where, message, path=None)`, у `error()` и
   `warning()` — `path` ключевым аргументом. Проверки датасета и индекса не меняются
   (`path` null). Проверки конфига ниже ставят `path`, чтобы UI показал ошибку у поля, а
   `where` у них — тот же путь или цель storage, его печатает CLI.
2. **`FieldError`** в валидаторах схем (п. 3).
3. **Проверки сценария** — список `SCENARIO_CHECKS` по образцу `INDEX_CHECKS`: только по
   конфигу и до создания клиентов. Новая `check_scenario(cfg) -> StorageLayout`
   объединяет `build_storage_layout`, `require_implemented` и эти проверки в одну
   `ValidationError`; `run_scenario` вызывает её первой строкой.

   | Код | Что сейчас | `path` |
   |---|---|---|
   | `retrieval.not_implemented` | `strategy: cascade` даёт `NotImplementedError` на первом запросе, уже после создания каталога | `retrieval.strategy` |
   | `retrieval.keyword_sources` | больше одного `keyword` — `NotImplementedError` в `BackendRegistry.build` | `retrieval.sources[2]`, у каждого лишнего |
   | `retrieval.backend_ref` | неизвестный `params.backend_ref` — `KeyError` на первом запросе | `retrieval.sources[1].params.backend_ref`; в сообщении доступные id целей |
   | `storage.layout` | повтор id целей или меньше двух `backends` при `sharded`/`replicated` — `ValueError` из `build_storage_layout` | `storage.backends` |

   `StorageTarget` получает поле `path` (`storage.backend`, `storage.backends[1]`), чтобы
   ошибки по целям ссылались на поле.
4. **Датасет не найден.** `LocalSource` без каталога бросает `FileNotFoundError`, то есть
   сбой задачи вместо ошибки конфига. Через API путь серверный, и опечатка в нём будет
   самой частой ошибкой. Теперь это `ValidationError` `dataset.not_found` с
   `path: dataset.id` и путём в `where`. Для `hf` так же обрабатываются
   `RepositoryNotFoundError` и `RevisionNotFoundError` (`dataset.id` /
   `dataset.revision`). Сетевые ошибки Hub остаются сбоем: это проблема стенда, а не
   конфига. Действует и в `build_index`.
5. **Недоступные сервисы в preflight.** Preflight и так обращается к Qdrant, но ошибка
   соединения сейчас выходит трейсбеком клиента. Теперь это `storage.unreachable` по
   каждой цели (`path: storage.backends[i].endpoint`). Если включён `cache.embedding`,
   preflight делает `PING` Redis с таймаутом и при неудаче даёт `cache.unreachable` с
   `path: cache.embedding.endpoint`; сейчас Redis трогает только `describe_services`,
   уже после загрузки моделей. Недоступный endpoint в конфиге чаще всего — опечатка в
   порту, поэтому это ошибка конфига, а не сбой. Обе проверки идут до побочных эффектов:
   каталог прогона ещё не создан, модели не загружены. В API это 422 на «Проверить», в
   CLI — код 2.
6. **`Preflight.collections`** — `CollectionInfo` всех целей, для `RunCheck.targets`.
   `manifest()` не меняется.
7. **`run_scenario(…, run_id=None)`** задаёт имя каталога прогона; `None` — как сейчас,
   `<name>_<UTC сейчас>`. С `run_id` каталог создаётся с `exist_ok=False`, и совпадение id
   даёт ошибку, а не смешивает артефакты двух прогонов (заметка из ragglet-23). Форма
   операции из рамки 38 сохраняется: это ещё один ключевой аргумент, CLI его не передаёт.
8. **Мелкое:** `run_plan()` (п. 8); `IMPLEMENTED_STRATEGIES`, `LATENCY_STEPS`,
   `metrics_for_level()` (п. 5); `list_collections()` и `params=None` в store (п. 7);
   `ping()` Redis; `predicted_compute()` (п. 6).

Не делаем проверку доступа к Hub для `rerank.model` из списка ragglet-15: без сетевого
запроса к Hub на каждый dry-run её не сделать, а модель и так скачивается при старте
прогона.

### 11. Конфиг сервера и `ragglet serve`

```python
class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGGLET_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    qdrant_endpoints: Annotated[list[str], NoDecode] = ["http://localhost:6333"]
    redis_url: str | None = "redis://localhost:6379/0"
    scenarios_dir: Path = Path("configs/scenarios")
    index_profiles_dir: Path = Path("configs/indexes")
    artifacts_dir: Path = Path("results/runs")
    dataset_roots: Annotated[list[Path], NoDecode] = [Path("datasets")]
    ui_dir: Path | None = None             # собранный UI (31); None — UI не раздаётся
    probe_timeout_s: float = 2.0
```

- Настройки читаются из переменных окружения `RAGGLET_*` и файла `.env` в рабочем
  каталоге: `pydantic-settings` 2.13 уже в зависимостях, `python-dotenv` приходит с ним.
  Списки задаются через запятую:
  `RAGGLET_QDRANT_ENDPOINTS=http://localhost:6333,http://localhost:6334` (`NoDecode` и
  валидатор, а не JSON). Приоритет: флаги `ragglet serve`, потом окружение, потом `.env`,
  потом умолчания.
- Почему не YAML-файл, как у сценариев: в Docker (ragglet-34) естественны переменные и
  `env_file:` в compose, а локально умолчания уже работают из корня репозитория.
  `ragglet serve` без настроек видит `configs/`, `datasets/` и `localhost:6333`.
  Отдельный файл стал бы третьим источником.
- `.env` читается только для `RAGGLET_*` и в `os.environ` не попадает. Ключ генерации
  (`OPENAI_API_KEY`) задаётся окружением процесса сервера — в shell или через `env_file:`
  compose, а `GET /v1/stand` показывает, задан ли он. `.env` добавляется в `.gitignore`.
- Относительные пути считаются от рабочего каталога сервера. При старте `ragglet serve`
  печатает итоговые настройки с абсолютными путями.
- **По умолчанию сервер слушает `127.0.0.1`**: авторизации нет, а API пишет файлы и
  ставит задачи на стенд. В контейнере `RAGGLET_HOST=0.0.0.0` задаётся явно.

```
ragglet serve [--host 127.0.0.1] [--port 8000]
```

Команда живёт в `cli/app.py`:
`uvicorn.run(create_app(ServerSettings(...)), host=…, port=…, workers=1)`. Остальные
команды CLI в этой задаче не меняются, клиентами API они станут в ragglet-37.

### 12. Пакеты, зависимости, статика

```
src/ragglet/api/
  app.py          create_app(settings, jobs=None, registry=None) -> FastAPI; обработчики ошибок; статика
  settings.py     ServerSettings
  problems.py     Problem, ProblemIssue; from_pydantic, from_yaml, from_validation_error
  schemas.py      модели запросов и ответов
  library.py      ConfigLibrary: чтение, запись, etag, черновики
  probes.py       опрос Qdrant и Redis с таймаутом, список коллекций
  core_calls.py   check_run, check_build: ядро в потоке
  deps.py         Depends: settings, библиотеки, jobs, registry
  routers/        meta, stand, scenarios, index_profiles, indexes, datasets, runs, builds, tasks
src/ragglet/jobs/      protocol.py, memory.py (InMemoryJobManager)
src/ragglet/registry/  protocol.py, memory.py (InMemoryRegistry)
```

- `routers/scenarios.py` и `routers/index_profiles.py` — тонкие обёртки над одной
  `ConfigLibrary`, `runs.py` и `builds.py` — над `core_calls`, `Registry` и `JobManager`.
- Зависимости: `fastapi>=0.115` и `uvicorn>=0.32` в основных, без `uvicorn[standard]`:
  uvloop и httptools одному пользователю не нужны. `TestClient` работает на `httpx`, он
  уже есть.
- **Статика UI.** Если задан `settings.ui_dir`, он монтируется в `/` после
  API-роутеров. Существующий файл отдаётся как есть, а любой другой путь вне `/v1`
  получает `index.html`, чтобы работали прямые ссылки вроде `/runs/:id?tab=queries`. Под
  `/v1` фолбэка нет: неизвестный путь — 404 `Problem`. Без `ui_dir` `GET /` отвечает
  `{"api": "/v1", "docs": "/v1/docs"}`. CORS не нужен: UI приходит с того же origin, а в
  разработке работает прокси dev-сервера (ragglet-15).

## Расхождения с таблицей ragglet-15

| В ragglet-15 | Здесь | Почему |
|---|---|---|
| `path` только у ошибок схемы, у проблем ядра только `where` | `path` есть и у части проблем ядра: у новых проверок конфига в preflight | `cascade`, `backend_ref`, endpoint-ы — это поля конфига; без `path` UI не поставит ошибку у поля |
| 409 по `updated_at`, «в `If-Match`-стиле» | 409 по `etag` (хэш содержимого) из тела `PUT` и параметра `DELETE`; `updated_at` отдаётся для показа | mtime грубее и меняется без правки; поле тела видно в схеме OpenAPI и проще для CLI |
| четыре способа создать сценарий | на сервере два: из YAML и из snapshot прогона; «дублировать» и «пустой» делает UI через `POST {yaml}` | дублирование — это правка `name` в тексте с комментариями, её уже умеет YAML-редактор UI; серверу понадобилась бы round-trip YAML-библиотека |
| `POST /v1/runs {scenario, scenario_id?, tags?}` | так же, и `scenario_id` обязан совпадать с `name` в содержимом (иначе 422 `request.library_ref`) | по правилу библиотеки id равен `name`; ссылка, не совпадающая с содержимым, ввела бы реестр в заблуждение |
| id сборки назначает 29 | id сборки `<профиль>_<UTC>` выдаёт реестр при постановке, как у прогона | один id на сущность, задача адресуется им же; сборки тоже попадают в реестр (рамка 38) |
| «два прогона в одну секунду невозможны» | в id время постановки, при совпадении — суффикс `-2`, `-3`: `<name>_<UTC>[-N]`; так же у сборок | id выдаётся при 202, а ставить можно быстрее раза в секунду; время в имени каталога теперь означает постановку, а не старт |
| `GET /v1/runs/{id}` с `fields=` или отдельный `/status` | лёгкий ответ по умолчанию, тяжёлое — через `?include=summary,manifest,scenario`; `phase` считает сервер | один URL, частый опрос дёшев без параметров; фаза одинакова в UI и CLI |
| `GET /v1/tasks?active=1` | `GET /v1/tasks` — только активные: текущая и очередь | завершённые задачи — это прогоны и сборки в реестре (30), второй список не нужен |
| `GET /v1/indexes`: поля коллекции | плюс `state`, `profiles` и список endpoint-ов с доступностью | правила четырёх состояний из ragglet-15 один раз считает сервер; `profiles` нужны для «Пересобрать» |
| проверка сборки: «существует ли коллекция, `points`, `built_at`, кто использует» | плюс `reachable` и `error`; недоступный endpoint — данные, а не 422 | dry-run сборки по рамке 38 к Qdrant не подключается, а API не добавляет ошибок сверх ядра |
| meta: схема и пять списков | плюс `defaults` и `retrieval_strategies` | умолчаний `default_factory` в схеме нет (`metrics.quality.k`); `cascade` в схеме есть, а не реализован |
| лог: «текст от смещения» | JSON `{text, offset, next_offset, complete}` | клиенту нужно знать, откуда продолжать и закончен ли лог |
| список прогонов, запросы, артефакты, список сборок | не в 28: endpoints делает 30, здесь только методы `Registry` | хранение — задача 30 |
| dry-run ловит `rerank.model` без доступа к Hub | не делаем | это сетевой запрос к Hub на каждый dry-run, а модель и так скачивается при старте |

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `api/` (новый) | приложение, настройки, схемы, проблемы, библиотека, опрос сервисов, роутеры |
| `jobs/`, `registry/` (новые) | `Protocol` и in-memory заглушки |
| `cli/app.py` | `ragglet serve` |
| `core/issues.py` | `Issue.path` |
| `config/errors.py` (новый), `config/scenario.py`, `config/index_profile.py`, `datasets/sources.py` | `FieldError` в model-валидаторах и `check_source` |
| `bench/preflight.py` | `SCENARIO_CHECKS`, `check_scenario`, `storage.unreachable`, `cache.unreachable`, `Preflight.collections` |
| `bench/run_scenario.py`, `bench/artifacts.py` | `check_scenario` первой строкой, `run_id`, `run_plan()` |
| `storage/layout.py` | `StorageTarget.path`; ошибки формы storage — `ValidationError` `storage.layout` |
| `datasets/sources.py` | `dataset.not_found` |
| `stores/qdrant_store_async.py` | `list_collections()`, `params=None` у чужих коллекций |
| `cache/embedding_redis.py` | `ping(endpoint, timeout)` |
| `retrieval/engine.py`, `bench/query_loop.py`, `bench/metrics.py` | `IMPLEMENTED_STRATEGIES`, `LATENCY_STEPS`, `metrics_for_level()` |
| `core/environment.py` | `predicted_compute()` |
| `pyproject.toml`, `uv.lock` | `fastapi`, `uvicorn` |
| `.gitignore` | `.env` |
| `README.md` | `ragglet serve`, переменные `RAGGLET_*`, новые коды проблем |

## Тесты (критерий приёмки)

Без Qdrant, Redis и torch. `fastapi.testclient.TestClient` на `create_app` с настройками
на `tmp_path` (копия `configs/scenarios`, `configs/indexes`, `datasets/demo`) и с
фейковыми store и embedder, как в `test_preflight.py`.

1. **Проблемы.** `config.extra_forbidden` с `path: bogus`; `retrieval.sources[1].params`;
   `retrieval.async`; `metrics.quality.k` и `generation.enabled` у валидаторов корня;
   `rerank.model` у вложенного; `run.load.type` и `run.load.concurrency`. Битый YAML даёт
   `config.yaml` с `where: "line 3, column 2"`, пустой текст и список в корне — тоже
   `config.yaml`. Ошибка тела запроса — `request.*`. Все model-валидаторы обеих схем
   бросают `FieldError` (обход по классам).
2. **Dry-run прогона** на `datasets/demo`: 200, `queries`, `plan.total = repeats ×
   queries`, предупреждение `index.not_green` при статусе `yellow`; каталог прогона не
   создан, модель не загружена. Ошибки: `k` больше глубины — 422 с
   `path: metrics.quality.k`; битые qrels во временном датасете — 422 с
   `where: "qrels/test.tsv:1"` и `path: null`; нет каталога — `dataset.not_found` с
   `path: dataset.id`; `cascade`, два `keyword`, чужой `backend_ref`, повтор id целей —
   свои коды и пути; store бросает ошибку соединения — `storage.unreachable`; Redis не
   отвечает при `cache.embedding` — `cache.unreachable`. `run.artifacts_dir` из запроса
   заменён серверным.
3. **Dry-run сборки** профиля `demo`: 200, `documents`, `corpus_sha256`; цели обогащены
   (`exists`, `points`, `built_at`, `used_by` по библиотеке); недоступный endpoint даёт
   `reachable: false` при ответе 200; битый датасет — 422 с `where`.
4. **Event loop не блокируется.** Пока идёт dry-run с медленным фейковым датасетом (1 с),
   `GET /v1/tasks` через `httpx.AsyncClient` на ASGI отвечает быстрее 0,2 с.
5. **Библиотека.** Список с валидными сценариями, черновиком и файлом с чужим `name`.
   `GET` отдаёт текст с комментариями байт в байт. `POST {yaml}` — 201, повторно — 409.
   `PUT` после изменения файла на диске — 409, с актуальным `etag` — 200. `PUT` с
   `name` ≠ id — 422 с `path: name`. Невалидный по схеме — 200 `valid: false`, файл
   записан. Битый YAML — 422, файл не изменился. `DELETE` — 204, затем 404. id `..%2Fx` —
   422. После записи в каталоге нет временных файлов. `POST {from_run, name}` по записи
   заглушки реестра. То же для профилей.
6. **Meta.** `load_models.closed` содержит `concurrency`; `dataset_sources` — ключи
   `SOURCES`; `storage_backends == ["qdrant"]`; в `retrieval_strategies` нет `cascade`; в
   `metrics.chunk` нет `ndcg`, `recall`, `r_cap`; в `latency_steps` нет `merge`;
   `defaults.metrics.quality.k == [1, 3, 5, 10]`; `defaults` вместе с `name`, `dataset` и
   `storage` проходят `ScenarioConfig`.
7. **Стенд при недоступных Qdrant и Redis** (фейковые probe бросают ошибку соединения
   или спят дольше таймаута): 200 не дольше таймаута, `available: false`, `error`
   заполнен. `env` отражает `monkeypatch.setenv`, значений переменных в ответе нет.
   `current` и `queued` берутся из заглушки.
8. **Коллекции и датасеты.** Фейковый клиент с четырьмя коллекциями даёт `ready`,
   `optimizing`, `incomplete` и `foreign`; `used_by` и `profiles` сходятся с библиотекой;
   при недоступном endpoint коллекции другого на месте. Датасеты: `datasets/demo` с
   `splits: ["test"]`, `hf`-id из библиотеки и метаданных.
9. **Постановка и жизнь задачи на заглушках.** `POST /v1/runs` → 202, id вида
   `<name>_YYYYMMDDTHHMMSSZ`, заголовок `Location`; в реестре запись с `scenario_id` из
   запроса, в очереди задача с `run_dir = artifacts_dir / id`. Три постановки за одну
   секунду (время заморожено) дают `…Z`, `…Z-2`, `…Z-3`; так же у сборок. Запрос без
   `scenario` — 422 `request.missing`; `scenario_id`, не равный `name`, — 422
   `request.library_ref` с `path: scenario_id`. После 422 нет ни записи, ни задачи. `GET`: `queued` с позицией → после `start` фаза `preparing` →
   после `report` фаза `progress` → после `finish(failed, ValidationError)` есть
   `error.problem` с issues. `cancel` в очереди — `cancelled`, завершённой задачи — 409.
   `log` по смещениям. `GET /v1/tasks` — текущая, затем очередь. `include=scenario` даёт
   snapshot, и `POST` его как `scenario` создаёт новый прогон. То же для сборок.
   Неизвестный id — 404.
10. **Ядро.** Проверки из п. 10 — unit-тесты рядом с `test_preflight.py`. `run_id`
    задаёт имя каталога, повтор — ошибка. `bench run --dry-run` на `cascade` выходит с
    кодом 2.
11. **Настройки и статика.** `RAGGLET_QDRANT_ENDPOINTS=a,b` превращается в список; флаг
    `--port` важнее переменной. С `ui_dir` (временный `index.html`) `/runs/x` отдаёт
    `index.html`, а `/v1/nope` — 404 JSON `Problem`.
12. Существующие тесты проходят.

Ручная проверка на Mac: `ragglet serve`, затем `GET /v1/stand` и `GET /v1/indexes` против
настоящих Qdrant и Redis и dry-run `nfcorpus_hf` через curl.

## Фазы (отдельными коммитами)

1. Ядро: `Issue.path`, `FieldError`, `check_scenario`, `dataset.not_found`, недоступные
   сервисы, `Preflight.collections`, `run_id`, `run_plan`, константы реализованного,
   `list_collections`, `predicted_compute`.
2. Каркас API: настройки, `create_app`, проблемы и обработчики ошибок, meta, stand,
   `ragglet serve`, зависимости.
3. Библиотека сценариев и профилей, коллекции, датасеты.
4. `jobs/` и `registry/` (Protocol и заглушки); прогоны и сборки: dry-run, постановка,
   статус, отмена, лог, `/v1/tasks`.
5. Статика UI, README.

После фазы 4 и до ragglet-29 прогоны и сборки через API ставятся в очередь, но не
выполняются: заглушка менеджера ничего не запускает. Это принято. Выполнять прогоны в
это время можно из CLI, который до ragglet-37 вызывает ядро напрямую, а заглушка нужна
тестам API и фронту (31).

## Вне скоупа

- Очередь, процессы, отмена работающей задачи, лог процесса, второй preflight при старте
  — ragglet-29. Здесь только интерфейс и заглушка без выполнения: до 29 прогоны через API
  остаются `queued`.
- Хранение прогонов и сборок, восстановление после рестарта сервера (записи
  `queued`/`running`, которых уже нет у менеджера), список прогонов с колонками сводки,
  per-query, артефакты и zip, наполнение `summary` и `manifest` — ragglet-30.
- CLI как клиент API — ragglet-37. До него CLI вызывает ядро напрямую, как сейчас, и
  остаётся единственным способом реально выполнить прогон, пока нет 29.
- Изоляция dry-run от идущего прогона по CPU — известное ограничение v1 (п. 8).
- UI — ragglet-31 и 32; здесь только раздача статики.
- Авторизация, несколько пользователей, несколько процессов сервера.
- SSE и WebSocket (решение 8 ragglet-15).
- Проверка Hub для `rerank.model` и проверка неизвестных `metrics.latency_steps` в схеме.
- Удаление коллекций, загрузка датасетов, ограничение `dataset.id` корнями датасетов.
- Сервер в Docker и compose — ragglet-34.

## Рассмотренные альтернативы

- **Путь у model-валидаторов по тексту сообщения**, таблицей «префикс сообщения → path»
  в API. Отвергнуто: ломается от правки текста, а путь знает только сам валидатор.
  `FieldError` — одна строка в каждом валидаторе.
- **Сохранять в библиотеку только валидное.** Инвариант простой, но IDE его и так
  нарушает, а пользователь теряет незаконченную работу или держит её в браузере, где её
  не видит CLI.
- **`If-Match` и 412.** Стандартнее, но ragglet-15 просит 409, а версия в теле видна в
  схеме OpenAPI и проще для CLI.
- **Dry-run в event loop сервера.** Синхронное чтение и хэш датасета и скачивание с Hub
  останавливают все запросы. **Dry-run как задача менеджера (202)** отвергнут ещё в
  ragglet-15: он лёгкий, а ответ нужен сразу.
- **Конфиг сервера в YAML** (`configs/server.yaml`). Это третий источник рядом с
  окружением и `.env`, и в Docker он неудобен.
- **Свои id у задач менеджера.** Два id на одну сущность и таблица соответствия, хотя UI
  думает о прогонах и сборках, а не о задачах (решение 1 ragglet-15).
- **Сценарий в запросе JSON-объектом.** YAML — надмножество JSON, текстовое поле
  принимает оба формата, а путь разбора и ошибки `config.yaml` остаются одни.
- **Заглушка менеджера, которая выполняет задачи в процессе сервера.** Позволила бы
  гонять прогоны до 29, но в обход правила «свежий процесс на задачу» из рамки 38 и с
  риском, что временное станет постоянным.
- **`POST /v1/runs {scenario_id}` без содержимого** — сервер сам читает файл библиотеки.
  Отвергнуто на ревью: рамка 38 требует передавать конфиг содержимым, а UI перед запуском
  и так сохраняет сценарий, и текст у него есть.
- **Сдвиг времени в id на секунду при совпадении.** Отвергнуто на ревью: в id оказалось
  бы ложное время.

## Решено на ревью

1. **Черновики, невалидные по схеме, сохраняются** с `valid: false` и `problem`;
   синтаксически битый YAML не сохраняется (422 `config.yaml`). Сценарий «на полпути»
   должен сохраняться (п. 4).
2. **Недоступный Qdrant или Redis при dry-run — 422**, проблема preflight с кодом
   `storage.unreachable` или `cache.unreachable`, а не сбой задачи. Проверка идёт до
   побочных эффектов, и в UI ошибка встаёт в «Проверить» (п. 10.5).
3. **Совпадение id решается суффиксом `-2`, `-3`** к `<name>_<UTC>`, а не сдвигом
   времени: id не должен содержать ложное время. То же для id сборки (п. 9).
4. **Конфиг в `POST /v1/runs` и `POST /v1/builds` — только содержимым** (`scenario` /
   `profile`), как требует рамка 38. `scenario_id` / `profile_id` — необязательная ссылка
   рядом с содержимым: по ней реестр знает, из какого сценария или профиля библиотеки
   прогон или сборка. Файл по ней сервер не читает (п. 8).
5. **Между 28 и 29 прогоны через API стоят в очереди и не выполняются** — приемлемо. CLI
   до ragglet-37 вызывает ядро напрямую, а заглушка нужна тестам и фронту («Фазы», «Вне
   скоупа»).
6. **Dry-run во время прогона делит с ним CPU — в v1 ничего не делаем**, это известное
   ограничение. Dry-run лёгкий, тяжёлое в нём только первое скачивание датасета с Hub, а
   к моменту прогона датасет уже в кэше после сборки индекса (п. 8).
