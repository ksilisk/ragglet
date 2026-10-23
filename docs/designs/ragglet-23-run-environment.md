# ragglet-23 — Метаданные экспериментального окружения

**Коротко.** Каждый прогон записывает в `manifest.json`, **где и чем** он выполнен:
commit ядра, Python, машину, устройство эмбеддингов (cpu/mps/cuda), версии и состояние
сервисов (Qdrant, Redis). Туда же попадают ревизия rerank-модели и окружение, в котором
собран индекс. Функция `compare_runs` сравнивает два каталога прогонов и помечает каждое
различие: данные, настройка RAG или окружение.

## Актуальность

После ragglet-17 каталог прогона содержит `scenario.snapshot.*` (конфигурация RAG) и
`manifest.json` (идентичность данных и сборки индекса). По ним видно, **что** измеряли,
но не видно, **где и чем**. Если два прогона одного сценария дают разные цифры, по
артефактам нельзя отличить эффект настройки RAG от эффекта машины, зависимостей или
состояния сервисов.

Чего не хватает (по коду и `docs/article-experiments-audit.md` §4):

| # | Ситуация | Почему не видно | Последствие |
|---|---|---|---|
| 1 | Прогоны с разных коммитов | commit не пишется | неизвестно, каким кодом (и каким `uv.lock`) получены цифры |
| 2 | Тот же сценарий на Mac и на Linux-сервере | sentence-transformers сам выбирает устройство (cuda → mps → cpu) | embed latency отличается в разы, причина не записана |
| 3 | `bench run` сразу после `index build` | Qdrant ещё строит HNSW (status `yellow`), поиск идёт частично перебором | latency и recall нерепрезентативны |
| 4 | Маленькая коллекция ниже `indexing_threshold` | `indexed_vectors_count = 0`: поиск точный, HNSW не используется | recall «лучше», чем будет на HNSW; никак не отмечено |
| 5 | Rerank-модель с HF `main` обновилась | ревизия rerank-модели не разрешается и не пишется | s2 меняется без изменения конфига |
| 6 | Индекс пересобран с другим `hnsw` | `m`/`ef_construct` есть только в конфиге коллекции Qdrant | старый прогон больше нельзя интерпретировать |
| 7 | Обновили Qdrant или Redis | версии сервисов не пишутся | сдвиг latency без видимой причины |

Ещё одна мелочь: snapshot и manifest пишутся **до** загрузки моделей. Если модель не
загрузилась, остаётся каталог с идентичностью, но без результатов.

## Решение

### 0. Три класса полей

Каждое поле snapshot и manifest относится к одному классу. Класс задаёт, как трактовать
различие двух прогонов:

| Класс | Где | Различие означает |
|---|---|---|
| **data** | `manifest.dataset` | quality-метрики несравнимы (правило ragglet-17) |
| **config** | `scenario.snapshot`, `manifest.index` (кроме `build_env`), `manifest.rerank` | настройка RAG — то, что исследуем |
| **env** | `manifest.env`, `manifest.index.build_env` | конфаундер: часть разницы может быть эффектом окружения |

Класс определяется **секцией**, а не списком полей. Новое поле попадает в одну из
секций и автоматически участвует в сравнении (п. 4). Отдельных правил на каждое поле нет.

### 1. `manifest.json`: что добавляется

```jsonc
{
  "dataset": { … },                        // без изменений (ragglet-17)
  "index": {
    "profile_id": "demo", "built_at": "…", "embedding_revision": "…", "chunking": { … },
    "params": {"vector_size": 384, "distance": "Cosine", "hnsw": {"m": 16, "ef_construct": 128}},
    "build_env": { … }                     // env процесса index build, та же схема без services
  },
  "rerank": {"revision": "ce0834f2…"},     // null, если rerank выключен
  "env": {
    "commit": "c5e29d5…",
    "python": "3.12.7",
    "system": {"platform": "macOS-15.1-arm64-arm-64bit", "cpu": "Apple M2 Pro",
               "cpus": 12, "ram_bytes": 34359738368},
    "compute": {"device": "mps", "accelerator": null, "torch_threads": 12},
    "services": {                          // роль → имя → описание от самого сервиса
      "storage": {"default": {"type": "qdrant", "version": "1.17.0", "status": "green",
                              "points": 53, "indexed_vectors": 0, "segments": 2}},
      "cache": {"embedding": {"type": "redis", "version": "7.4.1"}}  // только при cache.embedding
    }
  }
}
```

| Поле | Откуда | Что различает |
|---|---|---|
| `env.commit` | `git rev-parse HEAD` из каталога пакета; `null` вне git-checkout | версия ядра и, через `uv.lock`, зависимостей (сбой 1) |
| `env.python` | `platform.python_version()` | интерпретатор: в `uv.lock` есть только `requires-python` |
| `env.system` | `platform.platform()`, модель CPU, доступные процессу CPU, RAM | машина (сбой 2) |
| `env.compute` | загруженный embedder: устройство, имя GPU для cuda, `torch.get_num_threads()` | где считаются эмбеддинги (сбой 2) |
| `env.services.storage.<target>` | `describe()` store: `client.info()` и `get_collection` | версия сервера и состояние коллекции (сбои 3, 4, 7) |
| `env.services.cache.embedding` | `describe()` Redis-кэша: `INFO server` | версия кэша (сбой 7) |
| `index.params` | конфиг коллекции Qdrant (фактический, не профиль) | HNSW и метрика расстояния (сбой 6) |
| `index.build_env` | метаданные коллекции, пишутся в конце `index build` | каким кодом и на чём собран индекс |
| `rerank.revision` | механизм ревизий ragglet-17 §3, как у embedding | версия rerank-модели (сбой 5) |

Решения по составу:

- **Только commit.** Незакоммиченные правки не учитываем: вне кода бывают разные
  артефакты, а commit достаточен. Зависимости заморожены `uv.lock`, он в том же
  коммите, поэтому список пакетов и свой хэш lock не пишем.
- **Hostname не пишем.** Результаты коммитятся в репозиторий и идут в статью, а машину и
  так определяют `system`.
- **Пин ревизий моделей в конфиге.** `rerank.revision` в сценарии работает по правилу
  ragglet-17: ревизию копируют из manifest в конфиг. `CrossEncoder` принимает `revision`,
  как и `SentenceTransformer`. Остальные модели:
  - embedding уже пинится `embedding.revision` в index profile, а bench грузит ревизию
    из индекса;
  - у generation (openai-compatible API) ревизий нет, пином служит имя модели.
- **`build_env`** пишется в метаданные коллекции одним вызовом вместе с остальными, в
  `REQUIRED_FIELDS`. Это cutoff: старые индексы отклоняются с подсказкой пересобрать.

Расширяемость устроена так:

- факты процесса собирает реестр `ENV_FACTS` (имя → функция) в `core/environment.py`;
  новый факт — это одна функция и тест;
- сервисы описывают себя сами. У store и cache есть `async describe() -> dict`; общие
  поля — `type` и `version`, остальные зависят от типа. `run_scenario` собирает
  участников прогона по ролям (`storage`, `cache`) и вызывает `describe()`. Новая БД
  или сервис — это `describe()` в его классе и одна строка регистрации роли; схема
  manifest и `compare_runs` при этом не меняются.

### 2. Порядок в `run_scenario`

```
preflight → backends и модели → capture_env → run dir + snapshot + manifest → запросы
```

Каталог прогона появляется, только когда всё загружено. Manifest по-прежнему пишется до
запросов, поэтому у упавшего прогона идентичность сохраняется.

### 3. Предупреждения

`index.not_green` — новая запись в `INDEX_CHECKS`, уровень warning, прогон не
блокирует. Срабатывает, когда статус коллекции не `green`: оптимизатор работает
(`yellow`) или ждёт запуска (`grey`). Для неё `IndexContext` получает состояние
коллекции целиком, а не только метаданные. `bench run` и `bench validate` печатают
предупреждение тем же рендером, что и предупреждения датасета.

### 4. Сравнение прогонов: `compare_runs`

```python
@dataclass(frozen=True)
class Difference:
    kind: Literal["data", "config", "env"]
    path: str        # "env.compute.device", "scenario.rerank.enabled"
    a: Any
    b: Any

def compare_runs(a: Path, b: Path) -> list[Difference]: ...
```

- Глубокий diff snapshot и manifest двух каталогов. `kind` определяется по секции
  (таблица п. 0). `name`, `description`, `tags` и `run.artifacts_dir` игнорируются.
- **Правило (README):** разница метрик относится к настройке RAG, если нет различий
  `data` и `env`. Есть `data` — quality несравнимы. Есть `env` — часть разницы может
  дать окружение. Различия `system`, `compute` и `services` делают latency несравнимой.
- **CLI-команды нет** (ragglet-37). Потребители — реестр E3 (фильтр «то же окружение»),
  сравнение F3 и API. Функция — единственная реализация правила, чтобы они его не
  переизобретали. Она же читает каждое поле manifest, поэтому неиспользуемых полей нет.

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `core/environment.py` (новый) | `ENV_FACTS`, `capture_env(compute)`, `describe_services()` |
| `modules/embedders/st_embedder.py` | устройство и потоки загруженной модели |
| `rerank/cross_encoder.py`, `config/scenario.py` | `rerank.revision`: разрешение и пин |
| `stores/qdrant_store_async.py` | `describe()`; `collection_info()` вместо `collection_metadata()` |
| `cache/embedding_redis.py` | `describe()` |
| `indexing/build_index.py`, `indexing/metadata.py` | `build_env` в метаданных |
| `bench/preflight.py` | `CollectionInfo` в контексте, `index.not_green`, `index.params`/`build_env` в manifest |
| `bench/run_scenario.py`, `bench/artifacts.py` | порядок, секция `env`, предупреждения в результате |
| `bench/compare.py` (новый) | `compare_runs` |
| `cli/bench.py` | рендер предупреждений |
| `README.md`, `docs/config-reference.yaml` | состав manifest, правило сравнения, `rerank.revision` |

## Тесты (критерий приёмки)

Unit, без Qdrant и torch. Фейковые store и embedder, как в `test_preflight`.

1. commit: во временном git-репозитории это `HEAD`, вне репозитория — `null`.
2. `ENV_FACTS`: ключи и типы; `compute` берётся из фейкового embedder.
3. Сервисы: `describe()` каждого target под `storage`, `cache.embedding` — только при
   включённом Redis-кэше.
4. Сборка пишет `build_env`. Метаданные без него дают `index.no_metadata`.
5. Preflight:
   - статус `yellow` даёт предупреждение `index.not_green`, прогон идёт;
   - `index.params` берутся из конфига коллекции.
6. `run_scenario`:
   - manifest содержит `dataset`, `index`, `rerank` и `env`;
   - `rerank.revision` из сценария доходит до `CrossEncoder`;
   - ошибка загрузки модели не создаёт run dir.
7. `compare_runs`:
   - одинаковые каталоги дают `[]`;
   - различие каждого класса определяется верно;
   - игнорируемые поля не попадают в результат.
8. `tests/test_configs.py` проходит.

E2E на этой ноде нет: нет Qdrant-сервера и torch. Near-E2E с Qdrant в local-режиме
запускаю как в ragglet-17, но у local-клиента нет `info()`, поэтому версия будет фейковой.
Прошу smoke на Mac: `index build` и `bench run` на demo, затем проверить `env` в
manifest (ожидается `device: mps`).

## Фазы (отдельными коммитами в этот же PR)

1. `core/environment.py`, секция `env` без services, новый порядок `run_scenario`.
2. Сервисы, `index.params`, `index.not_green`, `build_env` (cutoff), `rerank.revision`.
3. `compare_runs`, README и config-reference.

## Вне скоупа

- `run_id`, статус и завершённость прогона, реестр — E3 (ragglet-30). Заметка для E3:
  каталоги, созданные в одну секунду, сейчас совпадают (`exist_ok=True`).
- Мониторинг ресурсов во время прогона (загрузка CPU/GPU, память) — это измерение, а
  не метаданные.
- Версия LLM в generation: у openai-compatible API нет запроса версии, модель уже есть в
  snapshot.
- `capture-manifest` в analyzer статьи становится избыточным. Analyzer не трогаем, как
  решено в ragglet-17.
- Commit в Docker-образе без `.git` решает G1 (ragglet-34). До этого там будет
  `commit = null`.

## Решено на ревью

- Учитываем только commit: без diff незакоммиченных правок и без предупреждения о них.
- Зависимости фиксирует `uv.lock` из того же коммита, своей машинерии вокруг пакетов нет.
- `services` расширяемые: сервис описывает себя сам через `describe()` и регистрируется
  под ролью.
- Ревизия rerank-модели пинится в сценарии.
- `build_env` делаем сейчас, с пересборкой индексов.
- `index.not_green` — предупреждение.
