# ragglet-37 — CLI как клиент единого API

**Коротко.** По решению 6 рамки ragglet-38 CLI одним шагом перестаёт вызывать ядро и
становится клиентом REST API (ragglet-28) на typer + httpx. Локального режима нет: если
сервер не запущен, CLI так и говорит и выходит с кодом 3. `bench run` и `index build`
остаются теми же командами с тем же выводом. Файл конфига уходит на сервер содержимым,
`--dry-run` печатает прежний отчёт из ответа 200, ответ 422 печатается прежним списком
проблем с кодом 2. После 202 CLI ждёт задачу: раз в секунду опрашивает её статус, рисует
фазы и полосу прогресса, а в конце печатает summary. Ctrl-C отменяет задачу на сервере,
`--no-wait` только ставит её в очередь. Появляются команды просмотра: `runs`, `builds`,
`tasks`, `stand`. Из `cli/` уходят все импорты ядра. Локальными остаются только
`dataset import` (конвертация файлов) и `serve`.

## Пример, на котором всё объясняется

Вечер из ragglet-29: исследователь пересобирает `nfcorpus_hf` (профиль
`configs/indexes/nfcorpus_hf.yaml`) и прогоняет сценарий `nfcorpus_hf` на 323 запросах.

**Сервер не запущен.**

```
$ uv run ragglet bench run configs/scenarios/nfcorpus_hf.yaml
Error: cannot reach the ragglet server at http://127.0.0.1:8000 (connection refused).
Start it with `ragglet serve` (or docker compose), or set --api / RAGGLET_API_URL.
$ echo $?
3
```

Ядро CLI не вызывает. Исследователь запускает `uv run ragglet serve` в соседнем
терминале.

**Проверка.** CLI шлёт `POST /v1/runs {"scenario": "<текст файла>", "scenario_id":
"nfcorpus_hf", "dry_run": true}` и печатает ответ 200 (`RunCheck`) в прежнем формате.
Две последние строки новые: они уже есть в ответе.

```
$ uv run ragglet bench run configs/scenarios/nfcorpus_hf.yaml --dry-run
Preflight: OK
scenario: nfcorpus_hf
dataset: hf:BeIR/nfcorpus@b5026a0e96e8+a451b3b26d3a
split: test, queries: 323
corpus sha256: 0b2e…
eval sha256: 51ab…
index: nfcorpus_hf, built 2026-10-24T17:40:12Z
embedding: all-MiniLM-L6-v2@c9745ed1d9f2…
chunking: recursive 800/120
plan: 323 queries x 1 repeat, load closed (concurrency 1)
target default: nfcorpus_v1 at http://localhost:6333, green, 3633 points
```

**Ошибка.** В профиле опечатка в `chunk_overlap`. Сервер отвечает 422, CLI печатает
`Problem` тем же `render_issues`, что и раньше, и выходит с кодом 2:

```
$ uv run ragglet index build configs/indexes/nfcorpus_hf.yaml
Index profile config is invalid (1 error)
  config.value_error  chunking.chunk_overlap  chunk_overlap must be less than chunk_size
$ echo $?
2
```

Раньше ошибка схемы выходила трейсбеком pydantic с кодом 1: `load_index_profile` вызывался
вне `exit_on_validation_error`. Теперь все ошибки конфига — это 422, и печатаются они одинаково.

**Сборка и прогон.** Исправленный профиль уходит в `POST /v1/builds` и сразу стартует. Во
втором терминале запускается прогон, и сервер ставит его в очередь за сборкой:

```
$ uv run ragglet bench run configs/scenarios/nfcorpus_hf.yaml
Queued: nfcorpus_hf_20261125T101540Z (position 1; running now: build nfcorpus_hf_20261125T101500Z)
Started: preparing (preflight, loading models)...
queries ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 323/323 0:01:41
Finishing (writing summary)...
Done
{'scenario': 'nfcorpus_hf', 'repeats': 1, 'doc_ndcg@10_mean_mean_over_repeats': 0.315, ...}
Run: nfcorpus_hf_20261125T101540Z
Artifacts (on the server): results/runs/nfcorpus_hf_20261125T101540Z
```

Прогон шёл в свежем процессе воркера (ragglet-29), попал в реестр с `origin: api` и виден
в UI, как и прогоны, запущенные оттуда.

**Передумал.** На 120/323 исследователь жмёт Ctrl-C:

```
^C
Cancelling nfcorpus_hf_20261125T101540Z... (Ctrl-C again to stop waiting)
Cancelled at queries 120/323
```

CLI отправил `POST /v1/runs/{id}/cancel` и дождался `cancelled`. Процесс CLI завершился
сигналом SIGINT, поэтому цикл `for` в bash остановился, а не перешёл к следующей итерации.

**Посмотреть.**

```
$ uv run ragglet runs list --scenario nfcorpus_hf --limit 3
 id                              state      duration  ndcg@10  recall@10  p50 ms  p95 ms   qps
 nfcorpus_hf_20261125T101540Z    cancelled  0:00:58         —          —       —       —     —
 nfcorpus_hf_20261124T200301Z    completed  0:01:41     0.315      0.150   152.5   232.1  95.4
 nfcorpus_hf_20261024T180917Z    completed  —           0.315      0.150   150.9   229.8  96.0
3 of 9
```

## Решение

### 1. Клиент и адрес сервера

`cli/client.py` — тонкая обёртка над синхронным `httpx.Client`:

```python
class ServerUnavailable(Exception): ...      # не подключиться; ответ не от ragglet; связь потеряна
class ApiError(Exception):                   # ответ не 2xx: status и тело Problem
    status: int
    problem: Problem

class ApiClient:
    def __init__(self, http: httpx.Client): ...
    def request(self, method: str, path: str, *, json=None, params=None, long: bool = False) -> httpx.Response: ...

def connect(api_url: str) -> ApiClient: ...  # httpx.Client(base_url=api_url, timeout=...)
```

- **Адрес** — опция корня `ragglet --api URL …` с переменной `RAGGLET_API_URL`, по
  умолчанию `http://127.0.0.1:8000`: это адрес `ragglet serve` по умолчанию
  (`ServerSettings.host`, `port`). Если сервер запущен на другом порту, адрес задаётся
  явно. Читать `RAGGLET_HOST` и `RAGGLET_PORT` сервера CLI не будет: это настройки
  другого процесса, а `0.0.0.0` не адрес для подключения.
- **Таймауты.** Подключение — 3 с. Ответ — 30 с, но у `POST /v1/runs` и `POST /v1/builds`
  ожидание ответа не ограничено (`long=True`): preflight синхронный, а первое
  обращение к `hf`-датасету скачивает его (ragglet-28 п. 8).
- **Сервер недоступен.** `httpx.ConnectError` и `ConnectTimeout` превращаются в
  `ServerUnavailable`, и CLI печатает сообщение из примера с кодом 3. Код 3 будет и в
  том случае, если по адресу отвечает не ragglet, то есть тело ошибки не `Problem` (на
  порту 8000 часто сидит что-то своё). Если связь пропала во время ожидания задачи,
  CLI печатает `lost connection to the server while waiting for <id>; check later:
  ragglet runs show <id>` и тоже выходит с кодом 3. Повторных попыток нет: после
  остановки сервера задача всё равно станет `failed` с `ServerShutdown` (ragglet-29 п. 8).
- **Ответы** разбираются моделями из `ragglet.api.schemas` (`RunCheck`, `RunStatus`,
  `RunList`, `Problem` и т.д.). Это тот же пакет и тот же контракт, что у сервера, поэтому
  опечатка в имени поля ловится тестом, а не в терминале. Импорт `ragglet.api.schemas`
  стоит около 0,4 с и не тянет torch и pandas (проверено). Сейчас один только старт CLI
  занимает около 8 с, потому что импортирует torch.

### 2. Команды и их соответствие API

| Команда | API | Вывод |
|---|---|---|
| `bench run FILE [--dry-run] [--tag T]… [--no-wait] [--json]` | `POST /v1/runs`, затем опрос `GET /v1/runs/{id}` | отчёт dry-run / ожидание и summary (п. 3, 4) |
| `index build FILE [--dry-run] [--no-wait] [--json]` | `POST /v1/builds`, затем опрос `GET /v1/builds/{id}` | то же для сборки |
| `runs list [--state] [--scenario] [--tag] [--sort] [--limit 20] [--json]` | `GET /v1/runs` | таблица (п. 6) |
| `runs show ID [--json]` | `GET /v1/runs/{id}?include=summary` | карточка прогона и summary |
| `runs wait ID` | опрос `GET /v1/runs/{id}` | как ожидание в `bench run` |
| `runs cancel ID` | `POST /v1/runs/{id}/cancel`, затем опрос до конца | итоговое состояние |
| `runs log ID [--follow]` | `GET /v1/runs/{id}/log?offset=` | текст лога |
| `builds list / show / wait / cancel / log` | то же под `/v1/builds` | то же |
| `tasks [--json]` | `GET /v1/tasks` | текущая задача и очередь |
| `stand [--json]` | `GET /v1/stand` | сервисы, вычислитель, коммит, очередь |
| `serve` | — | без изменений |
| `dataset import` | — | без изменений, локально (п. 8) |

`runs` и `builds` — одна фабрика команд, параметризованная видом задачи: endpoints и
статусы у них устроены одинаково. Библиотеку сценариев, коллекции и per-query CLI не
показывает (см. «Вне скоупа»): это страницы UI, а из терминала их и так видно через curl.

### 3. `bench run` и `index build`

**Файл уходит содержимым.** CLI читает файл как UTF-8 и кладёт текст в `scenario` /
`profile`, как требует рамка 38. YAML на клиенте не разбирается: ошибки синтаксиса и
схемы приходят от сервера как 422 `config.*` с `path`, и путь их печати один.

**Ссылка на библиотеку.** Перед постановкой CLI делает `GET /v1/scenarios/<имя файла без
.yaml>` (у сборки — `/v1/index-profiles/…`). Если ответ 200, сценарий в библиотеке
`valid` и его `yaml` совпадает с текстом файла байт в байт, в запрос идёт
`scenario_id` (`profile_id`). Тогда `last_run` в библиотеке UI и пометка «сценарий
изменился после прогона» работают и для прогонов из терминала. Совпадение текста
означает, что запущен именно файл библиотеки. Иначе ссылки нет, как у прогона из
временной копии: 404, 422 на недопустимое имя файла и любое расхождение — не ошибка. Цена —
один лёгкий GET.

**Dry-run** печатает `RunCheck` в формате нынешнего `_echo_preflight`: все строки на
месте, данные берутся из `dataset` и `index` ответа (`index` — те же метаданные ragglet
primary-цели, что у `Preflight.index`). Добавляются строки `plan` и по строке на цель
(`targets`). Предупреждения печатаются после отчёта, как сейчас: `Preflight warnings`.

`index build --dry-run` печатает `BuildCheck`. Строки `dataset: …, documents: N`,
`corpus sha256: …` и `will recreate: <collection> at <endpoint> (target <id>)` остаются
как есть, а после `;` к ним дописывается то, что API знает о цели: `exists, 3633 points,
built …, used by: nfcorpus_hf` или `unreachable: <error>`. Дамп профиля
(`rprint(cfg.model_dump())`) уходит: CLI не разбирает YAML, а файл и так перед глазами.

**422** — `ApiError` с `Problem` → `render_issues(problem.title, issues, problem.hint)` в
stderr и код 2. `ProblemIssue` переводится в `Issue`, а колонка «где» — это
`where or path`. У проблем ядра `where` заполнен, и вывод совпадает с нынешним
`str(ValidationError)` символ в символ. У ошибок схемы `where` пустой, и там стоит путь
поля: `chunking.chunk_overlap`.

**202** → предупреждения постановки (`RunAccepted.warnings`) печатаются сразу: если
индекс ещё `yellow`, исследователь узнает об этом до прогона, а не после. Дальше
ожидание (п. 4). С `--no-wait` CLI печатает в stdout только id, чтобы его можно было
взять в `id=$(…)`, а в stderr — `Queued: … Follow: ragglet runs wait <id>`. Код 0.

### 4. Ожидание задачи

`cli/watch.py`, общий для `bench run`, `index build`, `runs wait` и `builds wait`:

```python
def watch(api: ApiClient, kind: Kind, task_id: str) -> int:   # код выхода
    while True:
        status = api.get_status(kind, task_id)                    # лёгкий GET без include
        show(status)                                              # строки смены состояния, полоса
        if status.state in TERMINAL:
            return finish(api, kind, status)
        sleep(1.0)                                                # подменяется в тестах
```

- **Опрос раз в секунду**, как у UI (ragglet-28 п. 9): ответ без `include` лёгкий, а
  сервер — отдельный процесс от воркера, так что замеры опрос не трогает. Стриминга в
  API нет (SSE — вне скоупа 28).
- **Что видно.** Смена состояния и фазы — отдельная строка: `Queued: … (position 1;
  running now: …)` (что выполняется сейчас, CLI берёт один раз из `GET /v1/tasks`),
  `Started: preparing (preflight, loading models)...`, `Finishing (writing
  summary)...`. Фаза `progress` — полоса rich: стадия (`queries` / `corpus`), `done` /
  `total`, прошедшее время и оценка остатка. Полоса перерисовывается после каждого
  опроса, без фонового потока. Если вывод не в терминал (файл, CI), полосы нет, а строки
  смены фаз остаются. Фазу считает сервер (`phase`), так что CLI и UI показывают одно и то
  же.
- **`completed`.** Ещё один `GET …?include=summary`. У прогона CLI печатает `Done`,
  `summary.aggregate` тем же `rprint`, что сейчас печатает `RunResult.summary`, затем
  `Run: <id>` и `Artifacts (on the server): <run_dir>`. Путь серверный: при локальном
  `ragglet serve` он считается от рабочего каталога сервера, в Docker это путь внутри
  контейнера. Поля с путём в `RunStatus` пока нет, см. п. 9. У сборки печатается
  `Done: 3633 documents`, а дальше по строке на цель: `nfcorpus_v1 at
  http://localhost:6333`.
- **`failed`.** Если в `error.problem` что-то есть (preflight №2 при старте не прошёл:
  индекс пересобрали между постановкой и стартом), CLI печатает его через
  `render_issues` и выходит с кодом 2, как воркер. Иначе печатает
  `Failed: <type>: <message> (qid=<qid>)` и последние 20 строк лога. Лог читается через
  `…/log?offset=` от нуля до конца, это несколько запросов по 64 КиБ. В конце подсказка
  `Full log: ragglet runs log <id>`. Код 1.
- **`cancelled`** (например, отменили из UI): `Cancelled at queries 120/323` из
  `progress`, код 1.

### 5. Ctrl-C во время ожидания: отмена задачи

**Предлагаю: Ctrl-C отменяет задачу**, а не только прекращает ожидание.

1. CLI ловит `KeyboardInterrupt`, отправляет `POST …/cancel` и печатает `Cancelling <id>...
   (Ctrl-C again to stop waiting)`.
2. Дальше CLI ждёт конечного состояния. Задача из очереди уходит сразу, у идущей менеджер
   шлёт SIGTERM и через `cancel_grace_s` (10 с) — SIGKILL. CLI печатает `Cancelled at
   queries 120/323` (или `completed`, если задача успела закончиться).
3. Второй Ctrl-C прекращает ожидание сразу, отмена уже отправлена.
4. **Процесс завершается сигналом SIGINT**: `signal(SIGINT, SIG_DFL)` и `os.kill(getpid(),
   SIGINT)`, а не `exit(130)`. Проверено: если дочерний процесс обработал SIGINT и вышел
   с `exit(130)`, `for` в bash идёт на следующую итерацию, а если умер от SIGINT —
   останавливается. Click по умолчанию превращает Ctrl-C в `Aborted!` и код 1, так что
   нынешний CLI в цикле из `docs/article-experiments-audit.md` по Ctrl-C прерывает один
   прогон и запускает следующий. В тестах эта функция подменяется.
5. Ctrl-C до ответа 202 (пока идёт синхронный preflight) обрывает запрос. Сервер свой
   обработчик доводит до конца и может поставить задачу. Поэтому CLI печатает `The server
   may have queued it: check ragglet tasks` и выходит так же, сигналом.

Почему так:

- `bench run` значит «прогони и дай результат», CLI к задаче привязан, как
  `docker run`. Нынешний CLI по Ctrl-C прерывает прогон, и эта привычка сохраняется.
- Если бы Ctrl-C только отвязывал CLI от задачи, ненужный прогон продолжал бы занимать
  стенд. Следующий `bench run` с исправленным сценарием молча встал бы за ним в
  очередь, а в цикле cold/warm из статьи очистка Redis пошла бы параллельно с ещё
  идущим прогоном.
- Отвязаться можно явно: `--no-wait`, потом `runs wait <id>`. Закрытие терминала или
  обрыв SSH (SIGHUP) задачу не отменяют: CLI просто умирает, а задача идёт дальше.
  Отменяет только явный Ctrl-C.
- Отменённый прогон не пропадает: он остаётся в реестре с итоговым прогрессом и с
  per-query законченных повторов (ragglet-29 п. 5).

`runs wait` и `builds wait` ведут себя так же, потому что это то же ожидание. В
`runs log --follow` Ctrl-C только прекращает чтение лога (код 0): задачу эта команда не
трогает.

### 6. Команды просмотра

- **`runs list`** — таблица rich из `RunList`: `id`, `state`, `duration`, `ndcg@k`,
  `recall@k`, `p50 ms`, `p95 ms`, `qps`. Вместо `null` ставится `—`, под таблицей —
  `N of total`. Опции передаются в одноимённые параметры запроса: `--state`,
  `--scenario`, `--tag`, `--sort` (например `-ndcg`), `--limit` (по умолчанию 20). Остальные
  фильтры 30 (`dataset`, `collection`, `q`, `since`, `k`) доступны через curl. Добавить
  опцию — одна строка, если понадобится.
- **`builds list`** — `id`, `state`, `duration`, `dataset`, цели (`collection`),
  `documents`.
- **`runs show ID`** — карточка `RunStatus` строками `ключ: значение`: id и `origin`,
  сценарий и ссылка на библиотеку (`scenario_file`: `unchanged` / `changed` / `deleted`),
  состояние с фазой и прогрессом или позицией в очереди, времена и длительность,
  датасет, коллекция, теги, путь артефактов (п. 9), ошибка (как в п. 4) и предупреждения
  постановки. У `completed` ниже идёт summary, как в `bench run`. `builds show` — то же
  для `BuildStatus` с `result`.
- **`runs cancel ID`** — `POST …/cancel`. Задача из очереди сразу становится `cancelled`,
  у идущей CLI ждёт конечного состояния, как в п. 5. 409 `task.finished` — `already
  completed` и код 1.
- **`runs log ID`** печатает лог от начала до текущего конца. С `--follow` CLI раз в
  секунду дочитывает лог с `next_offset`, пока `complete` не станет `true`.
- **`tasks`** — таблица `TaskItem`: вид, id, имя, состояние, позиция, прогресс
  `stage done/total`. Пустая очередь — `No active tasks`.
- **`stand`** — `StandStatus` строками: каждый Qdrant (`up`/`down`, версия, число
  коллекций или ошибка), Redis, `compute` (устройство и потоки torch), коммит и Python
  сервера, переменные `env` (`set` / `not set`, значений API не отдаёт), текущая задача
  и сколько ждут.

### 7. Коды выхода и `--json`

| Код | Когда |
|---|---|
| 0 | dry-run прошёл; задача `completed`; `--no-wait` принят; команда просмотра выполнилась |
| 1 | задача `failed` (кроме preflight №2) или `cancelled`; ответ API 404, 409, 500 — печатается `Problem` |
| 2 | 422: проблемы конфига, датасета, индекса; задача `failed` с `error.problem` |
| 3 | сервер недоступен, связь потеряна во время ожидания, по адресу отвечает не ragglet |
| SIGINT (`$?` = 130 в bash) | Ctrl-C во время ожидания: задача отменена (п. 5) |

Коды 0 и 2 совпадают с нынешними, 1 сейчас означает любой трейсбек. Одна функция
`api_errors()` в `render.py` переводит `ServerUnavailable` и `ApiError` в вывод и код,
как сейчас это делает `exit_on_validation_error`.

**`--json`** есть у команд, которые печатают объект API: `bench run`, `index build`,
`runs|builds list|show`, `tasks`, `stand`. Вместо текста они печатают тело ответа как
есть. У `bench run` без `--no-wait` это итоговый `RunStatus` с `summary`, полосы при этом
нет, у dry-run — `RunCheck`. При ошибке печатается тело `Problem`, коды выхода те же.
Каждый такой флаг — одна строка, а скриптам не нужно разбирать таблицы:
`ragglet runs list --json | jq …` короче, чем собирать URL для curl.

### 8. Что уходит из CLI и что остаётся локальным

**Уходит.** Из `cli/` исчезают импорты `run_scenario`, `build_index`, `Preflight`,
`load_scenario`, `load_index_profile`, `ScenarioConfig`, `describe_dataset`, `anyio`. Из
`render.py` уходят `exit_on_validation_error` (его место занимает `api_errors()`) и
`progress_bar` с callback ядра (его место занимает полоса по опросу в `watch.py`).
`render_issues` остаётся: это форматирование из `core/issues.py` без зависимостей, а не
операция ядра. Правило проверяет тест: после `import ragglet.cli.app` в `sys.modules`
нет `ragglet.bench.run_scenario`, `ragglet.indexing.build_index` и `torch`.

Тесты ядра и API от удалённого не зависят. CLI используют только четыре теста:
`test_cli_dry_run_prints_preflight_report`, `test_cli_validation_error_exits_with_code_2`,
`test_cli_unsupported_strategy_exits_with_code_2` в `test_preflight.py` и
`test_cli_dry_run_lists_collections_to_recreate` в `test_build_index.py`. Они переезжают
в `tests/test_cli.py` и идут уже через API. Проверки ядра за ними остаются в тестах ядра
и API. `core/progress.py` по-прежнему нужен ядру и воркеру. `serve` в
`test_api_stand.py` не меняется.

**`dataset import` остаётся локальным.** Он переводит каталог `.md`/`.txt` в файлы BEIR
на диске клиента: не трогает Qdrant, Redis и модели, не создаёт прогонов и сборок и в
реестр не попадает. Рамка сводит в один вход то, что трогает стенд (прогон, dry-run,
сборка) и историю прогонов (решение 1). Конвертация файлов к этому не относится, это
такая же утилита, как `jq`. Стенда касается уже следующий шаг: профиль с
`dataset.id: <out_dir>` идёт через API. Отсюда одно условие, которое пишется в README:
`out_dir` должен быть виден серверу по тому же пути, относительно его рабочего каталога.
Локально это так и есть, в Docker нужен общий том (ragglet-34).

**`serve` не меняется.** Ядро по-прежнему можно вызвать без сервера, как и разрешает
рамка: из Python и `uv run python -m ragglet.jobs.worker results/runs/<id>` для отладки
задачи (ragglet-29). Отдельный «локальный» путь в CLI не появляется.

### 9. Требование к API: путь каталога задачи

В `RunStatus` и `BuildStatus` нет пути к каталогу, а печатать `Artifacts: <run_dir>`
нужно: по этому пути исследователь открывает per-query в pandas и передаёт его
`scripts/analyze_article_runs.py`. Нужны поля

```python
class RunStatus(_Live):
    run_dir: str          # str(record.run_dir): путь, как его видит сервер
class BuildStatus(_Live):
    build_dir: str
```

С настройками по умолчанию это относительный путь `results/runs/<id>` от рабочего
каталога сервера, и он верен и локально, и в Docker, если `results/` смонтирован тем же
путём. Данные уже лежат в записи реестра, так что это две строки в роутерах, два поля и
тест. По координации `src/ragglet/api/` этот PR не меняет, поэтому поле записано как
требование (открытый вопрос 2). Пока поля нет, CLI печатает вместо пути
`Artifacts: <api>/v1/runs/<id>/artifacts.zip`.

## Что меняется для пользователя

- Перед `bench run` и `index build` должен работать сервер: `uv run ragglet serve` в
  соседнем терминале или docker compose (ragglet-34). Без сервера команда выходит с
  кодом 3 и объясняет, что делать.
- Команды, аргументы и вывод `bench run` и `index build` прежние. Новое в них:
  `--no-wait`, `--tag`, `--json`; ожидание очереди; строки фаз; предупреждения сразу
  после постановки. `index build` больше не печатает дамп профиля.
- Пути в конфиге разрешает сервер: `dataset.id: datasets/demo` считается от рабочего
  каталога сервера, а не CLI. `run.artifacts_dir` сервер игнорирует, прогоны пишутся в
  `RAGGLET_ARTIFACTS_DIR` (`results/runs/<id>`), и это касается серии статьи
  (вопрос 3). Имя каталога — id постановки `<name>_<UTC>[-N]`, а не время старта.
- Ошибки схемы конфига печатаются списком с путём поля и кодом 2, а не трейсбеком
  pydantic.
- Сбой прогона печатается одной строкой с типом, сообщением и `qid`, за ней идёт хвост
  лога. Полный трейсбек выдаёт `ragglet runs log <id>`.
- Ctrl-C отменяет задачу на сервере и останавливает цикл bash. Отвязаться от задачи —
  `--no-wait`.
- Прогоны из терминала попадают в реестр с `origin: api`, видны и отменяются в UI. Два
  терминала больше не запускают два прогона одновременно: второй ждёт в очереди.
- CLI стартует примерно за 0,6 с вместо 8: torch не импортируется.

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `cli/client.py` (новый) | `ApiClient`, `connect`, `ServerUnavailable`, `ApiError` |
| `cli/watch.py` (новый) | ожидание задачи: опрос, фазы, полоса, итог, Ctrl-C |
| `cli/tasks.py` (новый) | группы `runs` и `builds` (одна фабрика), `tasks`, `stand` |
| `cli/app.py` | `--api` / `RAGGLET_API_URL` на корне, новые группы; `serve` как есть |
| `cli/bench.py`, `cli/index.py` | через `ApiClient`; без импортов ядра |
| `cli/render.py` | отчёты dry-run из `RunCheck` / `BuildCheck`, `Problem` → `render_issues`, таблицы, `api_errors()`; без `progress_bar` и `exit_on_validation_error` |
| `tests/test_cli.py` (новый) | тесты CLI через API; из `test_preflight.py` и `test_build_index.py` уходят 4 CLI-теста |
| `README.md` | разделы команд (ниже) |
| `docs/article-experiments-audit.md` | сервер перед серией, куда пишутся прогоны |

`src/ragglet/api/`, `jobs/`, `registry/` и ядро не меняются (кроме вопроса 2).

## Документация

- **README, «Как запустить»**: новый шаг «Запустить сервер» перед «Собрать индекс»
  (`uv run ragglet serve` в отдельном терминале, адрес и `RAGGLET_API_URL`). В шагах 3 и
  4 команды прежние, к ним добавляется: команда ждёт очередь, Ctrl-C отменяет задачу,
  `--no-wait`. В «Посмотреть результаты»: каталог `results/runs/<id>/`, где id — время
  постановки, и `ragglet runs list` / `runs show`. В «REST API» уходит фраза «До
  ragglet-37 CLI вызывает ядро напрямую», и CLI называется клиентом API.
- **README, «Полезные команды»**: `runs list`, `runs show`, `runs log --follow`,
  `runs cancel`, `tasks`, `stand`, `--api`.
- **README, «Частые проблемы»**: «`cannot reach the ragglet server` (код 3)» —
  запустить `ragglet serve` или compose, проверить `--api` / `RAGGLET_API_URL`; путь
  датасета считается от рабочего каталога сервера; `run.artifacts_dir` игнорируется.
  Абзац про `docker compose --profile app` («рабочий путь — локальный CLI») меняется:
  рабочий путь — `ragglet serve` плюс CLI.
- **`docs/article-experiments-audit.md`**: в подготовке появляется `uv run ragglet serve`.
  Команды `bench run` и `index build` остаются прежними. «Пятью независимыми
  CLI-процессами» меняется на «пятью задачами, каждая в свежем процессе воркера»:
  независимость сохраняется (ragglet-29 п. 1). Цикл cold/warm работает, как раньше:
  `bench run` ждёт конца прогона, и очистка Redis идёт строго между прогонами. Куда
  пишутся прогоны и как их находит анализатор — вопрос 3.
- **Docker (ragglet-34), что нужно от compose**: сервис с `ragglet serve`,
  `RAGGLET_HOST=0.0.0.0` и портом `127.0.0.1:8000:8000`, чтобы адрес CLI по умолчанию
  работал. `configs/`, `datasets/` и `results/` монтируются теми же относительными путями
  от рабочего каталога сервера, тогда `datasets/demo` и `results/runs/<id>` значат одно
  и то же на хосте и в контейнере. Endpoint `http://localhost:6333` в сценариях внутри
  контейнера указывает на сам контейнер, это 34 решает отдельно.
- Раздел README от ragglet-35 (сквозная проверка) этот PR не трогает.

## Тесты (критерий приёмки)

Без живого сервера, Qdrant и torch. `make_stand` из `tests/api_support.py` уже собирает
`create_app` с `InMemoryJobManager`, `InMemoryRegistry` или `FileRegistry` и фейковым
Qdrant. Его `TestClient` — подкласс `httpx.Client`, поэтому тест подменяет
`cli.client.connect` на `ApiClient(stand.client)`, и CLI ходит в настоящие роутеры через
ASGI. `ASGITransport` httpx только асинхронный, а `WSGITransport` FastAPI не подходит,
поэтому годится именно `TestClient`. `watch.sleep` подменяется сценарием, который на
каждом тике двигает заглушку менеджера: `anyio.run(jobs.start, ref)`, `jobs.report(ref,
"queries", 3, 10)`, `jobs.append_log`, `anyio.run(jobs.finish, ref, state, failure)`.

1. **Dry-run прогона**: код 0, `Preflight: OK`, `corpus sha256: <DEMO_SHA>`,
   `embedding: all-MiniLM-L6-v2@<MODEL_SHA>`, строка `plan`. Каталог прогона не создан.
2. **422**: `cascade` — код 2 и `retrieval.not_implemented`. Нет метаданных индекса —
   код 2 и `index.no_metadata`. Битый YAML — код 2 и `config.yaml` с `line 3, column 2`.
   Лишнее поле — код 2 и `config.extra_forbidden` с путём в колонке «где».
3. **Dry-run сборки**: `will recreate: chunks_v1 at http://q:6333 (target default)` и
   `exists, 53 points`.
4. **202 и ожидание**: `queued` → `running` без прогресса → `report` 3/10 → 10/10 →
   `completed`. В выводе по порядку `Queued`, `Started: preparing`, `Finishing`, `Done`,
   ключ aggregate (`FileRegistry`, тест кладёт `summary.aggregate.json` в каталог
   прогона), `Run: <id>`. Код 0. В записи реестра `tags` из `--tag`.
5. **Ссылка на библиотеку**: файл из библиотеки стенда (копия в `tmp_path/scenarios`) —
   в записи `scenario_id`; изменённая копия — `None`.
6. **`failed`**: `RuntimeError` с `qid` и текстом лога — код 1, в выводе `Failed:
   RuntimeError`, `qid=` и последние строки лога. `ValidationError` при старте (`failure`
   с `title` и `issues`) — код 2 и таблица проблем. `cancelled` из UI — код 1.
7. **Ctrl-C**: `sleep` бросает `KeyboardInterrupt` на втором тике. Задача в очереди
   становится `cancelled`. У `running` CLI отправил отмену (`cancel_requested`), а после
   `finish(cancelled)` печатает `Cancelled at`. Выход сигналом подменён и проверен вызовом.
8. **`--no-wait`**: в stdout ровно id, код 0, задача в очереди.
9. **Сервер недоступен**: `--api http://127.0.0.1:<свободный порт>` без подмены `connect`
   — код 3 и `ragglet serve` в сообщении. Не-`Problem` ответ (подменённый
   транспорт отдаёт HTML 404) — тоже код 3.
10. **Просмотр**: `runs list` после двух постановок показывает оба id и состояния,
    `--json` разбирается как `RunList`; `runs show` у `completed`; `runs cancel`
    завершённого — код 1; `runs log` по смещениям склеивает текст; `tasks` — текущая и
    очередь; `stand` — `down` у Redis из `make_stand`.
11. **Нет импортов ядра**: в подпроцессе `import ragglet.cli.app`, затем проверка
    `sys.modules` (п. 8).
12. Остальные тесты проходят, `ruff check` чистый.

Ручная проверка на Mac: `ragglet serve`, затем `index build --dry-run` и `bench run
--dry-run` по `demo`, `bench run` с Ctrl-C на середине, `runs list`, `runs log`; второй
`bench run` из соседнего терминала встаёт в очередь; `bash -c 'for i in 1 2; do ragglet
bench run …; done'` по Ctrl-C останавливается.

## Фазы (отдельными коммитами)

1. Клиент и ожидание: `client.py`, `watch.py`, `render.py`; `bench run` и `index build`
   через API; 4 CLI-теста переезжают в `test_cli.py`; тест «нет импортов ядра».
2. Команды просмотра: `runs`, `builds`, `tasks`, `stand`, `--json`.
3. Документация: README, `docs/article-experiments-audit.md`.

Если по вопросу 2 решим добавить поле здесь, это отдельный коммит перед фазой 1.

## Вне скоупа

- UI — ragglet-31; Docker и compose — ragglet-34 (что от них нужно, см. «Документация»).
- Авторизация и TLS: сервер слушает `127.0.0.1` (ragglet-28 п. 11). Адрес сервера
  задаётся, но профилей «стендов» и переключения между ними нет.
- Библиотека сценариев и профилей из CLI (`scenarios list/push`), список коллекций
  (`GET /v1/indexes`), per-query и деталь запроса, скачивание артефактов (`runs pull`),
  «Повторить», сравнение прогонов. Всё это есть в API, а в терминале пока не нужно.
- Повторные попытки при обрыве связи и переподключение к задаче после рестарта сервера.
- Совместимость CLI и сервера разных версий: они из одного пакета, и расхождение схем
  видно как ошибка разбора ответа.

## Рассмотренные альтернативы

- **Ctrl-C только прекращает ожидание** с подсказкой `ragglet runs cancel <id>`. Ничего
  не теряется по ошибке, но ненужный прогон держит стенд, следующий запуск молча встаёт за
  ним в очередь, а цикл из статьи начинает следующую итерацию поверх идущего прогона.
  Отвязаться явно позволяет `--no-wait`, поэтому выбран вариант с отменой (п. 5).
- **Падать в ядро, если сервер не запущен.** Запрещено решением 6 рамки: снова два входа
  в стенд.
- **Асинхронный клиент httpx.** CLI делает одно действие за раз, синхронный проще, и
  `TestClient` для тестов — синхронный `httpx.Client`.
- **Ответы как dict, без моделей `api/schemas.py`.** Опечатка в имени поля превращается в
  `KeyError` в терминале, а не в падение теста. Импорт схем стоит 0,4 с.
- **Сгенерированный по OpenAPI клиент.** Лишний шаг сборки при том, что схемы лежат в том
  же пакете.
- **Разбирать YAML на клиенте** ради ранних ошибок. Дублирует проверку сервера и даёт
  второй формат ошибок. 422 приходит за миллисекунды и уже с `path`.
- **Путь к файлу вместо содержимого.** Нарушает рамку: сервер может быть в контейнере.
- **`scenario_id` по имени файла без сверки текста.** Временная копия с тем же именем
  получила бы ссылку на чужой файл библиотеки, а UI показал бы «изменился» или
  «удалён». Сверка — один GET.
- **Bash + curl вместо Python-клиента.** Отвергнуто рамкой: опрос, отмену и коды выхода
  там не протестировать.

## Открытые вопросы

1. **Ctrl-C** отменяет задачу (предлагаю, п. 5) или только прекращает ожидание?
2. **`run_dir` / `build_dir` в `RunStatus` / `BuildStatus`** (п. 9). Предлагаю добавить в
   этом PR отдельным коммитом в `api/`: это два поля из записи реестра, 35 и 31 эти
   роутеры не меняют. Альтернатива — отдельная мелкая задача, а до неё CLI печатает
   ссылку на `artifacts.zip`.
3. **Серия статьи.** Её сценарии пишут в `results/article/raw` и `results/article/smoke`
   через `run.artifacts_dir`, а сервер это поле игнорирует, и через API серия попадёт в
   `results/runs/<id>`. `analyze_article_runs.py analyze` ищет прогоны по
   `--runs` (по имени сценария, поэтому `--runs results/runs` подходит), но smoke берёт
   из `<runs>/../smoke`. Предлагаю добавить анализатору `--smoke-runs` (по умолчанию как
   сейчас) и запускать `analyze --runs results/runs --smoke-runs results/runs`. Это
   несколько строк в `scripts/`, и с 35 нужно договориться, что этот файл они не
   трогают. Альтернатива без кода: на время серии `RAGGLET_ARTIFACTS_DIR=results/article/raw
   ragglet serve`, а smoke прогнать отдельным запуском сервера с `…/smoke`.
