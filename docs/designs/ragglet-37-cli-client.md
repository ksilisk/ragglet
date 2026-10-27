# ragglet-37 — CLI как клиент единого API

**Коротко.** По решению 6 рамки ragglet-38 CLI одним шагом перестаёт вызывать ядро и
становится клиентом REST API (ragglet-28). Локального режима нет. Клиент — один файл
`cli/ragglet_cli.py` на стандартной библиотеке Python 3.9+, без зависимостей, uv и venv.
На macOS и Linux это ссылка `ragglet` в PATH, на Windows — `py cli\ragglet_cli.py`. Всё
делается в одном терминале: `ragglet server start` поднимает сервер в фоне и печатает его
адрес (там же потом будет UI), а если сервер не запущен, команды так и говорят и выходят
с кодом 3. `bench run` и `index build` — прежние команды с прежним выводом. Файл конфига
уходит на сервер содержимым, `--dry-run` печатает прежний отчёт, ответ 422 — прежний
список проблем с кодом 2. После 202 CLI ждёт задачу, рисует полосу прогресса и печатает
summary. Ctrl-C отменяет задачу, `--no-wait` только ставит её в очередь. Новые команды:
`server`, `runs`, `builds`, `tasks`, `stand`. Typer-CLI `ragglet.cli` и `dataset import`
удаляются. В ответы API добавляется путь к каталогу прогона и сборки, а прогоны серии
статьи отбираются по тегу.

## Пример, на котором всё объясняется

Вечер из ragglet-29: исследователь пересобирает `nfcorpus_hf` (профиль
`configs/indexes/nfcorpus_hf.yaml`) и прогоняет сценарий `nfcorpus_hf` на 323 запросах.
Всё происходит в одном терминале, в клоне репозитория. Qdrant и Redis подняты, как
сейчас, через `docker compose`.

**Сервер не запущен.**

```
$ ragglet bench run configs/scenarios/nfcorpus_hf.yaml
Error: cannot reach the ragglet server at http://127.0.0.1:8000 (connection refused).
Start it with `ragglet server start` in the ragglet directory, or set --api / RAGGLET_API_URL.
$ echo $?
3
```

Ядро CLI не вызывает.

**Поднять сервер.**

```
$ ragglet server start
Preparing the server environment (uv sync, the first start downloads dependencies)...
Starting the server, log: .ragglet/server.log
Ragglet is running: http://127.0.0.1:8000 (API /v1, docs /v1/docs)
```

Сервер работает в фоне, терминал свободен. Окружение сервера (torch, модели) по-прежнему
ставится по `uv.lock`, но uv вызывает сам CLI, а человек его не набирает.

**Проверка.** CLI шлёт `POST /v1/runs {"scenario": "<текст файла>", "scenario_id":
"nfcorpus_hf", "dry_run": true}` и печатает ответ 200 (`RunCheck`) в прежнем формате. Две
последние строки новые: эти данные уже есть в ответе.

```
$ ragglet bench run configs/scenarios/nfcorpus_hf.yaml --dry-run
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
`Problem` тем же форматом `render_issues`, что и раньше, и выходит с кодом 2:

```
$ ragglet index build configs/indexes/nfcorpus_hf.yaml
Index profile config is invalid (1 error)
  config.value_error  chunking.chunk_overlap  chunk_overlap must be less than chunk_size
$ echo $?
2
```

Раньше ошибка схемы выходила трейсбеком pydantic с кодом 1: `load_index_profile`
вызывался вне `exit_on_validation_error`. Теперь любая ошибка конфига — это 422, и все они
печатаются одинаково.

**Сборка и прогон.** Исправленный профиль уходит в `POST /v1/builds`, и сборка сразу
стартует. Если поставить прогон с `--no-wait` или из UI, пока сборка идёт, сервер поставит
его в очередь за ней:

```
$ ragglet bench run configs/scenarios/nfcorpus_hf.yaml
Queued: nfcorpus_hf_20261125T101540Z (position 1; running now: build nfcorpus_hf_20261125T101500Z)
Started: preparing (preflight, loading models)...
queries [########################################] 323/323  1:41
Finishing (writing summary)...
Done
{'scenario': 'nfcorpus_hf', 'repeats': 1, 'doc_ndcg@10_mean_mean_over_repeats': 0.315, ...}
Run: nfcorpus_hf_20261125T101540Z
Artifacts (on the server): results/runs/nfcorpus_hf_20261125T101540Z
```

Прогон шёл в свежем процессе воркера (ragglet-29), попал в реестр с `origin: api` и виден
в UI наравне с прогонами, запущенными оттуда.

**Передумал.** На 120/323 исследователь жмёт Ctrl-C:

```
^C
Cancelling nfcorpus_hf_20261125T101540Z... (Ctrl-C again to stop waiting)
Cancelled at queries 120/323
```

CLI отправил `POST /v1/runs/{id}/cancel` и дождался `cancelled`. Процесс CLI завершился
сигналом SIGINT, поэтому цикл `for` в bash остановился, а не перешёл к следующей итерации.

**Посмотреть и закончить.**

```
$ ragglet runs list --scenario nfcorpus_hf --limit 3
id                            state      duration  ndcg@10  recall@10  p50 ms  p95 ms   qps
nfcorpus_hf_20261125T101540Z  cancelled  0:00:58         -          -       -       -     -
nfcorpus_hf_20261124T200301Z  completed  0:01:41     0.315      0.150   152.5   232.1  95.4
nfcorpus_hf_20261024T180917Z  completed  -           0.315      0.150   150.9   229.8  96.0
3 of 9
$ ragglet server stop
Stopping the server (pid 51234)...
Stopped.
```

## Решение

### 1. Один файл на стандартной библиотеке

Клиент — `cli/ragglet_cli.py`: исполняемый файл с `#!/usr/bin/env python3`, который
импортирует только стандартную библиотеку (`urllib.request`, `argparse`, `json`,
`signal`, `subprocess`). Рядом лежит `cli/pyproject.toml`: пакет `ragglet-cli`,
`dependencies = []`, `requires-python = ">=3.9"`, скрипт `ragglet = "ragglet_cli:main"`.
Он нужен только тем, кто хочет ставить клиент через pip.

**Как запускать** (README):

| Где | Как |
|---|---|
| macOS, Linux | один раз `ln -s "$PWD/cli/ragglet_cli.py" ~/.local/bin/ragglet` (или в любой каталог из PATH), дальше `ragglet …`; файл обновляется вместе с `git pull` |
| Windows | `py cli\ragglet_cli.py …`; или `pip install .\cli`, тогда появится `ragglet.exe` |
| любая ОС | `pipx install ./cli` или `pip install ./cli` в venv |

- **Почему stdlib.** Для входа нужен только Python, ставить ничего не надо. Python 3.9 —
  это `/usr/bin/python3` на macOS (Xcode или Command Line Tools ставят 3.9.6, проверено),
  поэтому клиент работает даже с системным интерпретатором. Не возникает ошибки `externally-managed-environment` (PEP
  668), с которой `pip install` падает на Homebrew-Python. Старт — около 50 мс (замерено).
  Для сравнения: typer, httpx и rich импортируются около 0,3 с, а нынешний CLI стартует
  около 8 с, потому что тянет torch.
- **Чем платим.** argparse вместо typer. HTTP на `urllib.request`. Свои таблицы и полоса
  прогресса — около 50 строк: столбцы выровнены, полоса рисуется одной строкой через `\r`.
  Вывод только ASCII (`-` вместо `—`, `#` в полосе), чтобы не упереться в кодировку
  консоли Windows при перенаправлении в файл. Ответы API — словари из JSON: модели
  `api/schemas.py` импортировать нельзя, это пакет сервера. Поэтому расхождение имён полей
  ловят тесты CLI, которые ходят в настоящие роутеры. `render_issues` (20 строк)
  копируется в клиент, а тест сверяет его вывод с `ragglet.core.issues.render_issues`.
- **Совместимость с 3.9** проверяют тесты: `ast.parse(source, feature_version=(3, 9))`,
  `from __future__ import annotations` и импорты только из `sys.stdlib_module_names`.
- Рамка 38 в решении 6 называла typer + httpx. Док 38 уточняется в этом PR. Bash + curl
  по-прежнему отвергнут: опрос, отмену и коды выхода там не протестировать, а на Windows
  нужен WSL.

### 2. Адрес сервера

Адрес берётся из первого источника, где он задан:

1. `--api URL`;
2. `RAGGLET_API_URL`;
3. `.ragglet/server.json` в текущем каталоге (его пишет `server start`, п. 3);
4. `http://127.0.0.1:8000` — адрес сервера по умолчанию (`ServerSettings.host`, `port`).

- **Таймауты.** У urllib таймаут один на любую операцию с сокетом: 30 с. У `POST /v1/runs`
  и `POST /v1/builds` таймаута нет: preflight синхронный, а первое обращение к
  `hf`-датасету его скачивает (ragglet-28 п. 8).
- **Сервер недоступен** (`ConnectionRefusedError`, таймаут подключения) — сообщение из
  примера и код 3. Код 3 будет и в том случае, если по адресу отвечает не ragglet, то есть
  тело ошибки не `Problem`. Если связь пропала во время ожидания задачи, CLI печатает
  `lost connection to the server while waiting for <id>; check later: ragglet runs show
  <id>` и тоже выходит с кодом 3. Повторных попыток нет: если сервер остановили, задача
  всё равно станет `failed` / `ServerShutdown` (ragglet-29 п. 8).

### 3. Сервер из того же CLI: `ragglet server start | status | stop`

**`server start`** запускается в каталоге стенда, то есть в клоне репозитория (там
`pyproject.toml` пакета `ragglet` и `configs/`). Иначе — ошибка `run it in the ragglet
directory` и код 1.

1. Если по адресу уже отвечает ragglet, CLI печатает `already running at …` и выходит с
   кодом 0.
2. Окружение сервера готовит `uv run --frozen python -c "import sys; print(sys.executable)"`.
   uv приводит `.venv` к `uv.lock`: в первый раз скачивает torch и всё остальное, это
   минуты, и вывод uv виден в терминале. Затем uv печатает путь к интерпретатору. Если uv
   нет, CLI даёт ссылку на установку uv и выходит с кодом 1.
3. CLI запускает `<python> -m ragglet.server --port <port>` в фоне: в новой сессии
   (`start_new_session`), с закрытым stdin, stdout и stderr идут в `.ragglet/server.log`.
   Поэтому ни Ctrl-C, ни закрытие терминала сервер не задевают. pid, адрес и время
   запуска пишутся в `.ragglet/server.json`.
4. CLI ждёт, пока ответит `GET /v1/tasks`, не дольше 120 с (импорт FastAPI и ядра, скан
   реестра). Если процесс вышел раньше, например потому что порт занят, CLI печатает
   последние 20 строк лога и выходит с кодом 1.
5. В конце CLI печатает `Ragglet is running: http://127.0.0.1:8000 (API /v1, docs /v1/docs)`.

Порт задаёт `--port`, по умолчанию 8000. Остальные настройки сервер, как и раньше, читает
сам из `RAGGLET_*` и `.env` каталога стенда (ragglet-28 п. 11). `.ragglet/` добавляется в
`.gitignore`.

**`server status`** печатает адрес, pid, отвечает ли сервер, сколько он работает, путь к
логу, текущую задачу и длину очереди (`/v1/tasks`). Код 0, если сервер отвечает, иначе 3.

**`server stop`** шлёт SIGTERM процессу из `server.json` и ждёт его выхода. Через 15 с
(это `cancel_grace_s` плюс запас) идёт SIGKILL. Затем `server.json` удаляется. Если на
сервере идёт или ждёт задача, команда без `--force` отказывается (код 1) и говорит, какая
задача: остановка сервера переводит текущую и ожидающие задачи в `failed` /
`ServerShutdown` (ragglet-29 п. 8).

- **Почему не `uv run` в фоне.** pid тогда был бы у обёртки uv, а сигнал нужно слать
  самому серверу: на второй SIGTERM uvicorn выходит без штатной остановки, и менеджер
  задач не успевает пометить задачи. Поэтому uv только готовит окружение, а сервер
  CLI запускает интерпретатором напрямую.
- **`python -m ragglet.server`** — это прежний `ragglet serve` из `cli/app.py`, перенесённый
  без изменения поведения в `src/ragglet/server.py`: флаги `--host` и `--port`, печать
  настроек, uvicorn с одним worker. Этой же командой сервер запускается в Docker
  (ragglet-34) и на переднем плане при отладке.
- **Где работает сервер.** Только на macOS и Linux: воркер задач держится на
  POSIX-сигналах (`add_signal_handler`, ragglet-29). На Windows `server start` отвечает,
  что сервер нужно запускать в Docker (ragglet-34) или WSL. Клиент на Windows работает с
  сервером на другой машине или в Docker.
- **Qdrant и Redis** `server start` не поднимает. Они, как и сейчас, запускаются через
  `docker compose … up -d qdrant redis` (шаг 2 README). Свести всё к одной команде через
  compose — задача ragglet-34.

### 4. Команды и их соответствие API

| Команда | API | Вывод |
|---|---|---|
| `server start [--port] / status / stop [--force]` | `GET /v1/tasks` (готовность, задачи) | п. 3 |
| `bench run FILE [--dry-run] [--tag T]… [--no-wait] [--json]` | `POST /v1/runs`, затем опрос `GET /v1/runs/{id}` | отчёт dry-run / ожидание и summary (п. 5, 6) |
| `index build FILE [--dry-run] [--no-wait] [--json]` | `POST /v1/builds`, затем опрос `GET /v1/builds/{id}` | то же для сборки |
| `runs list [--state] [--scenario] [--tag] [--sort] [--limit 20] [--json]` | `GET /v1/runs` | таблица (п. 8) |
| `runs show ID [--json]` | `GET /v1/runs/{id}?include=summary` | карточка прогона и summary |
| `runs wait ID` | опрос `GET /v1/runs/{id}` | как ожидание в `bench run` |
| `runs cancel ID` | `POST /v1/runs/{id}/cancel`, затем опрос до конца | итоговое состояние |
| `runs log ID [--follow]` | `GET /v1/runs/{id}/log?offset=` | текст лога |
| `builds list / show / wait / cancel / log` | то же под `/v1/builds` | то же |
| `tasks [--json]` | `GET /v1/tasks` | текущая задача и очередь |
| `stand [--json]` | `GET /v1/stand` | сервисы, вычислитель, коммит, очередь |

`runs` и `builds` реализованы одной функцией, параметризованной видом задачи: endpoints
и статусы у них устроены одинаково. `--api` и `--json` принимаются в любом месте команды
(общий родительский парсер argparse). Библиотеку сценариев, коллекции и per-query CLI не
показывает (см. «Вне скоупа»): это страницы UI, а из терминала их и так видно через curl.

### 5. `bench run` и `index build`

**Файл уходит содержимым.** CLI читает файл как UTF-8 и кладёт текст в поле `scenario`
или `profile`, как требует рамка 38. YAML на клиенте не разбирается: ошибки синтаксиса и
схемы приходят от сервера как 422 `config.*` с `path`, и печатаются они одним и тем же
путём.

**Ссылка на библиотеку.** Перед постановкой CLI делает `GET /v1/scenarios/<имя файла без
.yaml>` (у сборки — `/v1/index-profiles/…`). Ссылка `scenario_id` (или `profile_id`) идёт в
запрос, только если ответ 200, сценарий в библиотеке `valid` и его `yaml` совпадает с
текстом файла байт в байт. Тогда для прогонов из терминала тоже работают `last_run` в
библиотеке UI и пометка «сценарий изменился после прогона». Совпадение текста означает,
что запущен именно файл библиотеки. Во всех остальных случаях ссылки нет, как у прогона
из временной копии; 404, 422 на недопустимое имя файла и любое расхождение ошибкой не
считаются. Цена — один лёгкий GET.

**Dry-run** печатает `RunCheck` в формате нынешнего `_echo_preflight`: все строки на
месте. Данные берутся из полей `dataset` и `index` ответа; `index` — те же метаданные
ragglet primary-цели, что у `Preflight.index`. Добавляются строка `plan` и по строке на
каждую цель (`targets`). Предупреждения печатаются после отчёта, как сейчас: `Preflight
warnings`.

`index build --dry-run` печатает `BuildCheck`. Строки `dataset: …, documents: N`,
`corpus sha256: …` и `will recreate: <collection> at <endpoint> (target <id>)` остаются
как есть. После `;` к ним дописывается то, что API знает о цели: `exists, 3633 points,
built …, used by: nfcorpus_hf` или `unreachable: <error>`. Дамп профиля
(`rprint(cfg.model_dump())`) уходит: CLI не разбирает YAML, а файл и так перед глазами.

**422.** Тело `Problem` печатается через `render_issues(title, issues, hint)` в stderr, код
выхода 2. Колонка «где» — это `where or path`. У проблем ядра `where` заполнен, и вывод
совпадает с нынешним `str(ValidationError)` символ в символ. У ошибок схемы `where`
пустой, и там стоит путь поля: `chunking.chunk_overlap`.

**202.** Предупреждения постановки (`RunAccepted.warnings`) печатаются сразу: если индекс
ещё `yellow`, исследователь узнаёт об этом до прогона, а не после. Дальше идёт ожидание
(п. 6). С `--no-wait` CLI печатает в stdout только id, чтобы его можно было подставить в
`id=$(…)`, а в stderr — `Queued: … Follow: ragglet runs wait <id>`. Код 0.

### 6. Ожидание задачи

Одна функция ожидания обслуживает `bench run`, `index build`, `runs wait` и `builds wait`:

```python
def watch(api: Api, kind: str, task_id: str) -> int:        # код выхода
    while True:
        status = api.get(f"/v1/{kind}/{task_id}")              # лёгкий ответ без include
        show(status)                                           # строки смены состояния, полоса
        if status["state"] in TERMINAL:
            return finish(api, kind, status)
        sleep(1.0)                                             # подменяется в тестах
```

- **Опрос раз в секунду**, как у UI (ragglet-28 п. 9). Ответ без `include` лёгкий, а
  сервер — отдельный процесс, не тот, что делает замеры, так что опрос замеры не трогает.
  Стриминга в API нет (SSE не входит в скоуп ragglet-28).
- **Что видно.** Каждая смена состояния или фазы — отдельная строка в stderr: `Queued: …
  (position 1; running now: …)` (что выполняется сейчас, CLI один раз берёт из
  `GET /v1/tasks`), затем `Started: preparing (preflight, loading models)...`, затем
  `Finishing (writing summary)...`. В фазе `progress` рисуется полоса
  `queries [#####-----] 120/323  0:41  ~1:08 left`. Остаток CLI считает по
  `progress.first_at` и `updated_at` с сервера, как и UI. Полоса перерисуется через `\r`
  после каждого опроса и только если stderr — терминал. В файле и в CI её нет, а строки
  смены фаз есть. Фазу считает сервер (поле `phase`), поэтому CLI и UI показывают одно и
  то же.
- **`completed`.** CLI делает ещё один `GET …?include=summary`. У прогона он печатает
  `Done`, затем `summary.aggregate` через `pprint`: это тот же словарь, который сейчас
  печатает `rprint(res.summary)`, только NaN из JSON приходят как `None`. Дальше идут
  `Run: <id>` и `Artifacts (on the server): <run_dir>` (п. 11). У сборки печатается
  `Done: 3633 documents`, а затем по строке на цель: `nfcorpus_v1 at http://localhost:6333`.
- **`failed`.** Если заполнено `error.problem` (preflight №2 при старте не прошёл,
  например потому что индекс пересобрали между постановкой и стартом), CLI печатает его
  через `render_issues` и выходит с кодом 2, как и воркер. Иначе он печатает
  `Failed: <type>: <message> (qid=<qid>)` и последние 20 строк лога. Лог читается через
  `…/log?offset=` от нуля до конца, это несколько запросов по 64 КиБ. Последней строкой
  идёт подсказка `Full log: ragglet runs log <id>`, код выхода 1.
- **`cancelled`** (например, задачу отменили из UI): `Cancelled at queries 120/323` по
  полю `progress`, код 1.

### 7. Ctrl-C во время ожидания отменяет задачу

Решено на ревью.

1. CLI ловит `KeyboardInterrupt`, отправляет `POST …/cancel` и печатает
   `Cancelling <id>... (Ctrl-C again to stop waiting)`.
2. CLI ждёт конечного состояния. Задача из очереди уходит сразу. Идущей задаче менеджер
   шлёт SIGTERM, а через `cancel_grace_s` (10 с) — SIGKILL. В конце CLI печатает
   `Cancelled at queries 120/323` (или `completed`, если задача успела закончиться).
3. Второй Ctrl-C прекращает ожидание сразу: отмена к этому моменту уже отправлена.
4. **Процесс завершается сигналом SIGINT**, а не `exit(130)`: CLI вызывает
   `signal(SIGINT, SIG_DFL)`, затем `os.kill(getpid(), SIGINT)`. Это проверено: если
   дочерний процесс поймал SIGINT и вышел через `exit(130)`, цикл `for` в bash переходит к
   следующей итерации, а если процесс умер от SIGINT, цикл останавливается. Нынешний CLI
   через click печатает `Aborted!` и выходит с кодом 1. Поэтому в цикле из
   `docs/article-experiments-audit.md` Ctrl-C сейчас прерывает один прогон и тут же
   запускает следующий. На Windows CLI выходит с кодом 130. В тестах эта функция
   подменяется.
5. Ctrl-C до ответа 202, пока идёт синхронный preflight, обрывает запрос. Сервер свой
   обработчик всё равно доведёт до конца и может поставить задачу. Поэтому CLI печатает
   `The server may have queued it: check ragglet tasks` и тоже выходит сигналом.

Отвязаться от задачи можно явно: `--no-wait`, потом `runs wait <id>`. Закрытие терминала
или обрыв SSH (SIGHUP) задачу не отменяют: CLI просто умирает, а задача идёт дальше.
Отменённый прогон не пропадает: он остаётся в реестре с итоговым прогрессом и с per-query
законченных повторов (ragglet-29 п. 5). `runs wait` и `builds wait` ведут себя так же,
потому что это то же ожидание. В `runs log --follow` Ctrl-C только прекращает чтение лога
(код 0): задачу эта команда не трогает.

### 8. Команды просмотра

- **`runs list`** — таблица из `RunList`: `id`, `state`, `duration`, `ndcg@k`, `recall@k`,
  `p50 ms`, `p95 ms`, `qps`. Вместо `null` стоит `-`, под таблицей — `N of total`. Опции
  совпадают по имени с параметрами запроса: `--state`, `--scenario`, `--tag`, `--sort`
  (например, `-ndcg`), `--limit` (по умолчанию 20). Остальные фильтры ragglet-30
  (`dataset`, `collection`, `q`, `since`, `k`) доступны через curl, а добавить любой из
  них в CLI — одна строка.
- **`builds list`** — `id`, `state`, `duration`, `dataset`, цели (`collection`),
  `documents`.
- **`runs show ID`** — карточка `RunStatus` строками `ключ: значение`: id и `origin`;
  сценарий и ссылка на библиотеку (`scenario_file`: `unchanged`, `changed` или `deleted`);
  состояние с фазой и прогрессом или позиция в очереди; времена и длительность; датасет,
  коллекция, теги; путь к артефактам; ошибка (как в п. 6) и предупреждения постановки. У
  `completed` ниже идёт summary, как в `bench run`. `builds show` показывает то же для
  `BuildStatus`, с полем `result`.
- **`runs cancel ID`** — `POST …/cancel`. Задача из очереди сразу становится `cancelled`,
  идущую CLI ждёт до конечного состояния, как в п. 7. На 409 `task.finished` CLI печатает
  `already completed` и выходит с кодом 1.
- **`runs log ID`** печатает лог от начала до текущего конца. С `--follow` CLI раз в
  секунду дочитывает лог с `next_offset`, пока `complete` не станет `true`.
- **`tasks`** — таблица `TaskItem`: вид, id, имя, состояние, позиция, прогресс
  `stage done/total`. Если очередь пуста, CLI печатает `No active tasks`.
- **`stand`** — `StandStatus` строками: каждый Qdrant (`up` или `down`, версия, число
  коллекций или ошибка), Redis, `compute` (устройство и число потоков torch), коммит и
  Python сервера, переменные `env` (`set` или `not set`: значений API не отдаёт), текущая
  задача и сколько задач ждёт.

### 9. Коды выхода, потоки вывода, `--json`

| Код | Когда |
|---|---|
| 0 | dry-run прошёл; задача `completed`; `--no-wait` принят; сервер запущен, остановлен или отвечает; команда просмотра выполнилась |
| 1 | задача `failed` (кроме preflight №2) или `cancelled`; ответ API 404, 409 или 500 — CLI печатает `Problem`; сервер не удалось запустить или остановить |
| 2 | 422: проблемы конфига, датасета, индекса; задача `failed` с `error.problem` |
| 3 | сервер недоступен, связь потеряна во время ожидания или по адресу отвечает не ragglet |
| SIGINT (`$?` = 130 в bash) | Ctrl-C во время ожидания: задача отменена (п. 7) |

Коды 0 и 2 совпадают с нынешними. Код 1 сейчас означает любой трейсбек.

В stdout идёт результат: отчёт dry-run, summary, таблицы, id при `--no-wait`, JSON. В
stderr идёт ход работы: строки фаз, полоса, ошибки и подсказки. Поэтому `| jq` и
`id=$(…)` получают чистый вывод.

**`--json`** есть у команд, которые печатают объект API: `bench run`, `index build`,
`runs list|show`, `builds list|show`, `tasks`, `stand`. Вместо текста они печатают тело
ответа как есть. У `bench run` без `--no-wait` это итоговый `RunStatus` с `summary`, полосы
при этом нет; у dry-run — `RunCheck`. При ошибке печатается тело `Problem`, коды выхода
те же. В stdlib-клиенте каждый такой флаг стоит одной строки.

### 10. Что удаляется

- **`src/ragglet/cli/` целиком**, вместе с typer-CLI и импортами ядра (`run_scenario`,
  `build_index`, `Preflight`, `load_scenario`, `load_index_profile` и т.д.). Команда
  `serve` переезжает в `src/ragglet/server.py` (п. 3). Пакет сервера теряет
  `[project.scripts] ragglet`: имя `ragglet` теперь у клиента. Из зависимостей уходят
  `typer` и `rich` — их использует только CLI (проверено поиском). Ядро по-прежнему можно
  вызвать без сервера, как и разрешает рамка: из Python, а задачу —
  `uv run python -m ragglet.jobs.worker results/runs/<id>` для отладки (ragglet-29).
  Отдельного «локального» пути в CLI нет.
- **`dataset import`** — решено на ревью: это рудимент старого формата корпуса
  (каталог `.md`/`.txt` и legacy-манифест запросов). `datasets/demo` уже сконвертирован,
  других пользователей у команды нет. Удаляются `src/ragglet/datasets/import_dir.py` и
  `test_import_directory_roundtrip` в `tests/test_beir_dataset.py`, а из README и
  `docs/architecure-and-current-state.md` уходят упоминания. Док ragglet-8 остаётся как
  история.
- **Тесты.** CLI используют четыре теста: `test_cli_dry_run_prints_preflight_report`,
  `test_cli_validation_error_exits_with_code_2`, `test_cli_unsupported_strategy_exits_with_code_2`
  в `test_preflight.py` и `test_cli_dry_run_lists_collections_to_recreate` в
  `test_build_index.py`. Они переезжают в `tests/test_cli.py` и идут уже через API.
  Проверки ядра под ними остаются в тестах ядра и API. Тест
  `test_serve_command_passes_flags_to_uvicorn` в `test_api_stand.py` переходит на
  `ragglet.server.main(["--port", "8123"])`. Остальные тесты ядра и API от удалённого не
  зависят.

### 11. Путь каталога в ответах API (в этом PR)

Решено на ревью: поле добавляется здесь.

```python
class RunStatus(_Live):
    run_dir: str          # str(record.run_dir): путь, как его видит сервер
class BuildStatus(_Live):
    build_dir: str
class RunItem(BaseModel):
    run_dir: str
class BuildItem(BaseModel):
    build_dir: str
```

По этому пути исследователь открывает per-query в pandas, а анализатор статьи читает
прогоны (п. 12). С настройками по умолчанию это относительный путь `results/runs/<id>` от
рабочего каталога сервера. Он верен и локально, и в Docker, если `results/` смонтирован
тем же путём. Данные уже лежат в записи реестра, так что изменение сводится к четырём
полям и строке в каждом из `run_status`, `run_item`, `build_status`, `build_item`. Тест
добавляется в `test_api_registry.py`. Те же функции в `api/routers/runs.py` меняет
открытый PR ragglet-35 (#16), но в других строках: кто вольётся вторым, тот и
разрешает конфликт.

### 12. Серия статьи — по тегу

Решено на ревью. Сценарии статьи раскладывали прогоны по каталогам
`results/article/raw` и `results/article/smoke` через `run.artifacts_dir`. Сервер это поле
игнорирует, поэтому теперь прогоны серии отличаются тегом:

```bash
SERIES=article-2026-11
ragglet bench run configs/experiments/article/adaptive/smoke_top_n.yaml --tag "$SERIES"
…
for trial in 1 2 3 4 5; do
  ragglet bench run configs/experiments/article/cache/b0_disabled.yaml --tag "$SERIES"
done
uv run python scripts/analyze_article_runs.py analyze --tag "$SERIES" --output-dir results/article
```

- **Анализатор.** `analyze --tag T [--api URL] [--exclude ID]…` вместо `--runs`. Он
  запрашивает `GET /v1/runs?tag=T&state=completed&limit=500`, группирует строки по
  `scenario` и читает файлы из `run_dir`. Проверки числа прогонов (`EXPECTED_RUNS`) не
  меняются. Smoke-прогоны берутся с тем же тегом, по сценариям `article_smoke_*`.
  Каталог `<runs>/../rejected` заменяет `--exclude`: исключённые прогоны перечисляются в
  аномалиях отчёта по id. Анализатор запускают из каталога стенда, потому что `run_dir`
  отсчитывается от него. HTTP — через `httpx`, он уже есть в окружении проекта, где
  анализатор и так работает (pandas). `check-smoke` по-прежнему принимает явные каталоги,
  а CLI печатает их в строке `Artifacts`.
- **Почему тег, а не каталог.** Все прогоны лежат в одном реестре и видны в UI. Серия
  отделена от прежних попыток тех же сценариев. Анализатор выбирает прогоны тем же
  фильтром, что и UI.
- Из девяти конфигов `configs/experiments/article/**` удаляется строка `run.artifacts_dir`:
  она больше ни на что не влияет и только вводит в заблуждение. Теги самих сценариев
  (`article`, `smoke`, …) остаются.

## Что меняется для пользователя

- Клиент ставится один раз: ссылкой в PATH (macOS, Linux) или запуском через `py`
  (Windows). Дальше команда — `ragglet …`, без `uv run`. Для стенда нужен uv, но вызывает
  его `ragglet server start`.
- Перед `bench run` и `index build` нужно один раз выполнить `ragglet server start` в том
  же терминале. Без сервера команды выходят с кодом 3 и объясняют, что делать.
- Команды, аргументы и вывод `bench run` и `index build` прежние. Новое: `--no-wait`,
  `--tag`, `--json`, ожидание в очереди, строки фаз, предупреждения сразу после
  постановки. `index build` больше не печатает дамп профиля.
- Пути в конфиге разрешает сервер: `dataset.id: datasets/demo` считается от каталога, где
  запущен сервер, а не CLI. `run.artifacts_dir` сервер игнорирует: прогоны пишутся в
  `RAGGLET_ARTIFACTS_DIR` (`results/runs/<id>`). Имя каталога — id постановки
  `<name>_<UTC>[-N]`, а не время старта.
- Ошибки схемы конфига печатаются списком с путём поля и кодом 2, а не трейсбеком
  pydantic.
- Сбой прогона — одна строка с типом, сообщением и `qid`, за ней хвост лога. Полный
  трейсбек — `ragglet runs log <id>`.
- Ctrl-C отменяет задачу на сервере и останавливает цикл bash. Чтобы отвязаться от
  задачи, есть `--no-wait`.
- Прогоны из терминала попадают в реестр с `origin: api`, видны в UI и отменяются оттуда.
  Два терминала больше не запускают два прогона одновременно: второй ждёт в очереди.
- Команды `ragglet dataset import` больше нет.
- Серия статьи помечается `--tag`, и анализатор выбирает её по этому тегу.

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `cli/ragglet_cli.py`, `cli/pyproject.toml` (новые) | клиент на stdlib и пакет `ragglet-cli` для pip |
| `src/ragglet/cli/` | удаляется целиком |
| `src/ragglet/server.py` (новый) | `python -m ragglet.server`: бывший `ragglet serve` |
| `src/ragglet/api/schemas.py`, `api/routers/runs.py`, `api/routers/builds.py` | `run_dir` и `build_dir` в статусах и строках списков |
| `src/ragglet/datasets/import_dir.py` | удаляется |
| `pyproject.toml`, `uv.lock` | без `[project.scripts]`, `typer` и `rich`; pytest `pythonpath` включает `cli` |
| `.gitignore` | `.ragglet/` |
| `scripts/analyze_article_runs.py` | `--tag`, `--api`, `--exclude` вместо `--runs` и `rejected/` |
| `configs/experiments/article/**` | без `run.artifacts_dir` |
| `tests/test_cli.py` (новый), `test_preflight.py`, `test_build_index.py`, `test_beir_dataset.py`, `test_api_stand.py`, `test_api_registry.py` | тесты CLI через API; удалённые и перенесённые тесты; `run_dir` |
| `README.md`, `docs/article-experiments-audit.md`, `docs/designs/ragglet-38-api-surface.md`, `docs/architecure-and-current-state.md` | ниже |

Ядро, `jobs/` и `registry/` не меняются.

## Документация

- **README, «Как запустить».** Шаг «Подготовить окружение» превращается в «Поставить CLI»:
  ссылка в PATH или `py` на Windows; на машине стенда — uv. После шага с Qdrant и Redis
  появляется шаг `ragglet server start`, который печатает адрес. В шагах «Собрать индекс»
  и «Запустить benchmark» команды прежние, но без `uv run`. К ним добавляется: команда ждёт
  очередь, Ctrl-C отменяет задачу, есть `--no-wait`. В «Посмотреть результаты»: каталог
  `results/runs/<id>/`, где id — время постановки, и команды `ragglet runs list` /
  `runs show`. В «REST API» `uv run ragglet serve` заменяется на `ragglet server start`, а
  фраза «До ragglet-37 CLI вызывает ядро напрямую» уходит. Раздел про формат датасета
  теряет `dataset import`.
- **README, «Полезные команды»:** `server status` и `server stop`, `runs list`,
  `runs show`, `runs log --follow`, `runs cancel`, `tasks`, `stand`, `--api`.
- **README, «Частые проблемы».** Описывается `cannot reach the ragglet server` (код 3):
  запустить `ragglet server start` в каталоге стенда, проверить `--api` и
  `RAGGLET_API_URL`. Если сервер не стартует, смотреть `.ragglet/server.log`. Ещё два
  пункта: путь датасета считается от каталога сервера, а `run.artifacts_dir`
  игнорируется. Абзац про `docker compose --profile app` («рабочий путь — локальный CLI
  через `uv run`») меняется: рабочий путь теперь `ragglet server start` плюс CLI.
- **`docs/article-experiments-audit.md`.** В подготовку добавляется `ragglet server start`.
  Команды теряют `uv run` и получают `--tag "$SERIES"`. Анализатор вызывается с `--tag`.
  Фраза «пятью независимыми CLI-процессами» заменяется на «пятью задачами, каждая в свежем
  процессе воркера»: независимость сохраняется (ragglet-29 п. 1). Цикл cold/warm работает
  как раньше: `bench run` ждёт конца прогона, и очистка Redis идёт строго между прогонами.
- **Рамка ragglet-38**, решение 6 и строка таблицы про ragglet-37. Вместо «typer + httpx»
  там будет «один файл на stdlib». Вместо «сервер поднимается через `ragglet serve` или
  docker compose» — «через `ragglet server start` или docker compose».
- **`docs/architecure-and-current-state.md`:** список команд CLI.
- **Docker (ragglet-34), что нужно от compose.** Сервис запускается командой
  `python -m ragglet.server`, с `RAGGLET_HOST=0.0.0.0` и портом `127.0.0.1:8000:8000`, чтобы
  адрес CLI по умолчанию работал. `configs/`, `datasets/` и `results/` монтируются теми же
  относительными путями от рабочего каталога сервера, и тогда `datasets/demo` и
  `results/runs/<id>` значат одно и то же на хосте и в контейнере. Внутри контейнера
  `http://localhost:6333` из сценариев указывает на сам контейнер — это ragglet-34 решает
  отдельно. Позже `ragglet server start` может уметь поднимать compose.
- **ragglet-35 (PR #16).** `scripts/smoke_api.sh` запускает сервер через `uv run ragglet
  serve`, а шаг 6 «Прогон CLI мимо API» рассчитан на CLI, который вызывает ядро сам. После
  этой задачи сервер запускается `uv run python -m ragglet.server`, а прогон CLI идёт через
  API с `origin: api`. Если #16 вольётся раньше, скрипт поправит эта задача, иначе — 35.
  Раздел README от 35 этот PR не трогает.

## Тесты (критерий приёмки)

Тесты идут без живого сервера, Qdrant и torch. `make_stand` из `tests/api_support.py` уже
собирает `create_app` с `InMemoryJobManager`, `InMemoryRegistry` или `FileRegistry` и
фейковым Qdrant. В клиенте HTTP-вызов — одна функция `transport(method, url, body,
timeout) -> (status, bytes)`, по умолчанию на urllib. Тест подставляет вместо неё вызов
`stand.client.request(…)`, то есть `TestClient`, и CLI ходит в настоящие роутеры через
ASGI. `ragglet_cli.sleep` заменяется сценарием, который на каждом тике двигает заглушку
менеджера: `anyio.run(jobs.start, ref)`, `jobs.report(ref, "queries", 3, 10)`,
`jobs.append_log`, `anyio.run(jobs.finish, ref, state, failure)`. Тест вызывает
`ragglet_cli.main(argv)` и проверяет код выхода и вывод (`capsys`).

1. **Dry-run прогона**: код 0, `Preflight: OK`, `corpus sha256: <DEMO_SHA>`,
   `embedding: all-MiniLM-L6-v2@<MODEL_SHA>`, строка `plan`. Каталог прогона не создан.
2. **422**: `cascade` — код 2 и `retrieval.not_implemented`. Нет метаданных индекса —
   код 2 и `index.no_metadata`. Битый YAML — код 2 и `config.yaml` с `line 3, column 2`.
   Лишнее поле — код 2 и `config.extra_forbidden`, путь стоит в колонке «где».
   `render_issues` клиента совпадает с ядром на одном и том же наборе проблем.
3. **Dry-run сборки**: `will recreate: chunks_v1 at http://q:6333 (target default)` и
   `exists, 53 points`.
4. **202 и ожидание**: `queued` → `running` без прогресса → `report` 3/10 → 10/10 →
   `completed`. В выводе по порядку: `Queued`, `Started: preparing`, `Finishing`, `Done`,
   ключ aggregate, `Run: <id>`, `Artifacts (on the server): <run_dir>`. Для aggregate
   тест кладёт `summary.aggregate.json` в каталог прогона и использует `FileRegistry`. Код
   0, в записи реестра есть `tags` из `--tag`.
5. **Ссылка на библиотеку**: если запущен файл библиотеки стенда (копия в
   `tmp_path/scenarios`), в записи есть `scenario_id`; у изменённой копии — `None`.
6. **`failed`**: `RuntimeError` с `qid` и текстом лога — код 1, в выводе `Failed:
   RuntimeError`, `qid=` и последние строки лога. `ValidationError` при старте (`failure`
   с `title` и `issues`) — код 2 и таблица проблем. Отмена из UI — код 1.
7. **Ctrl-C**: `sleep` бросает `KeyboardInterrupt` на втором тике. Задача из очереди
   становится `cancelled`. Для `running` CLI отправил отмену (`cancel_requested`), а после
   `finish(cancelled)` печатает `Cancelled at`. Выход сигналом подменён, тест проверяет
   его вызов.
8. **`--no-wait`**: в stdout ровно id, код 0, задача стоит в очереди.
9. **Сервер недоступен**: настоящий urllib, `--api http://127.0.0.1:<свободный порт>` —
   код 3 и `ragglet server start` в сообщении. Ответ не-`Problem` (транспорт отдаёт HTML
   404) — тоже код 3. Порядок выбора адреса: `--api` важнее переменной, переменная важнее
   `.ragglet/server.json`.
10. **Просмотр**: `runs list` после двух постановок показывает оба id и их состояния,
    `--json` разбирается как `RunList`; `runs show` у `completed`; `runs cancel`
    завершённой задачи — код 1; `runs log` склеивает текст по смещениям; `tasks` — текущая
    задача и очередь; `stand` — `down` у Redis из `make_stand`.
11. **`server start | status | stop`** с подменённой командой запуска: вместо `uv run` и
    `ragglet.server` поднимается крошечный stdlib-сервер, который отвечает на
    `/v1/tasks`. `start` пишет `.ragglet/server.json` и печатает адрес, повторный `start` —
    `already running`. Процесс, упавший при старте, — код 1 и хвост лога. `stop` с идущей
    задачей без `--force` — код 1, с `--force` процесс завершён и `server.json` удалён. Вне
    каталога стенда — код 1.
12. **Клиент на stdlib**: исходник разбирается `ast.parse(…, feature_version=(3, 9))`, все
    импорты — из `sys.stdlib_module_names`, есть `from __future__ import annotations`.
13. **API**: `run_dir` и `build_dir` есть в статусах и строках списков и равны
    `str(record.run_dir)`.
14. **Анализатор**: выбор каталогов по тегу на заглушке ответа `RunList`, работа
    `--exclude`, ошибка при неверном числе прогонов.
15. Остальные тесты проходят, `ruff check src tests cli` чистый.

Ручная проверка на Mac: `ln -s` клиента, `ragglet server start` из чистого клона (первый
запуск ставит окружение), `index build --dry-run` и `bench run --dry-run` по `demo`,
`bench run` с Ctrl-C на середине, `runs list`, `runs log`, `server stop`. Плюс:
`bash -c 'for i in 1 2; do ragglet bench run …; done'` по Ctrl-C останавливается, а
клиент запускается системным `/usr/bin/python3` (3.9).

## Фазы (отдельными коммитами)

1. API: `run_dir` и `build_dir` и их тест.
2. Клиент: `cli/ragglet_cli.py` с `bench run`, `index build`, ожиданием и Ctrl-C;
   `ragglet.server`; удаление `src/ragglet/cli/`, `typer` и `rich`; перенос тестов CLI.
3. `server start | status | stop`, команды просмотра, `--json`.
4. Удаление `dataset import`.
5. Статья: теги в анализаторе, конфиги без `run.artifacts_dir`.
6. Документация: README, аудит статьи, рамка 38, `architecure-and-current-state.md`; при
   необходимости — `scripts/smoke_api.sh`.

## Вне скоупа

- UI — ragglet-31. Docker и compose, в том числе `server start` через compose, —
  ragglet-34 (что от них нужно, см. «Документация»).
- Сервер на Windows без Docker или WSL.
- Публикация `ragglet-cli` на PyPI и сборка бинарников.
- Авторизация и TLS: сервер слушает `127.0.0.1` (ragglet-28 п. 11). Адрес сервера
  задаётся, но профилей «стендов» и переключения между ними нет.
- Библиотека сценариев и профилей из CLI (`scenarios list/push`), список коллекций
  (`GET /v1/indexes`), per-query и деталь запроса, скачивание артефактов (`runs pull`),
  «Повторить», сравнение прогонов. Всё это есть в API, а в терминале пока не нужно.
- Повторные попытки при обрыве связи и переподключение к задаче после рестарта сервера.
- Совместимость CLI и сервера разных версий: оба лежат в одном репозитории, а
  расхождение полей видно как ошибка в выводе.

## Рассмотренные альтернативы

- **Лёгкий пакет на typer + httpx + rich**, установка через pipx или venv. Вывод
  красивее, но для входа нужна установка зависимостей. Голый `pip install` на
  Homebrew-Python упирается в PEP 668, а на Windows сначала нужен сам Python. Старт около
  0,3 с.
- **Один пакет с сервером, `.venv/bin/ragglet` ссылкой в PATH.** Модели ответов были бы
  общие, но CLI нельзя отделить от тяжёлого окружения стенда.
- **`uv run ragglet …` на каждый вызов**, как сейчас. Человеку приходится знать про uv,
  и каждый вызов идёт через обёртку.
- **Bash + curl + jq.** Python не нужен совсем, но на Windows это только WSL или Git
  Bash, нужен jq, а опрос, отмену и коды выхода не протестировать. Рамка 38 это уже
  отвергла.
- **Один бинарник** на Go или Rust, или собранный PyInstaller. На машине ничего не нужно,
  но это второй язык или сборка под каждую ОС в CI. Для одного исследователя избыточно.
- **Автозапуск сервера** при первой команде. Тогда после `bench run` оставался бы фоновый
  процесс, о котором человек не просил, а у сервера был бы неочевидный рабочий каталог,
  от которого зависят `datasets/…` и `results/`. Явный `server start` — одна команда, и
  она печатает адрес.
- **Сервер через `uv run` в фоне.** pid принадлежал бы обёртке, а на второй SIGTERM
  uvicorn выходит без штатной остановки (п. 3).
- **Падать в ядро, если сервер не запущен.** Запрещено решением 6 рамки: снова два входа в
  стенд.
- **Разбирать YAML на клиенте** ради ранних ошибок. Это дублирует проверку сервера и
  требует PyYAML, то есть зависимость.
- **Путь к файлу вместо содержимого.** Нарушает рамку: сервер может работать в
  контейнере.
- **`scenario_id` по имени файла без сверки текста.** Временная копия с тем же именем
  получила бы ссылку на чужой файл библиотеки, и UI показал бы «изменился» или «удалён».
- **Ctrl-C только прекращает ожидание.** Ненужный прогон продолжал бы занимать стенд,
  следующий запуск молча вставал бы за ним в очередь, а цикл из статьи начинал бы
  следующую итерацию поверх идущего прогона.
- **Серия статьи по каталогам через `RAGGLET_ARTIFACTS_DIR`** на время серии. Пришлось бы
  перезапускать сервер под каждую серию, и прогоны жили бы вне общего реестра.

## Решено на ревью

1. **CLI — самостоятельный компонент без `uv run`**: один файл на стандартной библиотеке
   Python. Порог входа — только Python (п. 1).
2. **Один терминал и один CLI**: сервер поднимает `ragglet server start`. Человеку не нужно
   знать про `uv run ragglet serve`, а чтобы открыть UI, хватит одной команды и ссылки
   (п. 3).
3. **Ctrl-C отменяет задачу** (п. 7).
4. **`run_dir` и `build_dir` добавляются в API в этом PR** (п. 11).
5. **Серия статьи различается тегом**, а не каталогом (п. 12).
6. **`dataset import` удаляется** как рудимент старого корпуса (п. 10).
