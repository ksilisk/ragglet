# ragglet-37 — CLI как клиент единого API

**Коротко.** По решению 6 рамки ragglet-38 CLI одним шагом перестаёт вызывать ядро и
становится клиентом REST API (ragglet-28). Локального режима нет. CLI — это один файл
`ragglet` в корне репозитория. Он использует только стандартную библиотеку и
запускается системным `python3` (3.9+) без venv и установки. Вся сложность живёт в
сервере, а файл — его фасад. `python3 ragglet start` при первом запуске сам готовит
окружение, поднимает Qdrant и Redis через Docker, запускает сервер в фоне и печатает адрес
UI. Пользователь не набирает `uv run` никогда.

Команды делятся на группы: стенд (`start`, `stop`, `status`), задачи (`run`, `build`,
`cancel`, `log`, `queue`), библиотека сценариев и профилей и результаты (`runs`,
`builds`, `indexes`). `run` и `build` ведут себя как прежние `bench run` и `index build`:
dry-run печатает прежний отчёт, ответ 422 — прежний список проблем с кодом 2. После 202
CLI ждёт задачу, рисует полосу прогресса и печатает summary, а Ctrl-C отменяет задачу на
сервере. Typer-CLI `src/ragglet/cli/` удаляется вместе с typer и rich. Сервер запускается
как `python -m ragglet.server`. В ответы API добавляются `run_dir` и `build_dir`, а прогоны
серии статьи отбираются по тегу.

## Пример, на котором всё объясняется

Новый человек клонирует репозиторий:

```bash
git clone … && cd ragglet
python3 ragglet start        # первый раз сам создаст окружение и поднимет Qdrant/Redis
# UI: http://127.0.0.1:8000
…
python3 ragglet stop
```

Всё, что не влезает в этот блок и в список команд ниже, CLI не нужно.

**Старт.**

```
$ python3 ragglet start
Qdrant (http://localhost:6333) and Redis (redis://localhost:6379/0) are down, starting them with Docker...
Preparing the environment (the first time takes a few minutes)...
Starting the server, log: results/server.log
Ragglet is running: http://127.0.0.1:8000 (API /v1, docs /v1/docs)
```

Повторный `start` при живом сервере ничего не ломает: он печатает адрес и состояние.

**Индекс и проверка.** `build` принимает путь к профилю или id профиля из библиотеки
сервера.

```
$ python3 ragglet build nfcorpus_hf
Started: nfcorpus_hf_20261125T101500Z
corpus [##############################----------] 2724/3633  3:12  ~1:04 left
...
Done: 3633 documents
  nfcorpus_v1 at http://localhost:6333
$ python3 ragglet run configs/scenarios/nfcorpus_hf.yaml --dry-run
Preflight: OK
scenario: nfcorpus_hf
dataset: hf:BeIR/nfcorpus@b5026a0e96e8+a451b3b26d3a
split: test, queries: 323
corpus sha256: 0b2e…
eval sha256: 51ab…
index: nfcorpus_hf, built 2026-11-25T10:21:10Z
embedding: all-MiniLM-L6-v2@c9745ed1d9f2…
chunking: recursive 800/120
plan: 323 queries x 1 repeat, load closed (concurrency 1)
target default: nfcorpus_v1 at http://localhost:6333, green, 3633 points
```

**Ошибка.** Сервер отвечает 422, и CLI печатает список проблем в формате серверного
`render_issues`, как раньше, с кодом выхода 2:

```
$ python3 ragglet build configs/indexes/broken.yaml
Index profile config is invalid (1 error)
  config.value_error  chunking.chunk_overlap  chunk_overlap must be less than chunk_size
$ echo $?
2
```

**Прогон.**

```
$ python3 ragglet run nfcorpus_hf
Started: nfcorpus_hf_20261125T101540Z
Preparing (preflight, loading models)...
queries [########################################] 323/323  1:41
Finishing (writing summary)...
Done
  repeats                    1
  doc_ndcg@10_mean           0.315
  doc_recall@10_mean         0.150
  lat_total_p50              152.5
  ...
Artifacts: results/runs/nfcorpus_hf_20261125T101540Z (on the server)
```

Если нажать Ctrl-C на середине, CLI отменит прогон на сервере (`Cancelled at queries
120/323`) и завершится сигналом SIGINT, поэтому цикл `for` в bash тоже остановится.

**Результаты.**

```
$ python3 ragglet runs list --scenario nfcorpus_hf -n 3
id                            state      duration  ndcg@10  recall@10  p50 ms  p95 ms   qps  timeouts
nfcorpus_hf_20261125T101540Z  completed  0:01:41     0.315      0.150   152.5   232.1  95.4      0.0%
nfcorpus_hf_20261124T200301Z  cancelled  0:00:58         -          -       -       -     -         -
nfcorpus_hf_20261024T180917Z  completed  -           0.315      0.150   150.9   229.8  96.0      0.0%
3 of 9
$ python3 ragglet runs queries nfcorpus_hf_20261125T101540Z --miss -n 3
```

## Решение

### 1. Один файл на стандартной библиотеке

- **Где.** Файл `ragglet` в корне репозитория, рядом с `pyproject.toml`. Ссылок в PATH,
  `pip install`, отдельных пакетов и venv для клиента нет.
- **Чем запускать.** Системным `python3` версии 3.9 и новее. 3.9.6 — это
  `/usr/bin/python3` на macOS (его ставят Xcode и Command Line Tools, проверено). На Windows
  — `py ragglet …`.
- **Импорты** — только stdlib: `argparse`, `urllib.request`, `json`, `subprocess`,
  `signal`, `socket`, `webbrowser`, `shutil`, `tempfile`, `zipfile`. Цвет выводится,
  только если вывод идёт в терминал. Таблицы — фиксированной ширины.
- **Почему так.** Порог входа — один Python. Старт занимает около 50 мс (замерено).
  Нынешний CLI стартует около 8 с, потому что импортирует torch.
- **Чем платим.** Вместо typer — argparse, вместо httpx — `urllib.request`. Таблицы и
  полоса прогресса свои: около 50 строк, полоса рисуется одной строкой через `\r`. Модели
  `api/schemas.py` клиенту недоступны (это пакет сервера), поэтому ответы API он читает
  как словари из JSON. Расхождение имён полей ловят тесты, которые ходят в настоящие
  роутеры. `render_issues` (20 строк) скопирован в клиент, тест сверяет вывод с
  `ragglet.core.issues.render_issues`.
- **Совместимость с 3.9 проверяют тесты:** `ast.parse(source, feature_version=(3, 9))`, а
  все импорты должны входить в `sys.stdlib_module_names`.
- **`--help`.** Без аргументов печатается `--help` с группами команд: стенд, задачи,
  сценарии и профили, результаты.
- **Имя файла — `ragglet` без расширения.** `python -m …` кладёт рабочий каталог первым в
  `sys.path`, поэтому `ragglet` в корне перекрыл бы пакет `ragglet` для
  `python -m ragglet.server` и `python -m ragglet.jobs.worker`, которые сервер и менеджер
  задач запускают из корня. Проверено: `'ragglet' is not a package`. Файл без расширения
  импортировать нельзя, а запускается он так же: `python3 ragglet start`, на Windows —
  `py ragglet start`. Ruff проверяет его через `extend-include` с
  `per-file-target-version = py39`. Тесты грузят его `SourceFileLoader`. Вариант с
  `ragglet` и `PYTHONSAFEPATH=1` для сервера и воркеров отвергнут: любой
  `python -m ragglet…` или `import ragglet` из корня (отладка воркера, ноутбуки, compose)
  молча получал бы клиент вместо пакета.
- **Рамка 38** в решении 6 говорит «клиент на typer + httpx» и «сервер поднимается через
  `ragglet serve`». Её текст уточняется в этом PR. Bash + curl по-прежнему отвергнут.

### 2. Стенд: `start`, `stop`, `status`

```
start [--port 8000] [--host 127.0.0.1] [--open] [--no-services] [--foreground]
stop [--services]
status
```

**`start`** делает по шагам следующее.

1. **Сервер уже отвечает** по адресу (п. 3) — печатает адрес и состояние, как `status`, и
   выходит с кодом 0.
2. **Сервисы.** Адреса Qdrant и Redis берутся из `RAGGLET_QDRANT_ENDPOINTS` и
   `RAGGLET_REDIS_URL`, а если их нет — из умолчаний сервера (`http://localhost:6333`,
   `redis://localhost:6379/0`). Доступность проверяется TCP-подключением с таймаутом 1 с.
   - Если сервис недоступен, адрес локальный и `docker` есть в PATH, CLI выполняет
     `docker compose -f docker/docker-compose.yaml --profile qdrant up -d <qdrant|redis>`
     для недостающих сервисов и ждёт их портов до 60 с.
   - Если Docker нет, CLI печатает эту команду и останавливается с кодом 1.
   - Нелокальные адреса CLI не поднимает, только предупреждает.
   - `--no-services` пропускает этот шаг.
3. **Окружение.**
   - Если `uv` есть в PATH, при каждом старте выполняется `uv sync --frozen`. Когда
     окружение уже совпадает с `uv.lock`, это занимает десятки миллисекунд, а после
     `git pull` с новыми зависимостями окружение догоняет lock и сервер не падает на
     импорте.
   - Если uv нет, `.venv` создаётся один раз: CLI ищет Python 3.12+ (`python3.13`,
     `python3.12`, затем `python3`, если у него подходящая версия), выполняет
     `<python> -m venv .venv` и `.venv/bin/pip install -e .` и предупреждает, что версии
     зависимостей не зафиксированы по `uv.lock`. Системный 3.9 для этого не подходит: у
     пакета `requires-python >= 3.12`. Если подходящего Python нет, CLI просит поставить
     uv (со ссылкой) или Python 3.12+ и выходит с кодом 1.
   - Пользователь видит только `Preparing the environment...`. Вывод uv или pip идёт в
     `results/env.log`, а при сбое CLI печатает его хвост.
4. **Сервер.** CLI запускает `.venv/bin/python -m ragglet.server --host H --port P` в
   фоне.
   - Процесс стартует в новой сессии (`start_new_session`), stdin закрыт, stdout и stderr
     дописываются в `results/server.log`. Поэтому ни Ctrl-C, ни закрытие терминала сервер
     не задевают.
   - Переменные `RAGGLET_*` передаются как есть.
   - Перед стартом сервера CLI собирает UI (ragglet-31): если `web/dist` нет или исходники
     `web/` новее, `npm ci` (только при новом `package-lock.json`) и `npm run build`, вывод — в
     `results/web.log`. Без Node.js 22+ или при ошибке сборки — предупреждение, сервер
     стартует без UI; `--no-ui` пропускает шаг; заданный `RAGGLET_UI_DIR` не трогается.
   - Если `RAGGLET_UI_DIR` не задан, а собранный UI `web/dist` есть, CLI передаёт
     `RAGGLET_UI_DIR=web/dist`. Тогда по адресу открывается UI.
   - pid, адрес и время старта записываются в `results/server.json`.
5. **Готовность.** CLI ждёт ответа `GET /v1/stand` до 120 с: это импорт FastAPI и ядра и
   скан реестра. Если процесс вышел раньше (например, порт занят), CLI печатает последние
   20 строк лога и выходит с кодом 1.
6. **Итог.** CLI печатает `Ragglet is running: http://127.0.0.1:8000 …`, а с `--open`
   открывает браузер (`webbrowser.open`).

Остальные флаги:

- `--foreground` выполняет шаги 2–3, а затем запускает сервер на переднем плане через
  `os.execv`, без pid-файла, так что Ctrl-C его останавливает. Это режим для разработки
  и для compose в ragglet-34.
- `--host 0.0.0.0` меняет только адрес, на котором слушает сервер. Сам CLI подключается
  к `127.0.0.1`.

Почему CLI не запускает сервер через `uv run` в фоне: pid принадлежал бы обёртке uv. А на
второй SIGTERM uvicorn выходит без штатной остановки, и менеджер задач не успевает
пометить задачи. Поэтому uv только готовит окружение, а сервер запускается интерпретатором
`.venv` напрямую.

**Платформа.** Сервер работает только на macOS и Linux: воркер задач держится на
POSIX-сигналах (`add_signal_handler`, ragglet-29). На Windows `start` отвечает, что
сервер запускается в Docker (ragglet-34) или WSL. Остальные команды на Windows работают с
сервером на другой машине.

**`stop`** берёт pid из `results/server.json`.

- Если на сервере идёт задача или в очереди есть ожидающие, CLI сначала спрашивает
  `A run is in progress (…). Stopping the server fails it. Stop anyway? [y/N]`. Без
  терминала он отказывается и предлагает сначала сделать `cancel`. Причина: остановка
  переводит текущую и ожидающие задачи в `failed` / `ServerShutdown` (ragglet-29 п. 8).
- Дальше CLI шлёт SIGTERM, ждёт выхода до 15 с (`cancel_grace_s` плюс запас), затем
  SIGKILL, и удаляет `server.json`.
- `--services` дополнительно выполняет `docker compose … --profile qdrant stop qdrant
  redis`.
- Если `server.json` нет, а по адресу кто-то отвечает, CLI сообщает, что этот сервер
  запущен не через `ragglet`, и выходит с кодом 1.

**`status`** печатает:

- сервер: адрес, pid и время работы;
- Qdrant по каждому endpoint: доступен или нет, версия, число коллекций;
- Redis: доступен или нет, версия;
- устройство, на котором считает модель, и коммит сервера;
- текущую задачу с прогрессом и длину очереди.

Источник — `GET /v1/stand`. Если сервер не отвечает, CLI печатает `server: not running
(python3 ragglet start)` и TCP-проверку сервисов без версий и выходит с кодом 3.

**`python -m ragglet.server`** — это прежний `ragglet serve` из `cli/app.py`, перенесённый
без изменения поведения в `src/ragglet/server.py`: флаги `--host` и `--port`, печать
настроек, uvicorn с одним worker. Эту же команду возьмёт compose в ragglet-34.

**Redis в compose.** Сейчас у сервиса `redis` в `docker/docker-compose.yaml` нет
`ports:`, и с хоста `redis://localhost:6379` недоступен, хотя README утверждает обратное.
Аудит статьи упоминает, что прогоны шли в Redis хоста. Добавляется
`ports: ["127.0.0.1:6379:6379"]`: без этого `start` поднял бы Redis, до которого сервер не
достучится.

### 3. Адрес сервера

Адрес берётся из первого источника, где он задан:

1. `--api URL`;
2. `RAGGLET_API_URL`;
3. `url` из `results/server.json` — его пишет `start`, поэтому `start --port 8100` и
   следующие команды сходятся;
4. `http://127.0.0.1:8000`.

Таймаут у urllib один на любую операцию с сокетом: 30 с. У `POST /v1/runs` и
`POST /v1/builds` таймаута нет: preflight синхронный, а первое обращение к `hf`-датасету
его скачивает.

Если сервер недоступен (отказ в соединении, таймаут подключения), CLI печатает
`cannot reach the ragglet server at …; start it: python3 ragglet start` и выходит с
кодом 3. Код 3 будет и в том случае, если по адресу отвечает не ragglet, то есть тело
ошибки не `Problem`. Если связь пропала во время ожидания задачи, CLI печатает
`lost connection …; check later: python3 ragglet runs show <id>`, тоже с кодом 3.

### 4. Задачи

```
run <сценарий.yaml | id> [--dry-run] [--tag T …] [--no-wait]
build <профиль.yaml | id> [--dry-run] [--no-wait]
cancel <id>
log <id> [--follow]
queue
```

- **Аргумент `run` и `build`.** Если это путь к существующему файлу, CLI читает файл как
  UTF-8 и шлёт текст в `POST /v1/runs` (`scenario`) или `/v1/builds` (`profile`), без
  ссылки на библиотеку. Иначе аргумент считается id библиотеки: CLI берёт YAML из
  `GET /v1/scenarios/{id}` (или `/v1/index-profiles/{id}`) и шлёт его вместе с
  `scenario_id` (`profile_id`). Так прогон привязан к сценарию библиотеки, и UI видит его в
  `last_run`. Если нет ни файла, ни id, сервер отвечает 404, CLI печатает
  `no such file or library scenario: X` и выходит с кодом 1. YAML клиент не разбирает:
  ошибки синтаксиса и схемы приходят от сервера как 422 `config.*`.
- **Dry-run.** `RunCheck` печатается в формате нынешнего `_echo_preflight`, все строки на
  месте, плюс строка `plan` и по строке на цель. `BuildCheck` — строки `dataset: …,
  documents: N`, `corpus sha256: …` и `will recreate: <collection> at <endpoint> (target
  <id>)`, а после `;` к ним дописывается `exists, 3633 points, built …, used by: …` или
  `unreachable: <error>`. Дамп профиля, который печатал прежний `index build`, уходит.
  Предупреждения печатаются после отчёта, как сейчас.
- **422.** Тело `Problem` печатается через `render_issues(title, issues, hint)` в stderr.
  Колонка «где» — это `where or path`. У проблем ядра `where` заполнен, и вывод совпадает с
  нынешним `str(ValidationError)` символ в символ. Код выхода 2.
- **202.** CLI сразу печатает предупреждения постановки: если индекс ещё `yellow`,
  исследователь узнаёт об этом до прогона. Дальше идёт ожидание (п. 7). С `--no-wait` CLI
  печатает в stdout только id, чтобы его можно было подставить в `id=$(…)`, и выходит с
  кодом 0. Вернуться к задаче можно через `log --follow` и `runs show`: команды `wait`
  нет.
- **`cancel <id>`.** Вид задачи (прогон или сборка) CLI берёт из `GET /v1/tasks`: отменить
  можно только активную задачу. Затем отправляет `POST /v1/{runs|builds}/{id}/cancel`.
  Задача из очереди отменяется сразу, идущую CLI ждёт до конечного состояния. Если задачи
  нет среди активных, CLI печатает `not active: <state>` и выходит с кодом 1.
- **`log <id> [--follow]`** печатает лог от начала: `GET /v1/runs/{id}/log?offset=`, а при
  404 — то же для `/v1/builds`. С `--follow` CLI раз в секунду дочитывает лог с
  `next_offset`, пока `complete` не станет `true`. Ctrl-C здесь только прекращает чтение,
  код 0.
- **`queue`** — `GET /v1/tasks`: текущая задача с прогрессом, затем очередь по порядку.

### 5. Сценарии и профили

```
scenarios list | show <id> | edit <id> | new <id> --from <id | run-id> | check <id> | delete <id>
profiles  list | show <id> | edit <id> | new <id> --from <id | build-id> | check <id> | delete <id>
```

Все команды работают с библиотекой сервера (`/v1/scenarios`, `/v1/index-profiles`) и
версией `etag` (ragglet-28 п. 4).

- **`list`** — таблица. У сценариев: `id`, `valid`, датасет, коллекции и последний
  прогон (состояние и время). У профилей: `id`, `valid`, датасет, chunking, модель, цели
  и последняя сборка.
- **`show <id>`** печатает YAML как есть, с комментариями. Если сценарий — черновик
  (`valid: false`), проблема уходит в stderr.
- **`edit <id>`** работает так:
  1. CLI делает GET и пишет текст во временный файл `<id>.yaml`.
  2. Открывает его в `$VISUAL` или `$EDITOR`; если их нет — `notepad` на Windows и `vi` на
     остальных системах.
  3. Если текст не изменился, печатает `no changes`.
  4. Иначе отправляет `PUT {yaml, etag}`. Ответ 200 — `saved`, а при `valid: false` —
     `saved as draft` и проблема.
  5. Ответ 409 — `changed on the server since you opened it`. Ответ 422 (битый YAML) —
     список проблем и код 2.

  В последних двух случаях CLI называет путь к временному файлу, чтобы правка не
  потерялась.
- **`new <id> --from <src>`.**
  - Если `src` — сценарий библиотеки, CLI берёт его YAML, заменяет строку верхнего уровня
    `name:` (у профиля — `id:`) на новый id регулярным выражением и шлёт `POST {yaml}`.
    Комментарии и порядок полей сохраняются.
  - Если `src` — id прогона, CLI шлёт `POST {from_run: src, name: id}`, и сервер берёт
    snapshot прогона.
  - У профиля `src` может быть id сборки: тогда текст берётся из
    `GET /v1/builds/{id}?include=profile`.
  - Если строки `name:` нет, CLI выходит с ошибкой и кодом 2.
- **`check <id>`** — dry-run сценария библиотеки, то же, что `run <id> --dry-run`.
- **`delete <id>`** спрашивает `Delete scenario <id>? [y/N]`, затем шлёт
  `DELETE …?etag=`. Ответ 409 означает, что файл изменился с момента чтения.

### 6. Результаты

```
runs list [--state S] [--scenario NAME] [--tag T] [-n 20] [-k 10]
runs show <id> | files <id> | export <id> [--zip] [-o DIR] | queries <id> [--outcome O] [--miss] [-n 50] [-k 10]
builds list | show <id>
indexes
```

- **`runs list`** — `GET /v1/runs` с теми же фильтрами, `limit` и `k`. Таблица:
  `id`, `state`, `duration`, `ndcg@k`, `recall@k`, `p50 ms`, `p95 ms`, `qps`, `timeouts`.
  Вместо `null` стоит `-`, под таблицей — `N of total`. Сценарий и время постановки
  читаются из id (`<сценарий>_<UTC>`). Отдельными колонками они бы дублировали id, а сам
  id нужен для `runs show` и остальных команд.
- **`runs show <id>`** — `GET /v1/runs/{id}?include=summary,manifest`, по разделам:
  - шапка: состояние, времена и длительность, теги, `run_dir`, ссылка на библиотеку и
    `scenario_file`;
  - условия: датасет с хэшами корпуса и eval, индекс (профиль, коллекция, `built_at`,
    модель и её ревизия), окружение (коммит, устройство, машина);
  - summary — тот же блок, что после `run`;
  - предупреждения постановки и ошибка.
- **Блок summary** — средние по повторам из `summary.aggregate`: ключи
  `*_mean_over_repeats` без суффикса, выровненными строками `ключ значение`, плюс
  `repeats`. Всё остальное доступно через `--json`.
- **`runs files <id>`** — `GET …/artifacts`: имя, размер, время изменения.
- **`runs export <id> [--zip] [-o DIR]`** скачивает `GET …/artifacts.zip`. С `--zip`
  архив сохраняется как `DIR/<id>.zip`. Без него архив распаковывается в `DIR/<id>/` с
  проверкой, что пути не выходят за `DIR`. По умолчанию `DIR` — текущий каталог.
- **`runs queries <id>`** — `GET …/queries` с `outcome`, `limit`, а `--miss` превращается в
  `miss_at=k`. Таблица: `qid`, `outcome`, `first_relevant_rank`, `lat_total_ms` и начало
  текста запроса. Колонок, которых нет в файле старого поколения, в таблице нет.
- **`builds list`** — `id`, `state`, `duration`, датасет, цели, `documents`. **`builds show
  <id>`** — карточка `BuildStatus` с `result` и `build_dir`.
- **`indexes`** — `GET /v1/indexes`: endpoint, коллекция, `state` (`ready`, `optimizing`,
  `incomplete`, `foreign`), статус, `points`, профили и сценарии, которые на неё
  ссылаются. Недоступный endpoint выводится отдельной строкой.

### 7. Ожидание и Ctrl-C

- **Опрос** `GET /v1/{runs|builds}/{id}` идёт раз в секунду, как в UI. Ответ без `include`
  лёгкий. Стриминга в API нет.
- **Строки фаз** (в stderr):
  - `Queued: <id> (position N; running now: …)` — что выполняется сейчас, CLI один раз
    берёт из `/v1/tasks`;
  - `Started: <id>`;
  - `Preparing (preflight, loading models)...`;
  - в фазе `progress` — полоса `queries [#####-----] 120/323  0:41  ~1:08 left`;
  - `Finishing (writing summary)...`.
- **Полоса** перерисовывается через `\r` после каждого опроса и только если stderr —
  терминал. Остаток CLI считает по `first_at` и `updated_at` с сервера. Фазу считает
  сервер, поэтому CLI и UI показывают одно и то же.
- **`completed`.** CLI делает ещё один GET с `include=summary`. У прогона печатаются
  `Done`, блок summary и `Artifacts: <run_dir> (on the server)`. У сборки — `Done: N
  documents` и цели.
- **`failed` с `error.problem`** (preflight №2 при старте не прошёл) — `render_issues` и
  код 2.
- **Другой `failed`** — `Failed: <type>: <message> (qid=…)`, последние 20 строк лога (его
  CLI читает через `log?offset=`) и подсказка `full log: python3 ragglet log <id>`. Код
  1.
- **`cancelled`** — `Cancelled at queries 120/323`, код 1.

**Ctrl-C отменяет задачу** (решено на ревью).

1. CLI отправляет `POST …/cancel` и печатает `Cancelling <id>... (Ctrl-C again to stop
   waiting)`.
2. CLI ждёт конечного состояния. Второй Ctrl-C прекращает ожидание сразу.
3. CLI завершается сигналом: `signal(SIGINT, SIG_DFL)`, затем `os.kill(getpid(), SIGINT)`.
   Проверено: если дочерний процесс поймал SIGINT и вышел через `exit(130)`, цикл `for` в
   bash переходит к следующей итерации, а если процесс умер от SIGINT — цикл
   останавливается. На Windows CLI выходит с кодом 130.
4. Ctrl-C до ответа 202 обрывает запрос, но сервер всё равно может поставить задачу. CLI
   говорит об этом и советует проверить `queue`.

Закрытие терминала (SIGHUP) задачу не отменяет. Отменённый прогон остаётся в реестре с
итоговым прогрессом.

### 8. Глобальные флаги, коды выхода, вывод

- **`--json`** есть у любой команды. CLI печатает сырой JSON ответа: у `run` с ожиданием —
  итоговый статус с `summary`, у `status` и `start` — `/v1/stand`. При ошибке печатается
  тело `Problem`. Полосы и таблиц при этом нет.
- **`--api URL`** и `RAGGLET_API_URL` задают адрес (п. 3). Оба флага принимаются в любом
  месте командной строки (общий родительский парсер argparse).
- **Потоки.** В stdout идёт результат: отчёты, таблицы, YAML, id, JSON. В stderr — ход
  работы, ошибки и подсказки.

| Код | Когда |
|---|---|
| 0 | готово |
| 1 | задача упала или отменена; 404, 409, 500 от API; не удалось поднять или остановить стенд |
| 2 | проблемы конфига, датасета или индекса (422, `failed` с `error.problem`) |
| 3 | сервер не запущен или недоступен; подсказка `python3 ragglet start` |
| SIGINT (`$?` = 130 в bash) | Ctrl-C во время ожидания: задача отменена |

### 9. Что уходит

- **`src/ragglet/cli/` целиком**, вместе с typer и rich в `pyproject.toml`. Их
  использует только CLI, это проверено поиском. Уходит и `[project.scripts] ragglet`, а
  `uv.lock` пересобирается.
- **`serve`** переезжает в `src/ragglet/server.py` (п. 2).
- **Команда `dataset import`** уходит. `datasets/import_dir.py` и его тест остаются как
  библиотека.
- **Тесты CLI.** Четыре теста в `test_preflight.py` и `test_build_index.py`, которые
  вызывают CLI, переезжают в `tests/test_cli.py` и идут через API.
  `test_serve_command_passes_flags_to_uvicorn` переходит на
  `ragglet.server.main(["--port", "8123"])`.
- **Ядро** по-прежнему вызывается без сервера, как и разрешает рамка: из Python и через
  `python -m ragglet.jobs.worker <каталог задачи>` для отладки.

### 10. `run_dir` и `build_dir` в API

Решено на ревью, сделано первым коммитом. В `RunStatus` и строке `RunItem` появилось поле
`run_dir`, в `BuildStatus` и `BuildItem` — `build_dir`. Значение — `str(record.run_dir)`,
путь, как его видит сервер: с настройками по умолчанию `results/runs/<id>` от рабочего
каталога сервера. Этот путь печатает строка `Artifacts`, и по нему анализатор статьи
читает прогоны.

### 11. Серия статьи — по тегу

Решено на ревью. Сервер игнорирует `run.artifacts_dir`, поэтому серия статьи отличается
тегом, а не каталогом:

```bash
SERIES=article-2026-11
python3 ragglet run configs/experiments/article/adaptive/smoke_top_n.yaml --tag "$SERIES"
for trial in 1 2 3 4 5; do
  python3 ragglet run configs/experiments/article/cache/b0_disabled.yaml --tag "$SERIES"
done
uv run python scripts/analyze_article_runs.py analyze --tag "$SERIES" --output-dir results/article
```

- **Анализатор.**
  - Флаг `--tag T [--api URL] [--exclude ID]…` заменяет `--runs`.
  - Анализатор запрашивает `GET /v1/runs?tag=T&state=completed&limit=500`, группирует
    строки по `scenario` и читает файлы из `run_dir`. Smoke-прогоны берутся с тем же
    тегом, по сценариям `article_smoke_*`.
  - `--exclude` заменяет каталог `rejected/`: исключённые прогоны перечисляются в отчёте
    по id.
  - Проверки числа прогонов не меняются. `check-smoke` по-прежнему принимает каталоги.
  - HTTP — через `urllib`, как остальные сетевые вызовы анализатора.
- Строка `run.artifacts_dir` уходит из девяти конфигов `configs/experiments/article/**`.
- Цикл cold/warm работает, как раньше: `run` ждёт конца прогона, и очистка Redis идёт
  между прогонами.

## Что меняется для пользователя

- Вместо `uv run ragglet …` теперь `python3 ragglet …`, а стенд поднимается одной
  командой `start`.
- `bench run X` становится `run X`, `index build X` — `build X`. Вывод, коды 0 и 2 и
  dry-run прежние. Новое:
  - аргументом можно передать id из библиотеки;
  - `--no-wait` и `--tag`;
  - ожидание в очереди и строки фаз;
  - предупреждения печатаются сразу после постановки.
- Пути в конфиге разрешает сервер: `dataset.id` считается от его рабочего каталога (корня
  репозитория), а `run.artifacts_dir` игнорируется. Каталог прогона называется по id
  постановки: `results/runs/<name>_<UTC>[-N]`.
- Ошибки схемы конфига печатаются списком с путём поля и кодом 2, а не трейсбеком
  pydantic.
- Сбой прогона — одна строка и хвост лога. Полный лог выдаёт `log <id>`.
- Ctrl-C отменяет задачу на сервере и останавливает цикл bash.
- Прогоны из терминала видны в UI и встают в общую очередь.
- Команды `dataset import` больше нет.

## Затрагиваемые модули

| Модуль | Изменение |
|---|---|
| `ragglet` (новый) | клиент на stdlib |
| `src/ragglet/cli/` | удаляется |
| `src/ragglet/server.py` (новый) | `python -m ragglet.server` |
| `src/ragglet/api/schemas.py`, `routers/runs.py`, `routers/builds.py` | `run_dir` и `build_dir` (сделано) |
| `pyproject.toml`, `uv.lock` | без `typer`, `rich` и `[project.scripts]` |
| `docker/docker-compose.yaml` | порт Redis `127.0.0.1:6379` |
| `scripts/analyze_article_runs.py`, `configs/experiments/article/**` | `--tag` вместо каталогов |
| `tests/test_e2e_api.py` (ragglet-35) | сервер через `python -m ragglet.server`; тест «прогон CLI мимо API импортируется» заменён на `start` → `run` → `stop` клиента против настоящего сервера (`origin: api`) |
| `tests/test_cli.py` (новый), `test_preflight.py`, `test_build_index.py`, `test_api_stand.py` | тесты CLI, переносы |
| `README.md`, `docs/article-experiments-audit.md`, `docs/designs/ragglet-38-api-surface.md`, `docs/architecure-and-current-state.md` | под `python3 ragglet …` |

## Документация

- **README.**
  - «Как запустить» — блок из примера: clone, `start`, ссылка, `stop`. Шаг «поднять Qdrant
    и Redis» становится частью `start` и остаётся только как пояснение.
  - «Полезные команды» — все группы команд.
  - «Частые проблемы»:
    - сервер не запущен (код 3);
    - нет Docker (команда для ручного запуска сервисов);
    - нет uv и Python 3.12+;
    - сервер не стартует — смотреть `results/server.log`;
    - путь датасета считается от корня, а `run.artifacts_dir` игнорируется.
  - Для разработчиков остаются `uv sync --extra dev`, `uv run pytest` и `start
    --foreground`.
- **`docs/article-experiments-audit.md`** — команды `python3 ragglet run … --tag
  "$SERIES"` и анализатор с `--tag`.
- **Рамка 38**, решение 6 — «один файл на stdlib, сервер через `ragglet start`».
- **`architecure-and-current-state.md`** — список команд.
- **ragglet-31.** Док UI пишет `RAGGLET_UI_DIR=web/dist uv run ragglet serve`. После этой
  задачи достаточно `python3 ragglet start`: `start` сам подставит `web/dist`.
- **ragglet-34.** Compose запускает `python -m ragglet.server` (или `ragglet start
  --foreground`) с `RAGGLET_HOST=0.0.0.0`. `configs/`, `datasets/` и `results/`
  монтируются теми же путями от рабочего каталога.

## Тесты (критерий приёмки)

**Как устроены.** `tests/test_cli.py` загружает `ragglet` через
`importlib.util.spec_from_file_location` и вызывает `main(argv)`, а коды выхода и вывод
проверяет через `capsys`. Сервер — `create_app` из `make_stand` (`InMemoryJobManager`,
`InMemoryRegistry` или `FileRegistry`, фейковый Qdrant), поднятый uvicorn в потоке на
свободном порту. Так клиент ходит по настоящему HTTP через свой urllib. Функция `sleep`
клиента подменяется сценарием, который двигает заглушку менеджера на каждом тике.

1. `run --dry-run` — отчёт и код 0, каталог прогона не создан. По id из библиотеки в
   запросе есть `scenario_id`.
2. 422: `cascade`, отсутствие метаданных индекса, битый YAML (`config.yaml` с `line 3,
   column 2`), лишнее поле — код 2. Вывод `render_issues` клиента совпадает с ядром.
3. `build --dry-run` — `will recreate: chunks_v1 at http://q:6333 (target default)`.
4. Ожидание: `queued` → `preparing` → `progress` → `finishing` → `completed`. В выводе
   строки фаз, блок summary, `Artifacts: <run_dir>`, код 0, теги из `--tag` в записи.
5. `failed` с `qid` — код 1 и хвост лога; `failed` с `problem` — код 2; `cancelled` — код
   1.
6. Ctrl-C (`sleep` бросает `KeyboardInterrupt`) — задача отменена, выход сигналом
   (функция подменена, тест проверяет её вызов). `--no-wait` — в stdout ровно id.
7. Сервер недоступен — код 3 и подсказка. Ответ не-`Problem` — код 3. Порядок выбора
   адреса: `--api`, затем переменная, затем `server.json`.
8. `cancel`, `log` по смещениям и с `--follow`, `queue`.
9. Библиотека:
   - `list`, `show` (текст байт в байт);
   - round-trip `edit`: подменённый `$EDITOR` — python-скрипт, правящий файл; `PUT` с
     `etag` сохраняет;
   - `edit` после правки файла на диске — 409 и путь к временному файлу;
   - `new --from` сценария (name заменён, комментарии на месте) и из прогона;
   - `check`;
   - `delete` с подтверждением через подменённый `input`.
10. Результаты:
    - `runs list` и `--json`;
    - `runs show` у `completed` (`FileRegistry`, каталог из `registry_support`);
    - `runs files`;
    - `runs export` с `--zip` и с распаковкой;
    - `runs queries --miss`;
    - `builds list` и `builds show`;
    - `indexes`.
11. `start`, `status`, `stop` с подменами. Вместо `.venv/bin/python -m ragglet.server`
    поднимается крошечный stdlib-сервер с `/v1/stand` и `/v1/tasks`, проверка сервисов и
    подготовка окружения тоже подменены. Проверяется:
    - `start` пишет `server.json` и печатает адрес;
    - повторный `start` — `already running`;
    - упавший при старте процесс — код 1 и хвост лога;
    - нет Docker при недоступном Qdrant — команда compose в выводе и код 1;
    - `stop` с активной задачей без терминала отказывается;
    - `stop` после подтверждения завершает процесс и удаляет `server.json`.
12. Клиент на stdlib: `ast.parse(…, feature_version=(3, 9))` и импорты из
    `sys.stdlib_module_names`.
13. Интеграционный (`-m integration` в `test_e2e_api.py`, пропускается без Qdrant и Redis):
    `start --port <свободный> --no-services` с временными `RAGGLET_*` каталогами → `run`
    `baseline_qdrant` по собранной фикстурой коллекции → `runs list --tag` → `stop`.
14. Остальные тесты проходят, `uv run ruff check src tests ragglet` по изменённым
    файлам чистый.

Ручная проверка на Mac: чистый клон, `/usr/bin/python3 ragglet start` (3.9) со
скачиванием окружения, UI по ссылке, `build demo`, `run baseline_qdrant` с Ctrl-C,
`runs list`, `stop --services`. Ещё проверяется, что `for` в bash по Ctrl-C
останавливается.

## Фазы (отдельными коммитами)

1. API: `run_dir` и `build_dir` (сделано).
2. Клиент: `ragglet` с `run`, `build`, ожиданием, Ctrl-C, `cancel`, `log`, `queue`,
   `--json` и `--api`; `ragglet.server`; удаление `src/ragglet/cli/`, typer и rich, команды
   `dataset import`; перенос тестов.
3. Стенд: `start`, `stop`, `status`, порт Redis в compose.
4. Библиотека: `scenarios`, `profiles`.
5. Результаты: `runs`, `builds`, `indexes`.
6. Статья: анализатор по тегу, конфиги.
7. Документация.

## Вне скоупа

- UI — ragglet-31. Docker-образ сервера и compose целиком — ragglet-34.
- Сервер на Windows без Docker или WSL.
- Авторизация, TLS, несколько стендов.
- Деталь одного запроса (`GET /v1/runs/{id}/queries/{qid}`), сравнение прогонов,
  «Повторить» — это UI.
- Повторные попытки при обрыве связи.
- Совместимость клиента и сервера разных версий: они лежат в одном репозитории.

## Рассмотренные альтернативы

- **Лёгкий пакет на typer + httpx + rich.** Вывод красивее, но для входа нужна установка
  зависимостей. `pip install` на Homebrew-Python упирается в PEP 668.
- **Клиент со ссылкой в PATH, `pip install ./cli`, отдельный пакет.** Это лишние шаги
  установки. Файл в корне запускается сразу после `git clone`.
- **Bash + curl + jq.** На Windows работает только через WSL, нужен jq, а опрос, отмену и
  коды выхода не протестировать. Рамка 38 это уже отвергла.
- **Один бинарник** на Go или Rust, или собранный PyInstaller. Это второй язык или
  сборка под каждую ОС.
- **Автозапуск сервера из `run`.** Тогда оставался бы фоновый процесс, о котором не
  просили. `start` — одна явная команда, и она печатает адрес.
- **Сервер через `uv run` в фоне.** pid принадлежал бы обёртке, а на второй SIGTERM
  uvicorn выходит без штатной остановки.
- **Падать в ядро без сервера.** Запрещено рамкой: снова два входа в стенд.
- **Разбирать YAML на клиенте.** Нужен PyYAML, то есть зависимость, и получился бы второй
  формат ошибок.
- **Серия статьи по каталогам через `RAGGLET_ARTIFACTS_DIR`.** Пришлось бы
  перезапускать сервер под каждую серию, и прогоны жили бы вне общего реестра.

## Решено на ревью

1. **CLI — самостоятельный компонент без `uv run`:** один файл `ragglet` в корне,
   только stdlib, системный `python3` 3.9+, без установки. Вся сложность — в сервере (п.
   1).
2. **Один терминал и одна команда для стенда:** `start` сам готовит окружение, поднимает
   сервисы и сервер и печатает ссылку на UI (п. 2).
3. **Набор команд** — стенд, задачи, сценарии и профили, результаты. Всё, что не входит
   в README-блок и этот список, не нужно. Команды `wait` нет.
4. **Ctrl-C отменяет задачу** (п. 7).
5. **`run_dir` и `build_dir` в API** добавлены в этом PR (п. 10).
6. **Серия статьи различается тегом** (п. 11).
7. **`dataset import` уходит** как рудимент старого корпуса, `import_dir.py` остаётся
   библиотекой (п. 9).
8. **Имя файла — `ragglet` без расширения** (п. 1): `ragglet` перекрывал бы пакет
   `ragglet` для `python -m` из корня. Вопрос задан, реализация идёт на этом варианте:
   переименовать — один `git mv` и замена в документации.
