# ragglet-34 — Воспроизводимое Docker-окружение

**Коротко.** Стенд целиком (API-сервер с веб-интерфейсом, Qdrant и Redis) поднимается одной
командой `docker compose … --profile app up -d --build --wait`. Образ собирается из
`docker/Dockerfile` в две стадии: Node собирает `web/dist`, Python ставит зависимости
`uv sync --frozen` с CPU-сборкой torch. Сервер в контейнере — тот же
`python -m ragglet.server`, что поднимает `python3 ragglet start` на хосте. Работает с теми же
каталогами `configs/`, `datasets/`, `results/` (bind mount) и с тем же Qdrant. Сценарии в
`configs/` пишут адреса `http://localhost:6333` и `redis://localhost:6379/0`. Внутри
контейнера по этим адресам никого нет, поэтому сервер подменяет `host:port` в момент
подключения по таблице `RAGGLET_ENDPOINT_ALIASES`, которую задаёт compose. Конфиги и
snapshot прогонов при этом не меняются, а сама таблица пишется в `manifest.env`. Профиль
`app-multi` — то же для кластера из двух шардов Qdrant (`qdrant-multi`). Фасад `ragglet`
не меняется: с сервером в контейнере он работает как обычный клиент.

## Пример, на котором всё объясняется

Исследователь отдаёт стенд рецензенту статьи. У того есть Docker, но нет uv, Node.js и
опыта с этим репозиторием.

```bash
git clone https://github.com/ksilisk/ragglet.git && cd ragglet
export RAGGLET_COMMIT=$(git rev-parse HEAD)
docker compose -f docker/docker-compose.yaml --profile app up -d --build --wait
# ✔ Container ragglet-redis   Healthy
# ✔ Container ragglet-qdrant  Healthy
# ✔ Container ragglet         Healthy
# UI: http://127.0.0.1:8000
```

Дальше всё как на хосте, из UI или клиентом:

```
$ python3 ragglet status
server: http://127.0.0.1:8000
qdrant: http://localhost:6333 up, 1.17.0, 0 collections
redis: redis://localhost:6379/0 up, 7.4.x
compute: cpu, 8 torch threads
commit: 22df6e9c1a2b, python 3.12.x
running: nothing
queued: 0
$ python3 ragglet build demo
$ python3 ragglet run baseline_qdrant
...
Artifacts: results/runs/baseline_qdrant_20261029T101540Z (on the server)
```

Каталог прогона лежит на хосте в `results/runs/…` (bind mount). В его `manifest.json` видно,
что прогон шёл в контейнере на CPU:

```json
"env": {
  "commit": "22df6e9c…",
  "python": "3.12.x",
  "system": {"platform": "Linux-6.10.14-linuxkit-aarch64-with-glibc2.41", "cpus": 8, …},
  "compute": {"device": "cpu", "accelerator": null, "torch_threads": 8},
  "endpoint_aliases": {"localhost:6333": "qdrant:6333", "localhost:6334": "qdrant-shard-1:6333",
                       "localhost:6379": "redis:6379"}
}
```

Сценарий `baseline_qdrant.yaml` — тот же файл, что на хосте: `endpoint: "http://localhost:6333"`.
`snapshot` прогона его не меняет. Если остановить контейнер сервера и запустить
`python3 ragglet start` на хосте, тот же сценарий пойдёт на тот же Qdrant и найдёт индекс,
собранный в контейнере. Прогон из контейнера при этом виден в `runs list`.

`docker compose -f docker/docker-compose.yaml --profile app down` останавливает стенд.
Результаты остаются в `results/`, индексы — в томе Qdrant, модели и датасеты HF — в томе
кэша.

## Решение

### 1. Образ

`docker/Dockerfile`, две стадии:

```dockerfile
# syntax=docker/dockerfile:1
FROM --platform=$BUILDPLATFORM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build                       # tsc --noEmit && vite build → /web/dist

FROM python:3.12.<patch>-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN useradd --create-home --uid 1000 ragglet \
 && mkdir /home/ragglet/.cache && chown ragglet: /home/ragglet/.cache
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
COPY --from=web /web/dist ./web/dist
ENV PATH=/app/.venv/bin:$PATH RAGGLET_HOST=0.0.0.0 RAGGLET_UI_DIR=/app/web/dist
USER ragglet
EXPOSE 8000
# п. 5; последним слоем, чтобы новый коммит не сбивал кэш остальных
ARG RAGGLET_COMMIT=
ENV RAGGLET_COMMIT=$RAGGLET_COMMIT
CMD ["python", "-m", "ragglet.server"]
```

- **Стадия UI** собирается на платформе сборщика (`$BUILDPLATFORM`): результат — статика, от
  архитектуры не зависит, а под эмуляцией `npm ci` шёл бы в разы дольше. Типы API закоммичены
  (`web/src/api/schema.d.ts`), поэтому Python этой стадии не нужен. В `package-lock.json`
  есть linux-варианты нативных пакетов rollup и esbuild для musl (проверено), поэтому подходит
  `node:22-alpine`. `web/dist` в git не попадает, как и раньше.
- **Стадия сервера.** Python 3.12, как на хосте (3.12.6 в `.venv`). Тег базового образа
  фиксирует версию до патча и дистрибутив, конкретный патч выбирается при реализации. uv —
  та же версия, что у разработчика (0.10.4), бинарником из официального образа. Зависимости
  ставятся отдельным слоем до копирования `src/`: правка кода не перекачивает torch.
  `uv sync --frozen` без `--extra dev`, поэтому pytest, ruff и mypy в образ не попадают.
  Проект ставится editable из `/app/src`, как на хосте. `README.md` нужен setuptools для
  метаданных пакета.
- **Пользователь** `ragglet` (uid 1000) без прав root. `.venv` и код принадлежат root и
  только читаются. Пишет сервер в смонтированные `configs/` и `results/` и в кэш
  `~/.cache` (п. 4).
- **Entrypoint** — `python -m ragglet.server`, а не `python3 ragglet start --foreground`:
  фасад перед стартом делает `uv sync`, `docker compose` и `npm run build`, а в образе всё это
  уже сделано. Настройки — переменные `RAGGLET_*`. `RAGGLET_HOST=0.0.0.0` задаётся в образе
  явно, как требует ragglet-28 §11; наружу порт публикуется только на `127.0.0.1` (п. 2).
- **`.dockerignore`** в корне в форме allow-list: `*`, затем `!pyproject.toml`, `!uv.lock`,
  `!README.md`, `!src/`, `!web/`, и снова исключаются `web/node_modules`, `web/dist`,
  `**/__pycache__`. В контекст сборки не попадут `.env` с ключами, `.git`, `.venv`,
  `results/`, `datasets/` и `.claude/worktrees/`. Последний в основном checkout весит как
  несколько копий репозитория.

**torch: CPU-колёса из индекса PyTorch для Linux.** Сейчас `uv.lock` для
`linux x86_64` тянет CUDA-сборку torch и её зависимости: torch 916 МБ, `nvidia-*` 15 пакетов
(cudnn 707 МБ, cublas 594 МБ, nccl 322 МБ…), `triton`, `cuda-bindings`. Всего около 4,3 ГБ
колёс, и в контейнере без GPU всё это мёртвый груз. Для `linux aarch64` (Docker на Apple
Silicon) PyPI-колесо torch и сейчас CPU-only (146 МБ). Решение — штатный приём uv для PyTorch:

```toml
dependencies = [
  …
  "torch==2.10.*",          # прямая зависимость: [tool.uv.sources] действует только на прямые
]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true             # из этого индекса берётся только torch
```

Проверено на копии lock: `uv lock` убирает `nvidia-*`, `triton` и `cuda-bindings`. Для
Linux (x86_64 и aarch64) в lock появляется `torch 2.10.0+cpu` с хэшами колёс из
`download.pytorch.org/whl/cpu`, macOS остаётся на `torch 2.10.0` из PyPI. Без ограничения
`2.10.*` uv выбрал для Linux самую свежую CPU-сборку (2.15.0+cpu), и версии torch на хосте и
в контейнере разошлись бы. Поэтому версия torch фиксируется в `pyproject.toml`, а обновление
torch — это правка диапазона и `uv lock`.

- **Почему так, а не фильтром в Dockerfile** (рассмотренные альтернативы): lock остаётся
  единственным источником версий. В образе `uv sync --frozen` ставит ровно то, что записано
  в lock, с хэшами, и Dockerfile не знает про torch.
- **Чем платим.** На Linux-хосте без Docker (`python3 ragglet start`) `uv sync` тоже
  поставит CPU-torch. На машине с NVIDIA это потеря CUDA. Сейчас таких стендов нет: стенд
  исследователя — macOS с MPS, а GPU-образы и GPU-стенды вне скоупа. Вернуть CUDA —
  убрать `marker`-строку локально или завести extra `cuda`, когда понадобится (открытый
  вопрос 1).

**Размер и время** (оценка по размерам колёс в lock, замер — в фазе 1). Колёса без dev-пакетов
весят около 300 МБ (aarch64) и 340 МБ (x86_64), из них torch CPU около 150–190 МБ. В
распакованном виде это примерно 0,9–1,1 ГБ venv. Образ целиком — около 1,1–1,3 ГБ, а с
CUDA-колёсами на x86_64 был бы около 8 ГБ. Первая сборка — 3–5 минут: скачивание
зависимостей и `npm ci`. Повторная после правки `src/` — 10–20 с, после правки `web/` — около
минуты: слой зависимостей берётся из кэша. Нативно собирается под архитектуру машины
(arm64 на Apple Silicon). Сборка под x86_64 с `--platform linux/amd64` идёт через эмуляцию
и проверяется в фазе 1 только на размер.

### 2. Compose: профили `app` и `app-multi`

Профили описывают топологию: где сервер и сколько Qdrant.

| Профиль | Сервисы | Кто поднимает |
|---|---|---|
| `qdrant` | `qdrant`, `redis` | `python3 ragglet start` (сервер на хосте), как сейчас |
| `qdrant-multi` | `qdrant-shard-0` (6333), `qdrant-shard-1` (6334), `redis` | руками, сервер на хосте, как сейчас |
| `app` | `ragglet`, `qdrant`, `redis` | `docker compose … --profile app up` |
| `app-multi` | `ragglet`, `qdrant-shard-0`, `qdrant-shard-1`, `redis` | `docker compose … --profile app-multi up` |

У `qdrant` и `qdrant-shard-0` один и тот же порт хоста 6333, поэтому профили с одиночным
Qdrant и с кластером взаимоисключающие, как и сейчас. `qdrant`, `redis` и шарды — те же
сервисы, что у хоста: тома `qdrant_data` и `redis_data` общие. Индекс, собранный
`python3 ragglet build` на хосте, виден серверу в контейнере, и наоборот.

```yaml
x-qdrant-health: &qdrant-health         # в образе Qdrant нет curl и wget, есть bash (проверено)
  test: ["CMD", "bash", "-c",
         "exec 3<>/dev/tcp/127.0.0.1/6333 && printf 'GET /readyz HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q ' 200 ' <&3"]
  interval: 10m
  start_period: 60s
  start_interval: 1s

services:
  ragglet:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      args:
        RAGGLET_COMMIT: ${RAGGLET_COMMIT:-}
    image: ragglet:local
    container_name: ragglet
    init: true                          # tini — PID 1: сигналы серверу, сироты-воркеры видят смену ppid
    stop_grace_period: 20s              # cancel_grace_s сервера (10 с) и запас; по умолчанию у Docker 10 с
    ports: ["127.0.0.1:8000:8000"]      # авторизации нет
    environment:
      RAGGLET_ENDPOINT_ALIASES: localhost:6333=qdrant:6333,localhost:6334=qdrant-shard-1:6333,localhost:6379=redis:6379
      RAGGLET_QDRANT_ENDPOINTS: ${RAGGLET_QDRANT_ENDPOINTS:-http://localhost:6333}
      OPENAI_API_KEY:                   # из shell, если задан; иначе переменной в контейнере нет
    volumes:
      - ../configs:/app/configs
      - ../datasets:/app/datasets:ro
      - ../results:/app/results
      - ragglet_cache:/home/ragglet/.cache
    depends_on:
      redis: {condition: service_healthy}
      qdrant: {condition: service_healthy, required: false}
      qdrant-shard-0: {condition: service_healthy, required: false}
      qdrant-shard-1: {condition: service_healthy, required: false}
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/v1/openapi.json', timeout=5)"]
      interval: 10m
      start_period: 120s
      start_interval: 1s
    profiles: ["app", "app-multi"]

  redis:          # + healthcheck redis-cli ping (тот же ритм), + профили app, app-multi
  qdrant:         # ports 127.0.0.1:6333:6333, healthcheck *qdrant-health, профили qdrant, app
  qdrant-shard-0: # ports 127.0.0.1:6333:6333, *qdrant-health, профили qdrant-multi, app-multi,
                  # networks: {default: {aliases: [qdrant]}}
  qdrant-shard-1: # ports 127.0.0.1:6334:6333, *qdrant-health, профили qdrant-multi, app-multi
```

- **`depends_on` с `required: false`** — так один сервис `ragglet` работает в обоих профилях.
  Неактивный Qdrant не мешает, активный сервер дожидается до `healthy`. Проверено на
  compose 2.31 (`--dry-run up` по каждому профилю). Сервер стартует и без Qdrant: стенд
  покажет его `down`, а dry-run даст `storage.unreachable`. `depends_on` нужен для
  предсказуемого `up` и для `--wait`.
- **Ритм healthcheck.** Docker запускает проверку отдельным процессом внутри контейнера.
  У сервера это `python -c …`: около 50 мс CPU на старт интерпретатора, на бенчмарке это
  шум в задержках. Поэтому на старте проверка идёт раз в секунду (`start_interval`), а после
  `healthy` — раз в 10 минут. `--wait` получает готовность быстро, а прогон почти не
  задевает. Проверка сервера дёргает `/v1/openapi.json`: он кэшируется FastAPI, не ходит в
  Qdrant и не грузит torch, в отличие от `/v1/stand`.
- **Порты Qdrant — только `127.0.0.1`.** Сейчас `qdrant` и шарды слушают `0.0.0.0:6333`, то
  есть Qdrant без авторизации открыт в локальную сеть. Redis на `127.0.0.1` уже с
  ragglet-37. С хоста ничего не меняется: и фасад, и клиенты ходят на `localhost`.
- **Сеть** — сеть проекта compose по умолчанию. Проект по-прежнему называется по каталогу
  файла (`docker`), имена томов `docker_qdrant_data` и др. не меняются. Если задать
  `name:` проекту, тома переименуются и данные исследователя «потеряются».
- **Уборка.** Сервис `ragglet` сейчас ссылается на несуществующий `docker/Dockerfile`,
  монтирует `../data` (такого каталога нет), задаёт неиспользуемый `RAGGLET_ENV=dev`, а
  комментарий в шапке называет файл `compose.yaml`. Всё это заменяется. Закомментированный
  блок Milvus удаляется. `pgvector` не трогаем: backend не реализован, это вне задачи.

### 3. Адреса сервисов внутри контейнера

**Проблема.** Адрес Qdrant и Redis задаёт конфиг: `storage.backend.endpoint`,
`storage.backends[].endpoint`, `cache.embedding.endpoint` в сценариях и `storage.endpoint` в
профилях. В `configs/` это везде `http://localhost:6333` (у шарда 1 — `:6334`) и
`redis://localhost:6379/<db>`. Внутри контейнера `localhost` — сам контейнер, а Qdrant
доступен как `qdrant:6333`. При этом те же файлы должны работать и на хосте. Адрес из конфига
используется в нескольких местах:

1. подключение: `AsyncQdrantStore`, `list_collections`, `QdrantIndexer`, `redis.from_url`
   (кэш и проба версии); через них идут preflight, прогон, сборка и пробы API;
2. snapshot прогона и etag библиотеки: «Повторить» берёт snapshot, а реестр сравнивает его с
   файлом сценария, чтобы показать `scenario_file: changed` (ragglet-35);
3. `GET /v1/indexes` и `used_by`/`profiles`: пары `(endpoint, collection)` из конфигов
   сверяются с `RAGGLET_QDRANT_ENDPOINTS` (ragglet-28 §7).

Метаданные индекса endpoint не хранят: preflight проверяет корпус, модель, раскладку storage
и полноту сборки, а не адрес. Поэтому индекс, собранный через `localhost:6333` с хоста, —
та же коллекция, что видна из контейнера через `qdrant:6333`.

**Варианты.**

| Вариант | Как | Почему нет |
|---|---|---|
| (а1) подмена в конфиге | перед preflight сервер переписывает `endpoint` в принятом конфиге | snapshot получает `qdrant:6333`: «Повторить» на хосте не найдёт Qdrant, etag разойдётся с файлом (ложный `scenario_file: changed`), `used_by` не сматчится с `localhost:6333` |
| **(а2) подмена при подключении** | конфиг не трогаем, `host:port` меняется там, где создаётся клиент | — выбран |
| (б) `network_mode: host` | `localhost` контейнера — сеть хоста | полноценно только на Linux; в Docker Desktop для macOS — бета за флагом в настройках (4.34+) |
| (в) переменные в YAML | `endpoint: ${QDRANT_URL:-http://localhost:6333}` | подстановка в загрузчике, правка 21 файла в `configs/`, в UI-редакторе и snapshot — шаблон вместо адреса; каждый новый сценарий должен помнить об этом |
| (г) общий network namespace | `qdrant` и `redis` с `network_mode: service:ragglet`, тогда `localhost:6333` в контейнере — Qdrant | у таких сервисов нельзя `ports:`, поэтому нужна вторая копия `qdrant`/`redis` на тех же томах, рядом с копией для хоста: два Qdrant на одном томе портят данные; шард 1 должен слушать 6334 внутри |
| (д) TCP-прокси в контейнере | socat `localhost:6333 → qdrant:6333` | лишний хоп в измеряемом пути retrieval и второй процесс в контейнере |
| (е) `/etc/hosts` / `extra_hosts` | имена `qdrant`, `redis` → 127.0.0.1 на хосте | `localhost` так не перенаправить, порт не переназначить; правка `/etc/hosts` хоста требует sudo |

**Решение (а2).** Новая настройка сервера:

```
RAGGLET_ENDPOINT_ALIASES=localhost:6333=qdrant:6333,localhost:6334=qdrant-shard-1:6333,localhost:6379=redis:6379
```

- **Формат** — пары `откуда=куда` через запятую, обе стороны — `host:port`. Правило
  срабатывает на любой URL сервиса (`http`, `https`, `redis`, `rediss`), у которого `host:port`
  совпал с левой частью. Схема, логин и пароль, путь, номер БД Redis (`/15` в
  `b1_cache_phases`) и query остаются как были. Имена loopback (`localhost`, `127.0.0.1`,
  `::1`) считаются одним хостом. Адрес без явного порта не подменяется. Битая строка —
  ошибка настроек на старте сервера.
- **Где применяется.** Одна функция `connect_url(url)` в новом `src/ragglet/core/endpoints.py`
  вызывается там, где создаётся клиент: `AsyncQdrantClient` в `AsyncQdrantStore` и
  `list_collections`, `QdrantClient` в `QdrantIndexer`, два `redis.from_url` в
  `cache/embedding_redis.py`. Других мест подключения в коде нет (проверено поиском). Всё
  остальное — preflight, `describe`, реестр, пробы и сопоставление коллекций — видит адрес из
  конфига.
- **Как значение доходит до воркера.** `ServerSettings.endpoint_aliases` разбирает и
  проверяет строку тем же парсером. Значит, работает и `.env`, как у остальных `RAGGLET_*`, а
  сервер печатает таблицу при старте. `python -m ragglet.server` кладёт итоговое значение в
  `os.environ`. Воркер ragglet-29 наследует окружение сервера, а ядро читает переменную в
  `connect_url`. `jobs/` не меняется. Адрес сервиса — свойство окружения стенда, а не
  эксперимента (как `HF_HOME`), поэтому живёт в окружении процесса, а не в конфиге.
- **Ошибки.** Если подменённый адрес недоступен, в тексте `storage.unreachable` и
  `cache.unreachable` стоят оба адреса: `http://localhost:6333 (via http://qdrant:6333):
  ConnectError …`. Иначе в контейнере было бы непонятно, куда на самом деле шёл запрос.
- **Что пишется в manifest.** Конфиг и snapshot — как есть, с `localhost`. Именно этот
  конфиг воспроизводит эксперимент на любом стенде, и «Повторить» из snapshot работает и на
  хосте, и в контейнере. Подмена — факт окружения, поэтому `manifest.env.endpoint_aliases`
  хранит действовавшую таблицу. Ключ есть, только если переменная задана: манифесты хоста не
  меняются. По адресу из конфига и таблице реальный адрес восстанавливается однозначно. Тот
  же `env` пишется в `build_env` метаданных индекса, так что видно, что индекс собран в
  контейнере.
- **`RAGGLET_QDRANT_ENDPOINTS` в контейнере** остаётся логическим `http://localhost:6333`.
  Пробы идут через `connect_url`, а `/v1/stand` и `/v1/indexes` показывают тот же адрес, что
  в сценариях и в браузере на хосте (порт опубликован). `used_by` сопоставляется как на
  хосте. Для кластера список задаётся той же переменной из shell (п. 7).
- **Кластер.** У `qdrant-shard-0` есть сетевой алиас `qdrant`, поэтому таблица одна на оба
  профиля. В `app` по имени `qdrant` отвечает одиночный Qdrant, в `app-multi` — шард 0, то
  есть ровно тот, кто на хосте занимает порт 6333. В `app` правило для 6334 ведёт на
  несуществующее имя. Сценарий с шардами получит `storage.unreachable`, как на хосте без
  `qdrant-multi`. Тест сверяет таблицу с `ports:` сервисов compose, чтобы они не разъехались.
- **Чего не делаем.** Конфиги с `http://qdrant:6333` будут работать только в контейнере. В
  README сказано писать `localhost`: таблица сделает остальное.

### 4. Тома: конфиги, датасеты, результаты, кэш

Рабочий каталог в образе — `/app`. Относительные умолчания `ServerSettings` (`configs/…`,
`datasets`, `results/…`) указывают туда же, куда на хосте, поэтому `dataset.id:
datasets/demo`, `run_dir: results/runs/<id>` и пути в ответах API одинаковы в обоих режимах.

| Хост | Контейнер | Режим | Зачем |
|---|---|---|---|
| `configs/` | `/app/configs` | чтение и запись | библиотека сценариев и профилей правится из UI и CLI (`PUT /v1/scenarios/…`) |
| `datasets/` | `/app/datasets` | только чтение | локальные BEIR-датасеты; сервер их только читает |
| `results/` | `/app/results` | чтение и запись | реестр и артефакты: `results/runs/<id>/`, `results/builds/<id>/` |
| том `ragglet_cache` | `/home/ragglet/.cache` | чтение и запись | модели и датасеты HF (`~/.cache/huggingface`), BEIR из parquet (`~/.cache/ragglet/beir`) |

- **Датасет.** Локальный — положить BEIR-триаду в `datasets/<имя>/` на хосте (формат — README
  «Формат датасета»). Он сразу виден в контейнере и в `GET /v1/datasets`, в конфиге —
  `dataset.id: datasets/<имя>`. HF-датасет (`source: hf`) скачивается при первом dry-run или
  сборке в том кэша. Модели — туда же при первой загрузке.
- **Кэш — именованный том**, а не `~/.cache` хоста. После пересоздания контейнера (новый
  образ) ничего не качается заново, а `down -v` его удаляет. Bind mount кэша хоста сэкономил
  бы одно скачивание MiniLM (~90 МБ), но завязал бы контейнер на пути хоста. На Linux
  добавились бы проблемы с uid, а контейнер писал бы в кэш хоста.
- **Забрать результаты** — файлы в `results/` на хосте (pandas, `compare_runs`), zip через API
  (`python3 ragglet runs export <id>`) или UI.
- **uid на Linux.** Docker Desktop на macOS отображает владельца bind mount прозрачно. На
  Linux файлы, которые пишет контейнер, принадлежат uid 1000; если у пользователя другой uid
  и `results/` не доступен на запись uid 1000, сервер не сможет писать. В README — одна
  строка с обходом (`chown` или `user:` в override-файле). Общее решение вне скоупа.
- **Один стенд — один сервер.** `results/` и Qdrant общие с режимом хоста. Если одновременно
  работают сервер на хосте и сервер в контейнере, это два менеджера задач: две тяжёлые
  задачи сразу (нарушение рамки 38), гонки записи реестра и чужие задачи, которые при
  рестарте помечаются `ServerRestart`. На порте 8000 это случайно предотвращается: второй
  сервер не займёт порт, а `python3 ragglet start` при живом контейнере говорит «already
  running». На другом порте защиты нет, README это оговаривает. Блокировка через файл в
  `results/` не делается: `flock` через bind mount Docker Desktop ненадёжен.

### 5. Коммит образа и что видно в manifest

- **Коммит.** `git_commit()` вызывает `git rev-parse HEAD`. В образе нет ни git, ни `.git`,
  и `manifest.env.commit` был бы `null` у всех прогонов в Docker, то есть дыра в
  воспроизводимости. Решение: аргумент сборки `RAGGLET_COMMIT` попадает в `ENV` образа, а
  `git_commit()` при неудаче git возвращает `os.environ.get("RAGGLET_COMMIT") or None`
  (одна строка в `core/environment.py`). Сначала git, потом переменная: если переменная
  экспортирована в shell, а потом запущен `python3 ragglet start` на хосте, сервер хоста
  всё равно запишет свой HEAD. Значение даёт `export RAGGLET_COMMIT=$(git rev-parse HEAD)`
  перед `up --build` (п. 7). Если его забыть, коммит будет `null`, и это видно в `status` и
  на странице стенда (`commit: -`).
- **Почему не автоматически.** Compose не умеет подставлять вывод команды в `args`.
  Скопировать `.git` в контекст сборки и вызвать там `git rev-parse` нельзя в worktree:
  там `.git` — файл со ссылкой на каталог вне контекста. Смонтировать `.git` в контейнер
  тоже нельзя: если после `git pull` образ не пересобрали, `HEAD` укажет не на тот код,
  что запущен.
- **Как отличить прогон из Docker:** `env.compute.device: "cpu"`; `env.system.platform`
  начинается с `Linux-…` (в Docker Desktop — `…-linuxkit-…`); есть
  `env.endpoint_aliases`; `cpus` и `ram_bytes` — ресурсы VM Docker Desktop (cpuset
  учитывается, ragglet-23). Индекс, собранный в контейнере, несёт то же в
  `index.build_env`. Отдельный флаг «контейнер» не нужен.
- **Несопоставимость замеров.** Задержки из контейнера нельзя сравнивать с хостовыми по двум
  причинам. Эмбеддинг считается на CPU вместо MPS. Путь до Qdrant тоже другой: с хоста
  запрос идёт через проброс порта Docker Desktop, из контейнера — через мост внутри VM.
  Качество (nDCG, recall) сопоставимо: модель та же (ревизия берётся из индекса), числа
  эмбеддингов CPU и MPS расходятся только в младших разрядах. README говорит об этом прямо.

### 6. Фасад `ragglet`

**`start --docker` не добавляется.** Предложение — README с двумя командами compose (`up`,
`down`). Причины:

- кто запускает стенд в Docker (рецензент, вторая машина, Linux-сервер), знает compose;
  исследователь на своём Mac продолжает жить в `python3 ragglet start` с MPS;
- клиент и так работает с контейнером. `status`, `run`, `build`, `runs …` ходят на
  `http://127.0.0.1:8000` (адрес по умолчанию, `server.json` нет). `start` при живом
  контейнере печатает «already running» и не поднимает второй сервер;
- `--docker` потянул бы около 40 строк и тестов: `up --build --wait` с `RAGGLET_COMMIT`,
  отметку в `server.json`, ветку в `stop`, выбор `app`/`app-multi`. А фасад по рамке 37 — тонкий.

Единственная правка в фасаде — подсказка в `stop`. Сейчас для сервера, поднятого не через
`start`, он пишет `stop it where it runs`, к этому добавляется `(Docker: docker compose -f
docker/docker-compose.yaml --profile app stop ragglet)`. Если `start --docker` всё же нужен —
открытый вопрос 2.

### 7. Документация

README:

- **Новый раздел «9. Docker»** в «Как запустить»: команды `up`/`down` для `app` и
  `app-multi`, что поднимается и на каких портах, тома (таблица из п. 4), как положить
  датасет и собрать индекс, где результаты и как их забрать. Плюс CPU и несопоставимость
  задержек (п. 5), `RAGGLET_COMMIT`, «один стенд — один сервер», uid на Linux, обновление
  (`git pull` → `up --build`), полная очистка (`down -v` удаляет индексы и кэш) и smoke ниже
  как пошаговая проверка.
- **Кластер в Docker:**
  ```bash
  RAGGLET_QDRANT_ENDPOINTS=http://localhost:6333,http://localhost:6334 \
    docker compose -f docker/docker-compose.yaml --profile app-multi up -d --build --wait
  ```
  Без переменной кластер работает, но в списке коллекций UI будут только коллекции шарда 0.
- **Таблица `RAGGLET_*`** в «6. REST API»: строка `RAGGLET_ENDPOINT_ALIASES` (по
  умолчанию пусто; в compose — таблица из п. 3). Отдельно перечисляется, что задаёт образ
  (`RAGGLET_HOST`, `RAGGLET_UI_DIR`, `RAGGLET_COMMIT`) и что задаёт compose.
- **«Нужны»** в начале «Как запустить»: строка «только Docker — раздел 9».
- **«Частые проблемы»:** строка «`--profile app` не собирается, это ожидаемо» заменяется
  тремя: `commit: -` (забыт `RAGGLET_COMMIT`), `storage.unreachable … (via …)` в контейнере
  (сервис не поднят или не тот профиль), занят порт 8000 (работает `python3 ragglet start`).
- Разделы 1–8 правятся точечно: ragglet-32 параллельно трогает раздел 8 и `web/`.

## Smoke (критерий приёмки)

Воспроизводится руками на macOS (arm64) перед мержем, вывод — в описании PR. Скрипт в
репозиторий не кладётся (по примеру ragglet-35).

1. **Сборка и старт.** `export RAGGLET_COMMIT=$(git rev-parse HEAD)` →
   `docker compose -f docker/docker-compose.yaml --profile app up -d --build --wait`. Все три
   контейнера `Healthy`. Замерить время первой и повторной сборки и размер образа
   (`docker image ls ragglet:local`), в том числе `--platform linux/amd64`.
2. **Стенд.** `python3 ragglet status`: Qdrant и Redis `up` по `localhost`-адресам,
   `compute: cpu`, `commit` = HEAD. То же curl-ом:
   `curl -s 127.0.0.1:8000/v1/stand | jq '{q: [.qdrant[].available], r: .redis.available, compute, commit}'`.
3. **UI.** `http://127.0.0.1:8000` открывается в браузере, прямая ссылка `/runs/<id>` отдаёт
   `index.html`, на странице стенда сервисы зелёные.
4. **Индекс.** `python3 ragglet build demo` → `completed`. То же curl-ом:
   `curl -s -X POST 127.0.0.1:8000/v1/builds -H 'content-type: application/json' -d "$(jq -n --rawfile p configs/indexes/demo.yaml '{profile: $p, profile_id: "demo"}')"`
   → 202 и id; статус — `GET /v1/builds/<id>`.
5. **Прогон.** `python3 ragglet run baseline_qdrant` → summary. Каталог
   `results/runs/<id>/` появился на хосте. В `manifest.json`: `env.compute.device = cpu`,
   `env.endpoint_aliases`, `env.commit` = HEAD, `scenario.snapshot` с `localhost:6333`.
6. **Правка из UI.** Сохранить сценарий в редакторе → файл в `configs/scenarios/` на хосте
   изменился (том на запись).
7. **Перезапуск.** `docker compose … --profile app up -d --force-recreate ragglet` → прогон
   на месте (`runs list`), повторный `run` не качает модель: в `task.log` нет загрузки,
   «подготовка» короче.
8. **Совместимость с хостом.** `docker compose … stop ragglet` → `python3 ragglet start` →
   `python3 ragglet run baseline_qdrant --dry-run` находит индекс, собранный в контейнере. В
   `runs list` есть прогон из контейнера. Затем `python3 ragglet stop`.
9. **Кластер.** `--profile app` `down` → `RAGGLET_QDRANT_ENDPOINTS=…6333,…6334 … --profile
   app-multi up -d --wait` → `build demo_sharded` → `run s5_distributed_like --dry-run`: OK,
   обе цели найдены.
10. **Остановка.** `docker compose … --profile app-multi down` → контейнеров нет, тома на
    месте. `stop` фасада при живом контейнере печатает подсказку из п. 6.

`tests/test_e2e_api.py` на контейнер не направить: он поднимает приложение в процессе
(`TestClient`) и свои процессы сервера во временных каталогах, роняет их `SIGKILL` и смотрит
pid воркеров. Режима «внешний сервер по `RAGGLET_API_URL`» у него нет, а добавлять его ради
этой задачи дорого. Тот же путь (сборка, прогон, результаты, рестарт) покрывает smoke выше.

Smoke идёт на машине исследователя в том же compose-проекте `docker`, что и его стенд.
Первый `up` пересоздаст контейнеры `ragglet-qdrant` и `ragglet-redis` (новые порты и
healthcheck), тома сохранятся. `build demo` пересоберёт коллекцию `chunks_v1` тем же
профилем `demo`, а шаг 9 — `chunks_v1_shard*` в томах шардов (открытый вопрос 3).

## Что меняется для пользователя

- Появляется работающий профиль `app` (и `app-multi`); README больше не называет его
  «зарезервированным».
- Qdrant (и шарды) публикуются только на `127.0.0.1`. Из локальной сети к ним больше не
  подключиться.
- Первый `docker compose … up` после обновления пересоздаёт контейнеры Qdrant и Redis
  (healthcheck, порты). Данные в томах не трогаются.
- На Linux без Docker `uv sync` ставит CPU-сборку torch (п. 1). На macOS ничего не меняется.
- Манифесты прогонов на хосте не меняются: `endpoint_aliases` появляется, только если
  переменная задана.

## Затрагиваемые модули

| Файл | Что |
|---|---|
| `docker/Dockerfile` | новый, п. 1 |
| `.dockerignore` | новый, allow-list |
| `docker/docker-compose.yaml` | `ragglet`, профили `app`/`app-multi`, healthcheck, порты `127.0.0.1`, алиас шарда 0, том `ragglet_cache`, уборка |
| `pyproject.toml`, `uv.lock` | `torch==2.10.*`, `[tool.uv.sources]`, индекс `pytorch-cpu` |
| `src/ragglet/core/endpoints.py` | новый: `parse_aliases`, `connect_url`, `active_aliases` |
| `src/ragglet/stores/qdrant_store_async.py`, `src/ragglet/indexing/qdrant_indexer.py`, `src/ragglet/cache/embedding_redis.py` | `connect_url(...)` при создании клиента (5 строк), «via» в тексте `ServiceUnavailable` |
| `src/ragglet/core/environment.py` | `RAGGLET_COMMIT` как запасной коммит; `endpoint_aliases` в `capture_env` |
| `src/ragglet/api/settings.py` | поле `endpoint_aliases`, строка в `describe()` |
| `src/ragglet/server.py` | экспорт `RAGGLET_ENDPOINT_ALIASES` в `os.environ` перед стартом |
| `ragglet` | подсказка в `stop` |
| `README.md` | раздел «9. Docker», строка таблицы, «Нужны», «Частые проблемы» |

`jobs/`, `registry/`, `api/routers/`, `web/` не меняются. Правки вне `api/settings.py` и
`server.py` из брифа: места подключения (п. 3) — это «место применения подмены».
`core/environment.py` (п. 5) нужен ради коммита в manifest.

## Тесты

Быстрые, без Docker (`-m "not integration"`):

1. `parse_aliases`: пары, пробелы, пустая строка; битая запись (`localhost=qdrant`, без порта,
   без `=`) → `ValueError` с текстом записи.
2. `connect_url`: без переменной — URL как есть. `http://localhost:6333` →
   `http://qdrant:6333`. `http://127.0.0.1:6333` → тоже (loopback). Для
   `redis://:pw@localhost:6379/15` сохраняются пароль и `/15`. `http://localhost:6335` и
   `http://localhost` (без порта) не меняются.
3. Места подключения: при заданной переменной `AsyncQdrantStore`, `list_collections`,
   `QdrantIndexer` и `RedisEmbeddingCache` создают клиента с подменённым адресом
   (monkeypatch конструктора). `ServiceUnavailable` содержит `via`.
4. `ServerSettings`: `RAGGLET_ENDPOINT_ALIASES` из окружения и из `.env`; битое значение —
   ошибка валидации; `describe()` печатает таблицу.
5. `capture_env`: `endpoint_aliases` есть только при заданной переменной; `git_commit()` без
   git (monkeypatch `subprocess.run` → `OSError`) возвращает `RAGGLET_COMMIT`, а без неё —
   `None`.
6. Compose: YAML разбирается. Для каждого правила `localhost:P=svc:Q` из
   `RAGGLET_ENDPOINT_ALIASES` есть сервис (или сетевой алиас) `svc`, публикующий
   `127.0.0.1:P:Q`. У `ragglet` есть тома `configs`, `datasets:ro`, `results`, кэш и порт
   `127.0.0.1:8000:8000`. `qdrant` и `qdrant-shard-0` не в одном профиле.
7. Фасад: `stop` при чужом сервере печатает подсказку про Docker (правка существующего теста).

Ручные: сборка образа и smoke выше. Перед коммитами с кодом —
`uv run --extra dev pytest -q -m "not integration"`, `uv run --extra dev ruff check src tests ragglet`.

## Фазы (отдельными коммитами)

1. **Образ.** `pyproject.toml` + `uv.lock` (CPU-torch), `docker/Dockerfile`, `.dockerignore`,
   `RAGGLET_COMMIT` в `core/environment.py`. Проверка: сборка arm64 и amd64, размер, время,
   `docker run` отвечает на `/v1/openapi.json` и отдаёт UI.
2. **Compose и адреса.** `core/endpoints.py`, места подключения, `settings.py`, `server.py`,
   `capture_env`; compose (`app`, `app-multi`, healthcheck, порты, том кэша); тесты 1–6.
3. **Фасад.** Подсказка в `stop` и тест 7.
4. **README и smoke.** Раздел «9. Docker» и правки, smoke по шагам 1–10, итог — в описании
   PR. Док сверяется с кодом.

## Вне скоупа

- GPU-образы и CUDA-сборка torch; extra `cuda` для Linux-стендов.
- Публикация образа в registry, теги версий, подписи, pinning базовых образов по digest.
- Несколько стендов на одной машине, оркестрация, авторизация и TLS.
- Windows без Docker; разные uid на Linux в общем виде.
- Изоляция CPU между Qdrant и сервером (`cpuset`) ради чистоты замеров.
- Направить `tests/test_e2e_api.py` на внешний сервер.
- Milvus, Weaviate, pgvector в compose сверх того, что есть.

## Рассмотренные альтернативы

- **torch фильтром в Dockerfile** (`uv export` без torch/nvidia/triton, затем `uv pip install
  torch==2.10.0 --index-url …/cpu`). Lock и хост не трогаются, но torch в образе ставится
  без хэшей мимо lock, окружение образа перестаёт равняться `uv.lock`, а фильтр по именам
  строк хрупок. Выбран штатный `[tool.uv.sources]`.
- **Образ только под arm64** (там PyPI-torch и так CPU). Воспроизведение на x86_64-сервере —
  основной случай для «воспроизводимого окружения», а там было бы ~8 ГБ.
- **`start --docker` в фасаде** — п. 6, открытый вопрос 2.
- **`python3 ragglet start --foreground` как entrypoint** (так предполагал ragglet-37) — п. 1:
  фасад готовит окружение, которое в образе уже готово.
- **Конфиги и `datasets/demo` внутри образа.** `docker run` без томов работал бы сразу, но
  правки из UI и результаты пропадали бы с контейнером. Compose всё равно монтирует каталоги,
  так что копия в образе — только лишний размер.
- **Кэш HF хоста bind mount-ом** — п. 4.
- **`.git` в образе или в томе ради коммита** — п. 5.
- **`restart: unless-stopped`.** Упавший сервер поднимался бы сам, но на стенде для замеров
  это прячет падения. Падение видно в `docker compose ps`, рестарт — руками, `recover` при
  этом сработает (ragglet-29 §8).

## Открытые вопросы

1. **CPU-torch для всех Linux** (`sys_platform == 'linux'`) — подходит? Альтернатива — extra
   `cpu` с конфликтом `cuda`: lock сохранит обе сборки, но на хосте придётся помнить про
   `--extra`, а `ragglet start` — его передавать. Для одного стенда на macOS я бы не
   усложнял.
2. **`python3 ragglet start --docker`** — не делать (п. 6) или всё же нужен как «одна
   команда» и для Docker-режима?
3. **Smoke на твоей машине.** Он пересоздаст контейнеры `ragglet-qdrant` и `ragglet-redis`
   (тома сохранятся) и пересоберёт `chunks_v1` и `chunks_v1_shard*` профилями `demo` и
   `demo_sharded`. Если в этих коллекциях сейчас что-то нужное (другая сборка), скажи:
   прогоню smoke на отдельных коллекциях через копии профилей.
