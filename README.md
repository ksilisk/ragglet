# Ragglet

**Ragglet** — исследовательский стенд для воспроизводимого бенчмаркинга retrieval-части в системах класса RAG (Retrieval-Augmented Generation). Проект ориентирован на учебно-научную работу: позволяет собирать индекс по корпусу документов, запускать сценарии экспериментов и сравнивать качество поиска по фиксированному набору запросов.

## Назначение проекта

Проект предназначен для проверки того, как параметры индексации, стратегия поиска и кэширование влияют на:

- качество извлечения релевантных фрагментов;
- задержки на этапах embedding и retrieval;
- устойчивость сценария при серийных прогонах.

В текущей версии основной акцент сделан именно на **retrieval benchmark**, а не на полной генеративной RAG-цепочке.

## Что реализовано

- поддержка готовых датасетов в индустриальном формате **BEIR** (локально или с HuggingFace Hub);
- офлайн-индексация корпуса в Qdrant;
- разбиение документов на чанки (или индексация документов целиком, `chunking.method: none`) и построение эмбеддингов через `sentence-transformers`;
- запуск экспериментальных сценариев из YAML-конфигураций;
- поддержка single-source retrieval и fanout/hybrid retrieval;
- подсчет метрик `nDCG@K`, `Recall@K`, `R_cap@K`, `Hit@K`, `MRR@K` на уровне документов (опционально `MRR@K`, `Precision@K` на уровне чанков), latency и cache hit rate; что значит каждая метрика — [docs/metrics.md](docs/metrics.md);
- сохранение покомпонентных и агрегированных результатов в `results/`.

## Формат датасета (BEIR)

Канонический формат — [BEIR](https://github.com/beir-cellar/beir): так раздаются MS MARCO,
NQ, HotpotQA, SciFact, FiQA, TREC-COVID и другие. Один формат — один адаптер. Датасет это
директория (или HF-репозиторий) из трёх частей:

```text
<dataset>/
  corpus.jsonl      {"_id": "d12", "title": "...", "text": "..."}
  queries.jsonl     {"_id": "q3",  "text": "..."}
  qrels/<split>.tsv query-id <TAB> corpus-id <TAB> score   # TREC-style, с заголовком
```

Ground truth (`qrels`) ссылается на **id документа** (`corpus._id`), поэтому метрики
считаются на document-level: retrieved-чанки проецируются в свой `doc_id` с дедупом.

Структура проекта:

- `datasets/demo/` — демо-датасет в формате BEIR;
- `configs/indexes/*.yaml` — профили построения индекса;
- `configs/scenarios/*.yaml` — сценарии экспериментальных прогонов.

Свой корпус из `.md`/`.txt` можно сконвертировать в BEIR:

```bash
uv run ragglet dataset import <corpus_dir> <out_dir> --queries <legacy.jsonl> --split test
```

## Как запустить

### 1. Подготовить окружение

Нужны:

- Python 3.12+;
- `uv`;
- Docker с Docker Compose.

Установить Python-зависимости:

```bash
uv sync
```

Если зависимости уже установлены в локальный `.venv`, можно запускать команды через `.venv/bin/ragglet`, но основной путь для проекта — `uv run ...`.

### 2. Поднять Qdrant и Redis

Для базового vector-only benchmark нужен Qdrant. Redis нужен для сценариев с кэшем, но его удобно поднимать сразу:

```bash
docker compose -f docker/docker-compose.yaml --profile qdrant up -d qdrant redis
```

Проверить, что контейнеры запущены:

```bash
docker compose -f docker/docker-compose.yaml ps
```

По умолчанию используются:

- Qdrant: `http://localhost:6333`;
- Redis: `redis://localhost:6379/0`.

### 3. Собрать индекс

Индекс строится по профилю `configs/indexes/demo.yaml` из BEIR-корпуса `datasets/demo/` в Qdrant collection `chunks_v1`:

```bash
uv run ragglet index build configs/indexes/demo.yaml
```

Датасет в профиле задаётся блоком `dataset: {source: local|hf, id: ...}`. Для готового
датасета с HuggingFace Hub достаточно указать `source: hf` и `id` репозитория (например
`BeIR/scifact`) — файлы BEIR скачаются в кэш автоматически.

Перед эмбеддингом `index build` проверяет весь датасет (corpus, queries, все split qrels):
повторы `_id`, qrels на несуществующие документы и запросы, битые строки, пустые
релевантные документы. Нарушения печатаются все сразу, с файлом и строкой; есть ошибка —
индекс не собирается.

Только эти проверки, без загрузки модели и без пересоздания коллекций (команда печатает,
какие коллекции будут пересозданы):

```bash
uv run ragglet index build configs/indexes/demo.yaml --dry-run
```

Метаданные индекса (хэш корпуса, ревизия модели эмбеддингов, chunking, storage layout)
записываются в Qdrant collection **после** последнего upsert: коллекция без них считается
недособранной.

### 4. Запустить benchmark

Перед прогоном `bench run` делает preflight: проверяет, что сценарий использует только
реализованное (стратегия `cascade`, второй `keyword`-источник, неизвестный
`params.backend_ref` отклоняются сразу), что датасет найден и валиден, что Qdrant и Redis
(при `cache.embedding`) отвечают, и что индекс собран полностью, из того же корпуса (по хэшу
содержимого), с той же моделью эмбеддингов и в том же storage layout. При любом
расхождении прогон не начинается и каталог результатов не создаётся. Модель эмбеддингов
загружается ровно той ревизии, которой собран индекс.

Тот же preflight без загрузки моделей и прогона запросов:

```bash
uv run ragglet bench run configs/scenarios/baseline_qdrant.yaml --dry-run
```

Ошибки конфига, датасета и индекса команды печатают списком и выходят с кодом 2.

Базовый сценарий: один vector source, без кэша.

```bash
uv run ragglet bench run configs/scenarios/baseline_qdrant.yaml
```

Hybrid/fanout сценарий: Qdrant vector + in-memory BM25, Redis embedding cache и memory LRU retrieval cache.

```bash
uv run ragglet bench run configs/scenarios/baseline_qdrant_fanout_two_vec.yaml
```

Для fanout-сценария Redis должен быть доступен, потому что `cache.enabled: true` включает Redis embedding cache.

### 5. Посмотреть результаты

Каждый запуск создает отдельную папку:

```text
results/runs/<scenario_name>_<utc_timestamp>/
```

Типичные файлы:

- `scenario.snapshot.json` и `scenario.snapshot.yaml` — нормализованный снимок scenario;
- `manifest.json` — что и где измеряли:
  - `dataset` — ревизия датасета, хэши корпуса и eval-набора;
  - `index` — сборка индекса: profile, время, ревизия модели, chunking, фактические параметры коллекции (`params`) и окружение сборки (`build_env`);
  - `rerank` — ревизия rerank-модели (`null`, если rerank выключен);
  - `env` — окружение прогона: commit ragglet, Python, машина (`system`), устройство эмбеддингов (`compute`), версии и состояние сервисов (`services`: storage, cache);
- `per_query.repeat0.csv` / `.parquet` — результаты по каждому запросу, если `run.save_per_query: true`;
- `summary.csv` и `summary.json` — сводка по каждому repeat: качество, перцентили задержек, исходы запросов (`ok`, `timeout`, `error`), время повтора `wall_ms` и пропускная способность `throughput_qps`. Правила подсчёта в [docs/metrics.md](docs/metrics.md#задержки-и-пропускная-способность);
- `summary.aggregate.csv` и `summary.aggregate.json` — агрегация по repeats.

Формат per-query файлов задается через `output.format: csv | parquet | both`. Summary-файлы сейчас всегда пишутся в CSV и JSON.

Quality-метрики двух прогонов сравнимы, если в `manifest.json` у них равны `dataset.corpus_sha256` и `dataset.eval_sha256`. Различия индекса (chunking, модель) — экспериментальный фактор и видны в секции `index`.

`ragglet.bench.compare.compare_runs(a, b)` сравнивает два каталога прогонов и относит каждое различие к одному из классов:

- `data` (`dataset`) — quality-метрики несравнимы;
- `config` (snapshot сценария, `index`, `rerank`) — настройка RAG, то, что исследуем;
- `env` (`env`, `index.build_env`) — окружение: часть разницы метрик может быть его эффектом; различия `system`, `compute` и `services` делают latency несравнимой.

Разница метрик — эффект настройки RAG, если различий `data` и `env` нет. Зависимости фиксирует `uv.lock` того коммита, что записан в `env.commit`.

### 6. REST API

Единый вход в стенд для веб-интерфейса, будущего CLI-клиента и скриптов (контракт —
[docs/designs/ragglet-28-rest-api.md](docs/designs/ragglet-28-rest-api.md)):

```bash
uv run ragglet serve
```

Сервер слушает `http://127.0.0.1:8000`; OpenAPI и Swagger UI — `/v1/openapi.json` и
`/v1/docs`. Авторизации нет, поэтому наружу сервер открывается только явно
(`RAGGLET_HOST=0.0.0.0`). Настройки — переменные `RAGGLET_*` или файл `.env` в рабочем
каталоге; умолчания работают из корня репозитория:

| Переменная | По умолчанию | Что задаёт |
|---|---|---|
| `RAGGLET_HOST`, `RAGGLET_PORT` | `127.0.0.1`, `8000` | адрес сервера (или флаги `--host`, `--port`) |
| `RAGGLET_QDRANT_ENDPOINTS` | `http://localhost:6333` | Qdrant через запятую: стенд и список коллекций |
| `RAGGLET_REDIS_URL` | `redis://localhost:6379/0` | Redis для страницы стенда; пусто — не показывать |
| `RAGGLET_SCENARIOS_DIR`, `RAGGLET_INDEX_PROFILES_DIR` | `configs/scenarios`, `configs/indexes` | библиотека сценариев и профилей |
| `RAGGLET_ARTIFACTS_DIR` | `results/runs` | куда пишутся прогоны (`run.artifacts_dir` из запроса игнорируется) |
| `RAGGLET_BUILDS_DIR` | `results/builds` | каталоги сборок индексов: файлы задачи и лог |
| `RAGGLET_DATASET_ROOTS` | `datasets` | где искать локальные датасеты для выбора в UI |
| `RAGGLET_UI_DIR` | — | собранный веб-интерфейс, раздаётся с того же адреса |
| `RAGGLET_CANCEL_GRACE_S` | `10` | отмена задачи и остановка сервера: SIGTERM, через столько секунд — SIGKILL |

Основные endpoints: `GET /v1/stand`, `GET /v1/indexes`, `GET /v1/datasets`, библиотека
`/v1/scenarios` и `/v1/index-profiles` (YAML-текст как есть, версия — `etag`, правка поверх
чужой — 409), `GET /v1/meta/scenario`. Проверка и запуск — `POST /v1/runs` и
`POST /v1/builds`: конфиг передаётся содержимым, `dry_run: true` отвечает синхронно 200 или
422, иначе 202 с id прогона (`<name>_<UTC>`, при совпадении `-2`). Любая ошибка — JSON
`{title, hint, issues: [{code, severity, path, where, message}]}`, где `path` — поле
конфига (`metrics.quality.k`), `where` — место вне конфига (`qrels/test.tsv:7`).

**Реестр прогонов** ([docs/designs/ragglet-30-run-registry.md](docs/designs/ragglet-30-run-registry.md)).
Запись прогона — `run.json` в его каталоге `results/runs/<id>/`, рядом с артефактами; базы
нет, сервер при старте сканирует каталог. Каталоги без `run.json` (прогоны из CLI и старые)
попадают в реестр как импортированные (`origin: import`): `completed`, если есть
`summary.aggregate.csv`, иначе `failed` / `RunIncomplete`; в них сервер ничего не пишет.
Записи `queued` / `running`, оставшиеся после падения сервера, при старте становятся
`failed` / `ServerRestart`. Удалить прогон — удалить его каталог.

- `GET /v1/runs` — список с колонками сводки из `summary.aggregate.json` (`ndcg`, `recall`
  при `k`, `p50_ms`, `p95_ms`, `qps`, `timeout_rate`; нет колонки в summary — `null`);
  фильтры `state`, `scenario`, `scenario_id`, `dataset`, `collection`, `tag`, `since`,
  `until`, `q`; `sort` (`-ndcg` — по убыванию); `offset`, `limit`. `GET /v1/builds` — то же
  для сборок.
- `GET /v1/runs/{id}?include=summary,manifest,scenario` — статус и тяжёлые поля;
  `scenario_file` говорит, изменился ли сценарий библиотеки после прогона.
- `GET /v1/runs/{id}/queries` — per-query повтора (`repeat`) с фильтрами `outcome`,
  `miss_at` (промах при k), `min_total_ms`, `q` и `first_relevant_rank`;
  `GET /v1/runs/{id}/queries/{qid}` — один запрос: выдача с рангами, релевантностью,
  rerank-score и попаданием в контекст.
- `GET /v1/runs/{id}/artifacts`, `…/artifacts/{name}`, `…/artifacts.zip` — файлы как есть.

```bash
curl -s 'localhost:8000/v1/runs?scenario=nfcorpus_hf&sort=-ndcg&k=10' | jq '.items[] | {id, ndcg, p95_ms, qps}'
```

Проверка через curl:

```bash
jq -n --rawfile s configs/scenarios/baseline_qdrant.yaml '{scenario: $s, dry_run: true}' | curl -s -X POST localhost:8000/v1/runs -H 'content-type: application/json' -d @-
```

**Задачи.** Прогоны и сборки, поставленные через API, выполняются по одной на стенд, по
очереди, каждая в свежем процессе `python -m ragglet.jobs.worker <каталог задачи>`
(дизайн — [docs/designs/ragglet-29-job-manager.md](docs/designs/ragglet-29-job-manager.md)).
Статус и прогресс — `GET /v1/runs/{id}`, очередь — `GET /v1/tasks`, отмена —
`POST /v1/runs/{id}/cancel`, лог — `GET /v1/runs/{id}/log?offset=`; то же у `/v1/builds`.
Сервер и воркер общаются файлами в каталоге задачи (`results/runs/<id>/`,
`results/builds/<id>/`):

| Файл | Что в нём |
|---|---|
| `task.json` | что выполнять: тип, id, конфиг; пишет сервер при постановке |
| `task.log` | stdout и stderr воркера: трейсбек, таблица проблем preflight, вывод библиотек |
| `progress.json` | `{stage, done, total, first_at, updated_at}`, не чаще раза в секунду |
| `result.json` | итог: `completed`, `failed` с ошибкой или `cancelled`, последний прогресс |

Если процесс умер, не записав `result.json` (OOM, segfault), задача становится `failed` с
`WorkerCrashed` и последними строками лога. При остановке сервера текущая и ожидающие
задачи становятся `failed` с `ServerShutdown`; если сервер упал, воркер сам завершается
через несколько секунд.

Упавшую задачу можно повторить руками, без сервера, для отладки:

```bash
uv run python -m ragglet.jobs.worker results/runs/<id>
```

Воркер прочитает `task.json` и выполнит задачу, вывод — в терминал. Прогон, у которого
уже есть `scenario.snapshot.json`, ядро повторить не даст: это защита от смеси
артефактов двух прогонов. До ragglet-37 CLI (`ragglet bench run`) вызывает ядро
напрямую, в обход очереди.

Интеграционный тест с настоящим ядром (`pytest -m integration`) нужен Qdrant на
`localhost:6333`; без него тест пропускается.

### 7. Веб-интерфейс

SPA в каталоге `web/` (React + TypeScript + Vite) — клиент того же REST API: стенд, библиотека
сценариев и профилей, редактор сценария с проверкой, запуск и страница прогона (дизайн —
[docs/designs/ragglet-31-web-ui-app.md](docs/designs/ragglet-31-web-ui-app.md)). Нужен
Node.js 22 или новее.

```bash
cd web && npm ci && npm run build      # → web/dist
cd .. && RAGGLET_UI_DIR=web/dist uv run ragglet serve
```

UI откроется на `http://127.0.0.1:8000/` (тот же адрес, что и API). `RAGGLET_UI_DIR` можно
записать в `.env`. Каталог должен существовать: без сборки сервер не стартует.
В шапке — тема (светлая / тёмная / как в системе) и язык интерфейса (RU / EN); выбор
запоминается в браузере.

Разработка — API и Vite-сервер рядом, `/v1` проксируется на API (`RAGGLET_API_URL`, по
умолчанию `http://127.0.0.1:8000`):

```bash
uv run ragglet serve
cd web && npm run dev                  # → http://localhost:5173
```

| Команда (в `web/`) | Что делает |
|---|---|
| `npm test` | unit- и компонентные тесты (Vitest) |
| `npm run typecheck`, `npm run lint` | `tsc` и ESLint |
| `npm run gen:api` | перегенерировать `src/api/schema.d.ts` из OpenAPI сервера (через `uv run`) после правок API |
| `npm run check` | `gen:api` + проверка, что типы клиента свежие, + `tsc` + ESLint |

## Конфиги

Подробный справочник по доступным параметрам, реальной поддержке и конфликтам находится в:

```text
docs/config-reference.yaml
```

Основные файлы:

- `configs/indexes/demo.yaml` — профиль индексации;
- `configs/scenarios/baseline_qdrant.yaml` — baseline vector retrieval;
- `configs/scenarios/baseline_qdrant_fanout_two_vec.yaml` — fanout/hybrid retrieval с кэшем.

Ключевые правила:

- `chunk_overlap` должен быть меньше `chunk_size`; иначе профиль не пройдет валидацию;
- `chunking.method: none` индексирует документ целиком (как в BEIR); `chunk_size`/`chunk_overlap` с ним запрещены;
- `qrels` ссылается на `corpus._id` (document-level); при индексации chunk получает `doc_id` и внутренний `chunk_id`;
- `metrics.quality` задает `k` и метрики уровней: `doc` (по умолчанию `ndcg`, `recall`, `hit`, `mrr`) и `chunk` (по умолчанию выключен; включайте, если сценарий варьирует chunking или context). `recall@k` — доля найденных релевантных документов, `hit@k` — найден ли хотя бы один, `ndcg@k` — качество порядка с учётом оценок qrels; подробно в [docs/metrics.md](docs/metrics.md). Каждое `k` не должно превышать глубину финального списка (`retrieval.top_k`, при rerank — `output_top_n`);
- `storage.backend.type` реально поддерживает только `qdrant`;
- `rerank` поддерживает локальный `cross_encoder` provider из `sentence-transformers`;
- `context` поддерживает сбор retrieved/reranked чанков по политикам `top_n`, `threshold` и `adaptive`;
- `generation` поддерживает `noop` и `openai_compatible`; при `enabled: true` question/context могут быть отправлены во внешний provider;
- `run.load` задает, когда запускать запросы. Модель `closed` держит `concurrency` запросов одновременно в работе, от эмбеддинга до ответа LLM (по умолчанию 1). Сетевые этапы идут конкурентно, а локальные модели на CPU (эмбеддинг, BM25, rerank) обрабатывают один вызов за раз, и ожидание своей очереди видно в `lat_*_ms`. Своя схема подачи запросов — класс в `src/ragglet/bench/load.py` и строка в `LOAD_MODELS`;
- каждый повтор (`run.repeats`) начинается с пустого кэша: кэш retrieval очищается, а прогон удаляет ключи Redis, которые записал сам. Чтобы стартовать с тёплого кэша эмбеддингов, запустите второй прогон с тем же `cache.embedding.namespace` (только с `repeats: 1`). Попал ли запрос в кэш, видно по колонкам `emb_cache` и `ret_cache` в `per_query`, доля попаданий и задержки попаданий и промахов — в summary ([docs/metrics.md](docs/metrics.md#кэш)).

## Полезные команды

Показать CLI help:

```bash
uv run ragglet --help
uv run ragglet index --help
uv run ragglet bench --help
uv run ragglet bench run configs/scenarios/baseline_qdrant.yaml --dry-run
```

Остановить сервисы:

```bash
docker compose -f docker/docker-compose.yaml --profile qdrant down
```

Остановить сервисы и удалить volume с данными:

```bash
docker compose -f docker/docker-compose.yaml --profile qdrant down -v
```

## Частые проблемы

Если `bench run` пишет `Index is not compatible with scenario` — индекс собран из другого корпуса (например, после правки `corpus.jsonl`), другой моделью, в другом storage layout или сборка не завершилась. Пересоберите индекс: `ragglet index build <index_profile.yaml>`.

Чтобы зафиксировать версию HF-датасета или модели, скопируйте ревизию из `manifest.json` в конфиг: `dataset.revision` (сценарий и index profile), `embedding.revision` (index profile) и `rerank.revision` (сценарий). У local-датасета ревизии нет: его версия — хэш содержимого.

Предупреждение `index.not_green` означает, что Qdrant ещё оптимизирует коллекцию (обычно сразу после `index build`): latency и recall такого прогона нерепрезентативны. Дождитесь статуса `green` и запустите заново.

Если индекс строится слишком долго или создается неожиданно много чанков, проверьте `chunk_size` и `chunk_overlap` в index profile. Схема уже запрещает `chunk_overlap >= chunk_size`, но слишком маленький `chunk_size` все равно может сделать эксперимент нерепрезентативным.

Если fanout-сценарий падает на кэше, проверьте, что Redis запущен и доступен по endpoint из scenario.

Если `docker compose --profile app` не собирается, это ожидаемо для текущего состояния: app container profile зарезервирован, а рабочий путь запуска сейчас — локальный CLI через `uv run` плюс Qdrant/Redis из compose.

## Научная ценность

Ragglet можно использовать как компактную экспериментальную платформу для:

- сравнения стратегий поиска в RAG;
- анализа компромисса между качеством и временем отклика;
- подготовки воспроизводимых вычислительных экспериментов для НИР, ВКР и конференционных докладов.

## Ограничения текущей версии

- основной поддерживаемый backend для экспериментов — `Qdrant`;
- генерация ответа LLM пока не является завершенным модулем;
- реестр прогонов пока живёт в памяти сервера: после рестарта API не видит прогоны, поставленные раньше, хотя их каталоги на месте (ragglet-30);
- проект ориентирован на исследование retrieval-контура, а не на промышленный production-ready пайплайн.

## Архитектура

```mermaid
flowchart LR
    U["Пользователь / CLI"] --> CLI["Typer CLI: ragglet"]
    CLI --> IDX["index build"]
    CLI --> BENCH["bench run"]

    IDX --> IPROF["Index profile YAML"]
    IPROF --> PARSE["BEIR corpus (local / HF)"]
    PARSE --> CHUNK["Чанкинг документов"]
    CHUNK --> EMB["SentenceTransformer Embedder"]
    EMB --> QI["Qdrant Indexer"]
    QI --> QDRANT[("Qdrant")]

    BENCH --> SCN["Scenario YAML"]
    SCN --> DATA["BEIR queries + qrels"]
    SCN --> CACHE["CacheManager"]
    SCN --> RET["Retrieval Engine"]

    CACHE --> REDIS[("Redis embedding cache")]
    CACHE --> LRU["LRU retrieval cache"]

    DATA --> LOOP["Query loop"]
    LOOP --> EMBQ["Embedding запроса"]
    EMBQ --> RET

    RET --> BACK["BackendRegistry"]
    BACK --> VEC["Vector search"]
    BACK --> KW["BM25 keyword search"]

    VEC --> QDRANT
    QDRANT --> KW

    RET --> MERGE["Merge: RRF / weighted_sum / interleave"]
    MERGE --> METRICS["nDCG@K / Recall@K / Hit@K / MRR@K / latency / throughput / timeout / cache hit rate"]
    METRICS --> RES["results/ artifacts"]

```
