# Аудит S0/S3/S4 и план дополнительных экспериментов

Дата аудита: 2026-09-12. Состояние репозитория: commit `b7be394`, ветка
`master`, рабочее дерево до аудита было чистым. Во время аудита benchmark и
индексация не запускались.

## 1. Общая экспериментальная база

Все три сценария используют один и тот же набор исходных условий:

- dataset: `data/queries/truth_manifest.jsonl` — 100 непустых записей,
  100 уникальных `id`, 100 уникальных текстов запросов;
- в каждой записи сейчас ровно один relevant `external_id`; всего 50 уникальных
  relevant id;
- corpus: `datasets/demo/corpus/` — 50 файлов;
- индекс: профиль `configs/indexes/demo.yaml`, Qdrant collection `chunks_v1`;
- chunking: посимвольные chunks по 800 символов с overlap 120;
- embeddings: `all-MiniLM-L6-v2`, `normalize: true`;
- retrieval: single-source Qdrant, `top_k: 30`, `ef_search: 64`, без rerank;
- generation во всех трех сценариях выключена.

Профиль индекса сохраняет в metadata Qdrant модель/нормализацию и параметры
chunking. Перед benchmark runtime проверяет только соответствие embedding model
и `normalize`; hash корпуса, dataset и содержимого индекса пока не фиксируется.

## 2. Существующие сценарии

### S0 — baseline

Config: `configs/scenarios/s0_baseline_rag.yaml`.

Запуск:

```bash
python3 ragglet run configs/scenarios/s0_baseline_rag.yaml --dry-run
python3 ragglet run configs/scenarios/s0_baseline_rag.yaml
```

Особенности:

- 5 полных проходов dataset;
- `shuffle: true`, порядок детерминированно выводится из `seed: 42`;
- cache, context, rerank и generation выключены;
- качество считается по `retrieved_ids`.

### S3 — adaptive context

Config: `configs/scenarios/s3_adaptive_context.yaml`.

Запуск:

```bash
python3 ragglet run configs/scenarios/s3_adaptive_context.yaml --dry-run
python3 ragglet run configs/scenarios/s3_adaptive_context.yaml
```

Отличия от S0:

- context включен;
- `policy: adaptive`;
- `max_tokens: 1200`;
- `threshold: 0.15`;
- в latency summary добавлен этап `context`;
- качество считается по `context_ids`, а не по исходным `retrieved_ids`.

Алгоритм adaptive selection:

1. Берутся до 30 retrieval-кандидатов в исходном ранговом порядке.
2. Для каждого запроса их raw Qdrant score min-max-нормализуется внутри этого
   top-30: `(score - min_score) / (max_score - min_score)`.
3. Остаются chunks с normalized score `>= threshold`.
4. Они последовательно укладываются в `max_tokens`; chunk, не помещающийся в
   оставшийся budget, пропускается.
5. Если после фильтрации ничего не осталось, runtime пытается взять первый
   retrieval-кандидат.

Текущий `estimate_tokens` — число whitespace-separated слов, а не tokenizer
модели. Если первый chunk больше бюджета, он все равно включается. При равных
scores все normalized scores становятся `0`; при положительном threshold
срабатывает fallback на первый chunk.

### S4 — cache

Config: `configs/scenarios/s4_cache.yaml`.

Запуск:

```bash
python3 ragglet run configs/scenarios/s4_cache.yaml --dry-run
python3 ragglet run configs/scenarios/s4_cache.yaml
```

Особенности:

- 5 полных проходов dataset, `shuffle: false`;
- Redis embedding cache: `redis://localhost:6379/0`, TTL 86400 секунд;
- in-process retrieval LRU: 10000 записей;
- context, rerank и generation выключены;
- качество считается по `retrieved_ids`.

Embedding cache key зависит от текста запроса, embedding model и `normalize`.
Redis сохраняется между прогонами (каждый идёт в свежем процессе воркера) и рестартами контейнера с volume.
Retrieval LRU живет только внутри одного прогона (`run`), но сохраняется между его
repeats. Его key зависит от имени сценария, текста запроса, `top_k`, strategy,
sources и merge, но не содержит collection/index fingerprint.

Важно: query embedding вычисляется/читается из Redis до проверки retrieval LRU.
Поэтому retrieval-cache hit сам по себе не устраняет embedding stage.

Текущий S4 смешивает состояния cache: retrieval LRU холодный в repeat 0 и
теплый в repeats 1–4. Состояние Redis перед repeat 0 не контролируется, поэтому
S4 не является воспроизводимым разделением cold/warm между независимыми
запусками.

## 3. Метрики: что доступно сейчас

| Требование | Per invocation | Per repeat | Aggregate over repeats | Ограничение |
|---|---:|---:|---:|---|
| Recall@k | Да: `recall@k` | Да: mean | Да: mean/std/min/max repeat means | Реализация фактически binary Hit@k. На текущем dataset с одним relevant id это равно Recall@k; при нескольких relevant id — уже нет. |
| MRR@k | Да: `mrr`, cutoff из `mrr_at_k` | Да: `mrr_mean` | Да | Имя per-query поля не содержит cutoff. |
| Latency | Да: raw latency всех этапов | p50/p95/p99 выбранных этапов | mean/std/min/max от repeat-percentiles | Нет p50/p95 для каждого `qid` по repeats; aggregate p95 не является percentile объединенного набора запросов. |
| Chunks после context selection | Да: `context_chunks` | Только mean | mean/std/min/max repeat means | Для S0/S4 всегда 0, потому что context выключен. |
| Размер контекста | `context_tokens` | Только mean | mean/std/min/max repeat means | Это число слов в chunk texts; labels metadata и separators не учитываются. Characters отсутствуют. |
| Cache hit/miss | Нет отдельных событий | Только `emb_hit_rate`, `ret_hit_rate` | mean/std/min/max repeat rates | Per-query поля — накопительная доля после текущего запроса. Summary ошибочно усредняет накопительные доли вместо `hits / (hits + misses)`. Точные counts не сохраняются. |

Смысл legacy `recall@k` и `mrr` зависит от последней включенной стадии:
retrieval для S0/S4, context selection для S3, rerank для S2. Для статьи это
нужно развести на stage-specific метрики, иначе сравнение S0 и S3 двусмысленно.

Один запрос имеет одно измерение latency за invocation. P50/p95 «для запроса»
можно определить только по повторным invocation одного `qid`; текущий runtime
не создает такой grouped artifact, хотя его можно постфактум получить из всех
`per_query.repeat*.parquet`.

## 4. Создаваемые артефакты

Каждый успешный run пишет в
`results/runs/<scenario_name>_<UTC timestamp>/`:

- `scenario.snapshot.json` и `scenario.snapshot.yaml`;
- `per_query.repeat<N>.csv` и/или `.parquet`, если `save_per_query: true`;
- `summary.csv` и `summary.json` — одна запись на repeat;
- `summary.aggregate.csv` и `summary.aggregate.json` — статистика по repeat
  summaries.

`output.format` влияет только на per-query файлы; summaries всегда CSV+JSON.
Не сохраняются git revision/dirty state, hashes dataset/corpus/lockfile,
фактические index metadata/point count, версии сервисов, hardware/OS,
cache state, warmup ids и признак завершенности run. Если run падает после
создания каталога, могут остаться только snapshots.

В репозитории есть полные сохраненные runs S0, S3 и S4 от июня 2026 года.
Snapshot S0 в сохраненном run запрашивал только Recall@5/10, тогда как текущий
S0 YAML запрашивает Recall@1/3/5/10/30; старые результаты нельзя автоматически
считать результатом текущего config только по имени scenario.

## 5. Доступные экспериментальные параметры

Adaptive context — прямые факторы:

- `context.enabled`;
- `context.policy`: `top_n`, `threshold`, `adaptive`;
- `context.max_tokens`;
- `context.threshold` — raw score для `threshold`, per-query min-max normalized
  score для `adaptive`;
- `context.include_metadata` — меняет rendered context, но сейчас не
  `context_tokens`.

Retrieval `top_k`, rerank и scores также косвенно меняют множество кандидатов,
но в первом sweep их следует зафиксировать.

Cache — прямые факторы:

- `cache.enabled`;
- наличие/отсутствие embedding и retrieval layers независимо;
- Redis endpoint и `ttl_seconds`;
- LRU `size`.

Сейчас нет параметров cache namespace, controlled reset, preload/warmup fraction,
cache bypass для warmup, occupancy/eviction и разделения warmup/measurement.

## 6. Решение для статьи с горизонтом два дня

Для текущей статьи принимается short track: **runtime Ragglet не изменяется**.
Обязательный объем разработки ограничивается:

1. несколькими scenario YAML, полученными копированием S0/S3/S4 и изменением
   только экспериментального фактора;
2. одним offline postprocessing script, читающим существующие
   `per_query.repeat*.parquet` и summaries.

Текущих per-query артефактов достаточно, потому что они уже содержат
`truth_ids`, `retrieved_ids`, `context_ids`, `context_text`, `context_tokens`,
`context_chunks`, raw stage latencies и cumulative cache rates. Поэтому
stage-specific quality, characters и нужные агрегаты можно вычислить после run,
не добавляя работу в измеряемый pipeline.

Осознанные ограничения short track:

- `context_tokens` публикуется как приблизительное число слов, а не tokenizer
  tokens; дополнительно offline считается число characters;
- per-query p95 по пяти repeats является неустойчивой описательной оценкой;
  основной latency result — p50/p95 по всем invocation внутри condition;
- cache hits/misses восстанавливаются только для специально организованных
  однородных cold/warm фаз, где rates обязаны быть соответственно 0 и 1;
- полная provenance automation и общая cache instrumentation откладываются.

## 7. Минимальная matrix A — adaptive context

Primary comparison — static `top_n` против трех заранее выбранных уровней
adaptive selection. S0 используется как retrieval-only reference, но не как
единственный control для context selection.

| Cell | Context | Policy | Threshold | Max tokens | Назначение |
|---|---:|---|---:|---:|---|
| A-S0 | off | — | — | — | Retrieval-only reference/ceiling |
| A0 | on | `top_n` | — | 1200 | Static context control |
| A1 | on | `adaptive` | 0.15 | 1200 | Текущий S3, слабый отбор |
| A2 | on | `adaptive` | 0.30 | 1200 | Средний отбор |
| A3 | on | `adaptive` | 0.50 | 1200 | Сильный отбор |

`adaptive threshold=0.00` выполняется только как smoke-инвариант: при тех же
candidates результат должен совпасть с `top_n`. Полный article run для него не
нужен. Thresholds 0.15/0.30/0.50 фиксируются до новых результатов и не
корректируются по ним.

Во всех article cells фиксируются corpus, truth dataset, один физический index,
embedding model/normalize, `top_k: 30`, `ef_search: 64`, отсутствие rerank,
cache и generation, seed, repeats, timeouts и окружение. Варьируется только
context policy/threshold. Для short track достаточно пяти repeats, уже принятых
в S0/S3.

Основные метрики:

- context Recall@1/3/5/10/30 и MRR@30;
- context chunks, characters и отдельно approximate whitespace word count;
- context и total latency p50/p95;
- retrieval Recall/MRR как guardrail: они должны совпадать между cells;
- timeout rate как технический guardrail.

## 8. Минимальная matrix B — cache без изменений runtime

Используется Redis endpoint `redis://localhost:6379/15`. Перед каждым
независимым cache trial удаляются **только ключи namespace эксперимента**
`emb:all-MiniLM-L6-v2:1:*`; `FLUSHDB` не нужен. Retrieval LRU создается пустым
при старте нового процесса прогона.

Важно для этого стенда: benchmark обращается к host Redis на `localhost:6379`,
а Redis из `docker/docker-compose.yaml` не публикует порт на host. Поэтому
`docker compose exec redis ... FLUSHDB` очищает другой экземпляр и не создает
cold phase для benchmark.

Cache scenario выполняет два repeats с `shuffle: false`:

| Фаза | Данные | Ожидаемые embedding/retrieval rates | Интерпретация |
|---|---|---:|---|
| repeat 0 | 100 уникальных запросов | 0 | Cold cache |
| repeats 0+1 | те же 100 запросов дважды | 0.5 | Partially repeated workload |
| repeat 1 | второй проход тех же запросов | 1 | Warm cache |

Это реальные последовательные фазы одного run, а не синтетическая
интерполяция. Cold и warm сравниваются парно. Для оценки вариативности
выполняются пять независимых trials; перед каждым Redis DB 15 снова очищается.

Отдельно запускается cache-disabled control с `shuffle: false` и теми же
запросами. Разделять embedding-only и retrieval-only cache в текущей статье не
обязательно: вывод формулируется про совместный cache stack S4. Такое
разделение остается возможным follow-up, но не входит в двухдневный объем.

Offline analyzer обязан проверить:

- все строки repeat 0 имеют `emb_hit_rate == ret_hit_rate == 0`;
- все строки repeat 1 имеют `emb_hit_rate == ret_hit_rate == 1`;
- в каждой фазе ровно 100 ожидаемых `qid` без пропусков;
- quality cold и warm совпадает.

Если любое из этих условий нарушено, соответствующий trial не включается в
статью до выяснения причины.

## 9. Единственная необходимая доработка: offline analyzer

Предлагаемый `scripts/analyze_article_runs.py` не импортируется benchmark
runtime и не влияет на latency. Он должен:

1. читать Parquet из выбранных run directories;
2. считать retrieval и context Recall@k/MRR@k отдельно из сохраненных ids;
3. считать `context_chars = len(context_text)` и использовать существующие
   `context_tokens` как явно подписанный word estimate;
4. считать per-condition pooled p50/p95;
5. для cache trials проверять фазовые инварианты выше и выдавать точные counts:
   cold — 0 hits/100 misses, warm — 100 hits/0 misses, combined workload —
   100 hits/100 misses;
6. проверять smoke-инвариант `adaptive threshold=0.00 == top_n`;
7. сохранять tidy CSV, JSON summary и Markdown report.

В текущем dataset у каждого запроса один relevant id, поэтому существующий
binary Hit@k численно равен Recall@k. Менять реализацию метрики до статьи не
требуется; это ограничение нужно явно указать в методике.

## 10. Команды

### Подготовка текущего стенда

Клиент `ragglet` в корне репозитория (ragglet-37) сам поднимает Qdrant и Redis через
Docker, готовит окружение и запускает сервер; прогоны идут через API, по одному на стенд:

```bash
python3 ragglet start
python3 ragglet status
```

Индекс следует построить один раз перед всей серией. Команда пересоздает
collection, поэтому ее не нужно повторять между cells:

```bash
python3 ragglet build demo
```

После построения индекса:

```bash
python3 ragglet run configs/scenarios/s0_baseline_rag.yaml --dry-run
python3 ragglet run configs/scenarios/s3_adaptive_context.yaml --dry-run
python3 ragglet run configs/scenarios/s4_cache.yaml --dry-run
```

### После добавления только YAML и analyzer

Предлагаемая структура:

```text
configs/experiments/article/adaptive/a_s0.yaml
configs/experiments/article/adaptive/a0_top_n.yaml
configs/experiments/article/adaptive/a1_t015.yaml
configs/experiments/article/adaptive/a2_t030.yaml
configs/experiments/article/adaptive/a3_t050.yaml
configs/experiments/article/adaptive/smoke_t000.yaml
configs/experiments/article/adaptive/smoke_top_n.yaml
configs/experiments/article/cache/b0_disabled.yaml
configs/experiments/article/cache/b1_cache_phases.yaml
scripts/analyze_article_runs.py
```

Прогоны пишутся в общий реестр (`results/runs/<id>`), а серию отличает тег: все прогоны
серии запускаются с одним `--tag`, по нему же анализатор их и находит.

```bash
SERIES=article-$(date +%Y%m%d)
```

До первого benchmark run фиксируется окружение:

```bash
uv run python scripts/analyze_article_runs.py capture-manifest \
  --output results/article/experiment-manifest.json
```

Smoke validation (каталоги прогонов `run` печатает в строке `Artifacts`):

```bash
python3 ragglet run configs/experiments/article/adaptive/smoke_top_n.yaml --tag "$SERIES"
python3 ragglet run configs/experiments/article/adaptive/smoke_t000.yaml --tag "$SERIES"
uv run python scripts/analyze_article_runs.py check-smoke \
  --top-n-run results/runs/<article_smoke_top_n_TIMESTAMP> \
  --adaptive-run results/runs/<article_smoke_adaptive_t000_TIMESTAMP>
```

Adaptive cells остаются обычными scenarios:

```bash
python3 ragglet run configs/experiments/article/adaptive/a_s0.yaml --tag "$SERIES"
python3 ragglet run configs/experiments/article/adaptive/a0_top_n.yaml --tag "$SERIES"
python3 ragglet run configs/experiments/article/adaptive/a1_t015.yaml --tag "$SERIES"
python3 ragglet run configs/experiments/article/adaptive/a2_t030.yaml --tag "$SERIES"
python3 ragglet run configs/experiments/article/adaptive/a3_t050.yaml --tag "$SERIES"
```

Cache-disabled control выполняется пятью задачами, каждая в свежем процессе воркера
(ragglet-29), поэтому прогоны независимы:

```bash
for trial in 1 2 3 4 5; do
  python3 ragglet run configs/experiments/article/cache/b0_disabled.yaml --tag "$SERIES"
done
```

Пять независимых cold/warm trials. Перед каждым удаляется только namespace
embedding cache этого benchmark; остальные ключи DB не затрагиваются. `run` ждёт конца
прогона, поэтому очистка идёт строго между прогонами; Ctrl-C отменяет текущий прогон и
останавливает цикл:

```bash
for trial in 1 2 3 4 5; do
  uv run python -c "import redis; r=redis.Redis.from_url('redis://localhost:6379/15'); ks=list(r.scan_iter(match='emb:all-MiniLM-L6-v2:1:*')); print('keys', len(ks)); r.unlink(*ks) if ks else None"
  python3 ragglet run configs/experiments/article/cache/b1_cache_phases.yaml --tag "$SERIES"
done
```

Postprocessing берёт завершённые прогоны серии из реестра сервера (`GET /v1/runs?tag=…`);
прогон, который нужно исключить, — `--exclude <id>` (он попадёт в аномалии отчёта):

```bash
uv run python scripts/analyze_article_runs.py analyze \
  --tag "$SERIES" \
  --output-dir results/article \
  --manifest results/article/experiment-manifest.json
```

## 11. Отложенный backlog, не блокирующий статью

После статьи можно добавить настоящие per-request cache flags, exact runtime
cache counters, stage-specific metric names, run manifest, hashes dataset/index,
cache namespace/warmup API, строгую config validation и расширенный sweep. Эти
изменения полезны для долгосрочного стенда, но не нужны для получения текущего
минимального воспроизводимого набора данных.

## 12. Статус реализации и проверок (2026-09-12)

- Environment manifest сохранён до benchmark runs: `results/article/experiment-manifest.json`.
- Все article-конфиги прошли live `bench validate` на существующем Qdrant index:
  100 truth ids, 53 indexed chunks, missing/duplicates — 0.
- Smoke-инвариант `adaptive threshold=0.00 == top_n` строго пройден на 10
  запросах; latency не сравнивалась.
- Выполнены все adaptive cells и по пять независимых B0/B1 cache trials.
- Analyzer принял 4500 invocation: уникальные ключи строк, 0 timeout, одинаковый
  query ordering, B1 cold=0% hits и warm=100% hits.
- Четыре предварительных B1 runs, где был очищен не тот Redis, исключены и
  сохранены в `results/article/rejected` как audit trail.
- Unit tests: 15 passed; Ruff и Python compile check пройдены.
- Benchmark runtime в `src/ragglet` не изменялся.
- Итоги: `results/article/tidy.csv`, `results/article/summary.json` и
  `results/article/report.md`.

## 13. Актуализация под BEIR и doc-level метрики (2026-10-17, ragglet-8/ragglet-16)

Результаты из раздела 12 получены на `b7be394`. После ragglet-8 (BEIR) и ragglet-16
(метрики) конфиги и analyzer в том виде не загружались, поэтому они переведены на
текущий контракт. Экспериментальные факторы не менялись.

- Конфиги: `dataset` → `source: local`, `id: datasets/demo`, `split: test` (тот же
  демо-набор, сконвертированный в BEIR); `metrics.quality` → `k: [1, 3, 5, 10, 30]`,
  `doc: [recall, hit, mrr]`. В ячейках с включённым context добавлен chunk-level
  `[mrr, precision]`.
- Analyzer: hash датасета считается по `datasets/demo/{corpus.jsonl,queries.jsonl,qrels/test.tsv}`.
  Retrieval/context Recall@k и MRR@30 считаются по `retrieved_doc_ids` и `final_doc_ids`
  (при включённом context это context), а не по chunk id. Smoke-сравнение берёт колонки
  `doc_*@k`/`chunk_*@k`.
- Сопоставимость со статьёй: тогда truth был chunk-level (`<file>#0`), а `recall@k` в
  runtime был Hit@k. Теперь truth doc-level: засчитывается любой чанк релевантного
  документа. В демо 50 документов дают 53 чанка, поэтому расхождение ожидается
  небольшим, но 1:1 старые и новые числа не сравниваются. При одном релевантном документе
  на запрос `recall@k == hit@k`.
- Команды из раздела 10 остаются в силе. Индекс собирается тем же
  `configs/indexes/demo.yaml`, который теперь читает BEIR.
