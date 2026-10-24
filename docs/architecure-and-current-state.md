Ниже — единый ragglet_status.md, который можно прямо скопировать в репо и скормить Codex/Claude как “контекст проекта”. Я специально описал что уже реализовано, как устроено, как запускать, и что нужно сделать дальше.

# ragglet — status & architecture (May 2026)
## Goal
ragglet is a Python benchmarking playground for Retrieval-Augmented Generation (RAG) retrieval architectures.
Focus: quality–performance tradeoffs in distributed / multi-source retrieval:
- retrieval strategies: `single`, `fanout_merge` (async or sequential)
- merge algorithms: currently `rrf`, in progress: `weighted_sum`, `interleave`
- caching: embedding cache (Redis), retrieval cache (in-memory LRU)
- timeouts: per-query `total_ms` + per-stage (retrieval/rerank/llm) using AnyIO cancel scopes
- evaluation: document-level nDCG@k / Recall@k / R_cap@k / Hit@k / MRR@k (optional chunk-level MRR@k / Precision@k), latency percentiles, query outcomes (ok/timeout/error), repeat wall time and throughput; semantics in `docs/metrics.md`
- indexing: build vector DB index (Qdrant) from a directory corpus; payload carries `external_id` for ground truth
RAG generation can be enabled as an optional post-context stage; retrieval-only benchmarks remain the default.
---
## Repo layout (current direction)
Suggested / partially implemented modular split:
- `src/ragglet/cli/`  
  Typer CLI: `ragglet bench run ...`, `ragglet index build ...`, `ragglet serve` (serve is stub)
- `src/ragglet/config/`  
  Pydantic v2 schemas:
  - `scenario.py` for `scenario.yaml`
  - `index_profile.py` for `index_profile.yaml`
- `src/ragglet/indexing/`  
  Offline indexing pipeline:
  - parse corpus files as plain text
  - chunk text (simple recursive chunker by chars)
  - embed chunks
  - create Qdrant collection + upsert points
  - store Ragglet index metadata on the Qdrant collection
  Important: Qdrant point IDs must be UUID or uint64. We use deterministic UUIDv5 for point_id and store a human-readable `external_id` in payload.
- `src/ragglet/stores/`
  - `AsyncQdrantStore` (async client) for vector search + scrolling payloads
  - `KeywordBm25Store` (in-memory BM25) for keyword search (built from exported Qdrant payloads)
  - (sync store exists historically but async is used for correct timeouts/fanout)
- `src/ragglet/retrieval/`
  - `fanout_rrf.py`: fanout runner (parallel/sequential) + merge helpers (RRF) + deduplicate
  - `engine.py`: `retrieve_ids(...)` implements strategies `single` and `fanout_merge` for mixed source kinds
- `src/ragglet/cache/`
  - embedding cache: Redis (`RedisEmbeddingCache`)
  - retrieval cache: in-memory LRU (`LRURetrievalCache`)
  - cache manager (optional) to unify stats + get/set
- `src/ragglet/bench/`
  - dataset loader: JSONL (`load_jsonl_dataset`)
  - dataset/index validation: compare JSONL relevant[] ids with indexed payload external_id values
  - metrics: nDCG@k / Recall@k / R_cap@k / Hit@k / MRR@k / Precision@k on document and chunk levels (`metrics.py`, `evaluation.py`)
  - load models (`load.py`): registry `LOAD_MODELS`; a load model only decides when to launch each query (`closed` = N workers)
  - query loop: the core runs one query per launch (embed -> retrieval -> rerank -> context -> generation -> metrics), keeps rows in repeat order and stops on the first unexpected error
  - summarize: per-repeat summary + aggregate over repeats
  - artifacts: run folder + scenario snapshot + per-query outputs + summaries
---
## Configs
### scenario.yaml (v1)
Key sections:
- `timeouts`:
  - `total_ms`: end-to-end per-query deadline (embed + retrieval + rerank + llm)
  - `retrieval_ms`: retrieval stage deadline (single or fanout_merge incl. merge)
  - `rerank_ms`: rerank stage deadline when rerank is enabled
  - `llm_ms`: generation stage deadline when generation is enabled
- `run`:
  - `repeats`: how many full dataset runs
  - `shuffle`: shuffle dataset between repeats (deterministic using `seed`)
  - `save_per_query`: store per-query CSV/Parquet per repeat
  - `artifacts_dir`: directory where run artifacts are created
  - `load`: load model, `{type: closed, concurrency: N}` by default with N=1. Concurrency covers the whole query: Redis, Qdrant and generation calls run concurrently, while local CPU models (embedding, BM25, rerank) go through `core/lanes.py`: one call at a time in a worker thread, so the event loop keeps serving network replies. A timed-out model call finishes before the timeout fires.
- `output`:
  - `format`: `csv | parquet | both`
  - `include_debug_fields`: reserved
- `dataset`:
  - `id`: path to `data/queries/dev.jsonl` or registry key
  - `limit`, `query_field`, `truth_field`
- `storage`:
  - currently supports Qdrant backend:
    - `endpoint`, `collection`
  - `mode` reserved for future: `single | sharded | replicated`
- `embedding`:
  - SentenceTransformers model name
  - `normalize`
- `retrieval`:
  - `strategy`: `single | fanout_merge` (cascade was considered but postponed)
  - `top_k`
  - `async`: controls fanout execution:
    - `true` -> concurrent fanout (AnyIO task group)
    - `false` -> sequential execution
  - `sources`: list of sources:
    - `kind`: `vector` or `keyword` (hybrid = using both sources with merge)
    - `weight`: used by merge (RRF currently)
    - `params`: vendor-specific:
      - vector: Qdrant `ef_search` etc.
      - keyword: BM25 params `k1`, `b` used during BM25 index build
  - `merge`:
    - `method`: currently `rrf`; planned: `weighted_sum`, `interleave`
    - `rrf_k`, `deduplicate`
- `cache`:
  - `embedding`: Redis cache for query embeddings
  - `retrieval`: in-memory LRU cache for retrieval results, including retrieved ids and candidate payloads
  Notes:
  - retrieval LRU is per-process; hit rate across separate CLI runs is expected to be 0 unless repeats>1 or queries repeat.
- `metrics`:
  - `quality.k`, `quality.doc`, `quality.chunk`
  - `percentiles` from YAML for emitted latency fields
  - step latency fields measured in per-query table (embed/retrieve/rerank/context/generation/total); a step that did not complete (timed out, failed, not reached or disabled) is empty and skipped by percentiles; `lat_total` covers every query
  - `outcome` (`ok`/`timeout`/`error`) and `timeout_stage` are recorded per-query; summary stores `<outcome>_count`/`<outcome>_rate` and timeout stage distribution
  - `t_start_ms` per query (start from the repeat start); summary `wall_ms` = last query end, `throughput_qps` = answers (`ok`) per second
  - a query without an answer is a miss: empty final list, all quality metrics 0
### index_profile.yaml
Current “scenario 1” indexing (directory corpus):
```yaml
id: "demo_chunks_v1"
corpus: "datasets/demo/corpus"
parser: "plain_text"
chunking:
  method: "recursive"
  chunk_size: 800
  chunk_overlap: 120
embedding:
  model: "all-MiniLM-L6-v2"
  normalize: true
  batch_size: 64
storage:
  type: "qdrant"
  endpoint: "http://localhost:6333"
  collection: "chunks_v1"
index_params:
  distance: "cosine"
  hnsw:
    m: 16
    ef_construct: 128
```

Indexing details:

* Qdrant point_id: deterministic UUIDv5 derived from external_id (required by Qdrant: UUID or uint64)
* payload includes:
    * external_id: human-readable identifier used for ground truth + evaluation, format: "{filename_or_relpath}#{chunk_idx}"
    * doc_path, chunk_index, text
        Ground truth in data/queries/dev.jsonl references external_id strings.

⸻

How to run (local)

1) Start Qdrant (and Redis if caching enabled)

* Qdrant on http://localhost:6333
* Redis on redis://localhost:6379/0 (optional)

2) Prepare corpus

Put ~50 markdown/text files into:

* datasets/demo/corpus/
    Ensure early definitions are near the beginning if you plan to label relevant chunks as #0.

3) Build index

ragglet index build configs/indexes/demo.yaml

4) Prepare queries with ground truth (manual)

Create data/queries/dev.jsonl lines like:

{"id":"q001","question":"What is RAG?","relevant":["rag_overview_definition_and_pipeline.md#0"]}

relevant must match payload external_id.

5) Run benchmark

ragglet bench run configs/scenarios/baseline_qdrant.yaml

Artifacts:

* results/runs/<scenario>_<timestamp>/
    * scenario.snapshot.yaml/json
    * manifest.json (dataset/index identity, rerank revision, run environment)
    * per_query.repeat0.csv (if enabled)
    * summary.csv/json (per-repeat)
    * summary.aggregate.csv/json (mean/std/min/max over repeats)

⸻

What is implemented today (core)

✅ Index build for Qdrant from local corpus directory:

* chunking
* embedding
* collection recreate + upsert (UUID point IDs)
* payload includes external_id
* collection metadata includes index_profile_id, embedding model/normalize, and chunking params

✅ Retrieval strategies:

* single: one vector source (keyword single retrieval is rejected by schema as not implemented yet)
* fanout_merge: multi-source, async or sequential based on retrieval.async
    * sources can be mixed vector + keyword => hybrid retrieval
* merge: rrf with weights + deduplicate by external_id

✅ Keyword backend:

* In-memory BM25 built from exported Qdrant payloads (external_id, text, …)
* BM25 params k1/b taken from YAML keyword source params

✅ Caches:

* Redis embedding cache: query text -> embedding vector
* LRU retrieval cache: query+scenario signature -> retrieved ids + retrieved item payloads (in-process only)
* Cache stats are summarized as rates (per repeat), not as per-row arrays.

✅ Rerank:

* Optional rerank stage after retrieval when `rerank.enabled=true`
* Providers:
    * `cross_encoder`: local `sentence-transformers` CrossEncoder
    * `noop`: pass-through provider for testing and compatibility
* Uses `rerank.input_top_k` and `rerank.output_top_n`
* Emits `reranked_ids`, `final_ids`, `rerank_scores`, and `lat_rerank_ms`

✅ Context assembly:

* Optional context stage after retrieval/rerank when `context.enabled=true`
* Policies:
    * `top_n`: fill the token budget in candidate rank order
    * `threshold`: filter by raw candidate score, then fill budget
    * `adaptive`: filter by normalized candidate score, then fill budget
* Uses `context.max_tokens`, `context.threshold`, and `context.include_metadata`
* Emits `context_ids`, `context_tokens`, `context_chunks`, optional `context_text`, and `lat_context_ms`

✅ Generation:

* Optional generation stage after context assembly when `generation.enabled=true`
* Requires `context.enabled=true`
* Providers:
    * `noop`: no external call; useful for smoke tests and artifact contract checks
    * `openai_compatible`: chat-completions-compatible HTTP provider
* Uses `generation.model`, `generation.temperature`, and `generation.max_output_tokens`
* `generation.save` controls prompt, answer, and raw response persistence
* Emits `answer`, `prompt_text`, token fields, `generation_error`, and `lat_generation_ms`

✅ Timeouts:

* Async Qdrant store enables correct AnyIO timeouts.
* retrieval_ms is enforced at retrieval stage.
* total_ms is intended at per-query wrapper.
* Timeouts are handled without crashing the whole run:
    * per-query row: outcome, timeout_stage
    * summary: outcome counts/rates and timeout stage distribution

✅ Evaluation:

* document-level nDCG@k / Recall@k / R_cap@k / Hit@k / MRR@k, optional chunk-level MRR@k / Precision@k
* latency step times (lat_embed_ms, lat_retrieve_ms, lat_rerank_ms, lat_context_ms, lat_generation_ms, lat_total_ms)
* configurable latency percentiles from YAML for emitted latency steps (embed/retrieve/total)
* per repeat: wall_ms, throughput_qps, ok/timeout/error counts and rates; rules in `docs/metrics.md`

⸻

Current limitations / TODO

1. Merge methods:
    * add weighted_sum (needs score normalization for vector+BM25)
    * add interleave (round-robin + dedup)
    * keep RRF as default baseline
2. Robust timeout handling:
    * ensure total_ms wraps embed+retrieval+rerank+context+generation
    * stage-level timeout errors should not abort run; record them per-query
3. Retrieval “cascade”:
    * postponed because with pure vector sources and no early-stop policy it collapses to single or sequential fanout.
    * revisit once filters/shards/keyword+vector stop-policy exists.
4. Distributed storage modes:
    * implement storage.mode=sharded|replicated and routing policies
    * use existing fanout+merge on shards/replicas
5. Generation providers:
    * add provider-specific token/cost accounting beyond usage fields
    * add more providers if experiments require them
6. Dataset and corpus formats:
    * current indexing parser is plain_text on files.
    * later: add corpus ingestion from jsonl/tsv and proper html/pdf cleaning.

⸻

Notes from recent debugging

* Qdrant requires point IDs be UUID or uint64; human-readable IDs must be stored in payload (external_id) for evaluation.
* If `bench run` reports `Index is not compatible with scenario`: the index was built from another corpus, with another embedding model or storage layout, or the build did not finish; rebuild it. `ragglet bench validate` prints the same preflight report without running queries.
* AnyIO timeouts don’t work with sync clients; async Qdrant store is required for correct per-query and per-stage timeouts.

⸻

Desired next tasks (high priority)

* Implement merge.method=interleave and merge.method=weighted_sum with normalization.
* Add a separate merge latency timer if merge-level percentiles are needed.
* Implement storage.mode=sharded using fanout across shard backends + merge.
