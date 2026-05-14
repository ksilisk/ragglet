Ниже — единый ragglet_status.md, который можно прямо скопировать в репо и скормить Codex/Claude как “контекст проекта”. Я специально описал что уже реализовано, как устроено, как запускать, и что нужно сделать дальше.

# ragglet — status & architecture (May 2026)
## Goal
ragglet is a Python benchmarking playground for Retrieval-Augmented Generation (RAG) retrieval architectures.
Focus: quality–performance tradeoffs in distributed / multi-source retrieval:
- retrieval strategies: `single`, `fanout_merge` (async or sequential)
- merge algorithms: currently `rrf`, in progress: `weighted_sum`, `interleave`
- caching: embedding cache (Redis), retrieval cache (in-memory LRU)
- timeouts: per-query `total_ms` + per-stage (retrieval/rerank/llm) using AnyIO cancel scopes
- evaluation: Recall@k, MRR@k, latency percentiles, timeout rate
- indexing: build vector DB index (Qdrant) from a directory corpus; payload carries `external_id` for ground truth
RAG generation (LLM) is not implemented yet; the platform currently benchmarks retrieval.
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
  - metrics: Recall@k / MRR@k
  - query loop: run per-query (embed -> retrieval -> metrics) and collect per-query rows
  - summarize: per-repeat summary + aggregate over repeats
  - artifacts: run folder + scenario snapshot + per-query outputs + summaries
---
## Configs
### scenario.yaml (v1)
Key sections:
- `timeouts`:
  - `total_ms`: end-to-end per-query deadline (embed + retrieval + rerank + llm)
  - `retrieval_ms`: retrieval stage deadline (single or fanout_merge incl. merge)
  - `rerank_ms`: reserved (rerank stage not implemented yet)
  - `llm_ms`: reserved (generation not implemented yet)
- `run`:
  - `repeats`: how many full dataset runs
  - `shuffle`: shuffle dataset between repeats (deterministic using `seed`)
  - `save_per_query`: store per-query CSV/Parquet per repeat
  - `artifacts_dir`: directory where run artifacts are created
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
  - `retrieval`: in-memory LRU cache for retrieval results
  Notes:
  - retrieval LRU is per-process; hit rate across separate CLI runs is expected to be 0 unless repeats>1 or queries repeat.
- `metrics`:
  - `quality.recall_at_k`, `quality.mrr_at_k`
  - `percentiles` from YAML for emitted latency fields
  - step latency fields measured in per-query table (embed/retrieve/total)
  - `timed_out`, `timeout_stage` fields are recorded per-query; summary stores timeout rates/cnts.
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
* LRU retrieval cache: query+scenario signature -> retrieved_ids (in-process only)
* Cache stats are summarized as rates (per repeat), not as per-row arrays.

✅ Timeouts:

* Async Qdrant store enables correct AnyIO timeouts.
* retrieval_ms is enforced at retrieval stage.
* total_ms is intended at per-query wrapper.
* Timeouts are handled without crashing the whole run:
    * per-query row: timed_out, timeout_stage
    * summary: timeout rate and stage distribution

✅ Evaluation:

* Recall@k, MRR@k
* latency step times (lat_embed_ms, lat_retrieve_ms, lat_total_ms)
* configurable latency percentiles from YAML for emitted latency steps (embed/retrieve/total)

⸻

Current limitations / TODO

1. Merge methods:
    * add weighted_sum (needs score normalization for vector+BM25)
    * add interleave (round-robin + dedup)
    * keep RRF as default baseline
2. Robust timeout handling:
    * ensure total_ms wraps embed+retrieval (+ future rerank/llm)
    * stage-level timeout errors should not abort run; record them per-query
3. Retrieval “cascade”:
    * postponed because with pure vector sources and no early-stop policy it collapses to single or sequential fanout.
    * revisit once filters/shards/keyword+vector stop-policy exists.
4. Distributed storage modes:
    * implement storage.mode=sharded|replicated and routing policies
    * use existing fanout+merge on shards/replicas
5. Rerank and generation:
    * rerank stage (cross-encoder / local model) with timeout
    * generation stage (LLM) with timeout, context selection policies
6. Dataset and corpus formats:
    * current indexing parser is plain_text on files.
    * later: add corpus ingestion from jsonl/tsv and proper html/pdf cleaning.

⸻

Notes from recent debugging

* Qdrant requires point IDs be UUID or uint64; human-readable IDs must be stored in payload (external_id) for evaluation.
* If Recall/MRR are all zeros: usually mismatch between queries.relevant[] and payload external_id.
* AnyIO timeouts don’t work with sync clients; async Qdrant store is required for correct per-query and per-stage timeouts.

⸻

Desired next tasks (high priority)

* Implement merge.method=interleave and merge.method=weighted_sum with normalization.
* Add a separate merge latency timer if merge-level percentiles are needed.
* Implement storage.mode=sharded using fanout across shard backends + merge.
