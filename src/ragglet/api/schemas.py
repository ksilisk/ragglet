from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, StringConstraints, model_validator

from ragglet.api.problems import Problem, ProblemIssue, issues_out
from ragglet.config.errors import FieldError
from ragglet.jobs.protocol import ProgressInfo, TaskFailure, TaskKind, TaskState, TaskStatus
from ragglet.registry.protocol import BuildRecord, Origin, RunRecord

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
ConfigId = Annotated[str, StringConstraints(pattern=ID_PATTERN)]

# -----------------------
# Библиотека
# -----------------------


class CollectionRef(BaseModel):
    endpoint: str  # "http://localhost:6333"
    collection: str


class ConfigDocument(BaseModel):
    id: str
    yaml: str  # текст файла как есть, с комментариями
    updated_at: datetime  # mtime файла
    etag: str  # sha256 текста, первые 16 hex
    valid: bool
    problem: Problem | None = None  # почему невалиден (черновик)


class ConfigWrite(BaseModel):
    yaml: str
    etag: str | None = None  # версия, с которой начата правка; None — файла ещё нет


class ScenarioCreate(BaseModel):
    """Ровно одно из yaml / from_run."""

    yaml: str | None = None  # id берётся из name
    from_run: str | None = None  # id прогона: берётся его snapshot
    name: ConfigId | None = None  # обязательно вместе с from_run

    @model_validator(mode="after")
    def _one_source(self) -> ScenarioCreate:
        if (self.yaml is None) == (self.from_run is None):
            raise FieldError("yaml", "exactly one of yaml and from_run is required")
        if self.from_run is not None and self.name is None:
            raise FieldError("name", "name is required with from_run")
        return self


class IndexProfileCreate(BaseModel):
    yaml: str  # id берётся из поля id


class Pipeline(BaseModel):
    strategy: str
    sources: int
    rerank: bool
    context: bool
    generation: bool
    cache: bool
    concurrency: int | None
    repeats: int


class RunBrief(BaseModel):
    id: str
    state: TaskState
    enqueued_at: datetime
    ndcg_at_10: float | None = None  # doc_ndcg@10, среднее по повторам; None — нет в summary


class BuildBrief(BaseModel):
    id: str
    state: TaskState
    enqueued_at: datetime


class ScenarioItem(BaseModel):
    id: str
    updated_at: datetime
    etag: str
    valid: bool
    description: str | None = None
    tags: list[str] = []
    dataset: str | None = None  # "hf:BeIR/nfcorpus"; None у черновика без dataset
    collections: list[CollectionRef] = []  # цели storage
    pipeline: Pipeline | None = None
    last_run: RunBrief | None = None


class IndexProfileItem(BaseModel):
    id: str
    updated_at: datetime
    etag: str
    valid: bool
    dataset: str | None = None
    chunking: str | None = None  # "recursive 800/120"
    embedding_model: str | None = None
    targets: list[CollectionRef] = []
    last_build: BuildBrief | None = None

# -----------------------
# Meta, стенд, коллекции, датасеты
# -----------------------


class DatasetSourceMeta(BaseModel):
    id: str  # local | hf
    supports_revision: bool


class ScenarioMeta(BaseModel):
    json_schema: dict[str, Any]  # ScenarioConfig.model_json_schema()
    defaults: dict[str, Any]  # умолчания всех полей, включая default_factory: в схеме их нет
    load_models: dict[str, dict[str, Any]]  # run.load.type -> JSON Schema параметров модели
    dataset_sources: list[DatasetSourceMeta]
    storage_backends: list[str]  # реализованные
    retrieval_strategies: list[str]  # реализованные
    metrics: dict[Literal["doc", "chunk"], list[str]]
    latency_steps: list[str]  # стадии, для которых ядро пишет lat_*_ms


class IndexProfileMeta(BaseModel):
    json_schema: dict[str, Any]
    defaults: dict[str, Any]
    dataset_sources: list[DatasetSourceMeta]
    storage_backends: list[str]


class ServiceStatus(BaseModel):
    url: str
    available: bool
    version: str | None = None
    error: str | None = None  # "ConnectError: All connection attempts failed"
    collections: int | None = None  # только у Qdrant


class ProgressOut(BaseModel):
    stage: str
    done: int
    total: int
    first_at: datetime
    updated_at: datetime

    @classmethod
    def build(cls, progress: ProgressInfo | None) -> ProgressOut | None:
        if progress is None:
            return None
        return cls(stage=progress.stage, done=progress.done, total=progress.total,
                   first_at=progress.first_at, updated_at=progress.updated_at)


class TaskItem(BaseModel):
    kind: TaskKind
    id: str
    name: str
    state: TaskState
    position: int | None
    progress: ProgressOut | None
    enqueued_at: datetime
    started_at: datetime | None

    @classmethod
    def build(cls, status: TaskStatus) -> TaskItem:
        return cls(
            kind=status.ref.kind, id=status.ref.id, name=status.name, state=status.state,
            position=status.position, progress=ProgressOut.build(status.progress),
            enqueued_at=status.enqueued_at, started_at=status.started_at,
        )


class StandStatus(BaseModel):
    qdrant: list[ServiceStatus]  # по qdrant_endpoints из конфига сервера
    redis: ServiceStatus | None  # None — redis_url не задан
    compute: dict[str, Any]  # где будет считать модель: {"device": "mps", ...}
    system: dict[str, Any]
    commit: str | None
    python: str
    env: dict[str, bool]  # переменная из generation.api_key_env -> задана ли
    current: TaskItem | None
    queued: int


class CollectionItem(BaseModel):
    endpoint: str
    name: str
    state: Literal["ready", "optimizing", "incomplete", "foreign"]
    status: str  # green / yellow / grey / red
    points: int | None
    indexed_vectors: int | None
    segments: int
    params: dict[str, Any] | None  # None — не один безымянный вектор
    ragglet: dict[str, Any] | None  # метаданные ragglet целиком; None — их нет или они неполные
    used_by: list[str]  # сценарии библиотеки, у которых коллекция среди целей storage
    profiles: list[str]  # профили библиотеки, которые её собирают


class IndexList(BaseModel):
    endpoints: list[ServiceStatus]
    collections: list[CollectionItem]


class LocalDataset(BaseModel):
    id: str  # как писать в dataset.id: "datasets/demo"
    splits: list[str]


class HfDataset(BaseModel):
    id: str  # "BeIR/nfcorpus"
    seen_in: list[str]  # "scenario:nfcorpus_hf", "profile:…", "collection:http://localhost:6333/nfcorpus_v1"


class DatasetList(BaseModel):
    local: list[LocalDataset]
    hf: list[HfDataset]

# -----------------------
# Прогоны и сборки
# -----------------------


class RunRequest(BaseModel):
    scenario: str  # YAML-текст сценария: конфиг передаётся только содержимым
    scenario_id: ConfigId | None = None  # ссылка для реестра: из какого сценария библиотеки прогон
    dry_run: bool = False
    tags: list[str] = []  # метки прогона в реестре, сверх scenario.tags


class BuildRequest(BaseModel):
    profile: str  # YAML-текст профиля
    profile_id: ConfigId | None = None  # ссылка для реестра: из какого профиля библиотеки сборка
    dry_run: bool = False


class RunPlanOut(BaseModel):
    repeats: int
    queries_per_repeat: int
    total: int  # то же total, что у progress("queries", …)
    load: dict[str, Any]  # {"type": "closed", "concurrency": 16}


class DatasetCheck(BaseModel):
    label: str  # "hf:BeIR/nfcorpus@b5026a0e…"
    revision: str | None
    split: str
    queries: int
    corpus_sha256: str
    eval_sha256: str


class TargetCheck(BaseModel):
    id: str
    endpoint: str
    collection: str
    status: str
    points: int | None
    indexed_vectors: int | None


class RunCheck(BaseModel):
    scenario: str  # name
    dataset: DatasetCheck
    index: dict[str, Any]  # метаданные ragglet primary target
    params: dict[str, Any] | None  # параметры коллекции primary target
    targets: list[TargetCheck]
    plan: RunPlanOut
    warnings: list[ProblemIssue]


class BuildTargetCheck(BaseModel):
    id: str
    endpoint: str
    collection: str
    reachable: bool
    error: str | None = None
    exists: bool | None = None  # None — endpoint недоступен
    status: str | None = None
    points: int | None = None
    built_at: str | None = None  # из метаданных ragglet
    index_profile_id: str | None = None
    used_by: list[str] = []  # сценарии библиотеки


class BuildCheck(BaseModel):
    profile: str  # id профиля
    dataset: str  # label
    corpus_sha256: str
    documents: int
    targets: list[BuildTargetCheck]
    warnings: list[ProblemIssue]


class RunAccepted(BaseModel):  # 202, Location: /v1/runs/{id}
    id: str
    state: TaskState
    position: int | None
    warnings: list[ProblemIssue]


class BuildAccepted(BaseModel):  # 202, Location: /v1/builds/{id}
    id: str
    state: TaskState
    position: int | None
    warnings: list[ProblemIssue]


class TaskErrorOut(BaseModel):
    type: str
    message: str
    qid: str | None = None
    problem: Problem | None = None  # второй preflight при старте не прошёл

    @classmethod
    def build(cls, failure: TaskFailure | None) -> TaskErrorOut | None:
        if failure is None:
            return None
        problem = None
        if failure.title is not None:
            problem = Problem(title=failure.title, hint=failure.hint, issues=issues_out(failure.issues))
        return cls(type=failure.type, message=failure.message, qid=failure.qid, problem=problem)


Phase = Literal["preparing", "progress", "finishing"]


class _Live(BaseModel):
    """То, что у прогона и сборки склеивается из реестра и менеджера задач."""

    state: TaskState
    phase: Phase | None = None  # только при running
    position: int | None = None
    progress: ProgressOut | None = None  # у завершённой — на момент конца
    enqueued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: TaskErrorOut | None = None


def live_fields(record: RunRecord | BuildRecord, status: TaskStatus | None) -> dict[str, Any]:
    state = status.state if status else record.state
    progress = (status.progress if status else None) or record.final_progress
    phase: Phase | None = None
    if state is TaskState.RUNNING:
        if progress is None:
            phase = "preparing"  # preflight, загрузка моделей, индекс BM25
        elif progress.done < progress.total:
            phase = "progress"
        else:
            phase = "finishing"  # запись summary или метаданных
    return {
        "state": state,
        "phase": phase,
        "position": status.position if status else None,
        "progress": ProgressOut.build(progress),
        "enqueued_at": record.enqueued_at,
        "started_at": (status.started_at if status else None) or record.started_at,
        "finished_at": (status.finished_at if status else None) or record.finished_at,
        "error": TaskErrorOut.build((status.failure if status else None) or record.failure),
    }


class RunSummaryOut(BaseModel):
    repeats: list[dict[str, Any]]
    aggregate: dict[str, Any]


ScenarioFile = Literal["unchanged", "changed", "deleted"]


class RunStatus(_Live):
    id: str
    scenario: str  # name
    scenario_id: str | None  # ссылка на сценарий библиотеки, если её передали
    etag: str | None  # None — прогон импортирован из каталога: текста сценария не было
    origin: Origin  # import — каталог без записи реестра: CLI до ragglet-37 и старые прогоны
    scenario_file: ScenarioFile | None = None  # файл сценария библиотеки против etag; None — ссылки нет
    plan: RunPlanOut | None  # None — импортирован: плана preflight не было
    dataset: str
    collection: str
    tags: list[str]
    warnings: list[ProblemIssue]  # preflight при постановке
    summary: RunSummaryOut | None = None  # ?include=summary: summary.json и summary.aggregate.json как есть
    manifest: dict[str, Any] | None = None  # ?include=manifest
    scenario_yaml: str | None = None  # ?include=scenario: snapshot для «Повторить» и «Открыть как сценарий»


class BuildStatus(_Live):
    id: str
    profile: str  # id профиля
    profile_id: str | None
    etag: str
    dataset: str
    targets: list[CollectionRef]
    warnings: list[ProblemIssue]
    result: dict[str, Any] | None = None  # documents и записанные метаданные
    profile_yaml: str | None = None  # ?include=profile


class RunItem(BaseModel):
    """Строка списка прогонов: запись реестра и колонки сводки, без чтения per-query."""

    id: str
    scenario: str  # name
    scenario_id: str | None
    origin: Origin
    state: TaskState
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_s: float | None  # finished_at − started_at
    dataset: str
    collection: str
    tags: list[str]  # теги записи и сценария
    repeats: int | None  # из summary.aggregate
    ndcg: float | None  # doc_ndcg@k, среднее по повторам
    recall: float | None  # doc_recall@k
    p50_ms: float | None  # lat_total_p50
    p95_ms: float | None  # lat_total_p95
    qps: float | None  # throughput_qps
    timeout_rate: float | None


class RunList(BaseModel):
    k: int  # для ndcg и recall
    total: int
    items: list[RunItem]


class BuildItem(BaseModel):
    id: str
    profile: str  # id профиля
    profile_id: str | None
    state: TaskState
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_s: float | None
    dataset: str
    targets: list[CollectionRef]
    documents: int | None  # из result


class BuildList(BaseModel):
    total: int
    items: list[BuildItem]


class ArtifactOut(BaseModel):
    name: str
    size: int  # байт
    modified_at: datetime


class QueryPageOut(BaseModel):
    repeat: int
    repeats: list[int]  # какие повторы записаны: у упавшего прогона — частичные результаты
    source: str  # "per_query.repeat0.parquet"
    columns: list[str]  # колонки строк: у старых поколений формата их меньше
    total: int
    items: list[dict[str, Any]]  # скалярные колонки per-query и first_relevant_rank; NaN -> null


class TruthDoc(BaseModel):
    doc_id: str  # из truth_ids строки
    doc_rank: int | None  # ранг в final_doc_ids; None — не найден


class ResultChunk(BaseModel):
    rank: int  # позиция в final_ids, с 1
    chunk_id: str
    doc_id: str  # из chunk_id «<doc_id>#<chunk_index>» — контракт сборки
    doc_rank: int | None  # ранг документа в final_doc_ids: по нему считаются doc-метрики
    relevant: bool  # doc_id в truth_ids строки
    rerank_score: float | None  # по chunk_id из reranked_ids и rerank_scores
    in_context: bool | None  # None — context выключен


class QueryDetail(BaseModel):
    qid: str
    repeat: int
    query: str
    row: dict[str, Any]  # строка per-query целиком: задержки, метрики, контекст, ответ
    truth: list[TruthDoc]
    results: list[ResultChunk]  # пусто у прогонов без final_doc_ids


class LogChunkOut(BaseModel):
    text: str
    offset: int
    next_offset: int
    complete: bool


class RootInfo(BaseModel):
    api: str = "/v1"
    docs: str = "/v1/docs"

