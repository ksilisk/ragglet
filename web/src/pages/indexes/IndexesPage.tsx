import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { ApiProblem, type S } from "../../api/client";
import { useIndexes, useProfiles, useStand } from "../../api/queries";
import { msg, rich, useT } from "../../i18n/i18n";
import { fmtDataset, fmtNumber, fmtWhen } from "../../lib/format";
import { compactNames } from "../../lib/names";
import { Button, LinkButton } from "../../ui/Button";
import { Chip, type ChipTone } from "../../ui/Chip";
import { Dialog, useConfirm } from "../../ui/Dialog";
import { Empty, Page, PageHead } from "../../ui/Layout";
import { Menu } from "../../ui/Menu";
import { ProblemBox, ErrorBox } from "../../ui/ProblemView";
import { Tabs } from "../../ui/Tabs";
import { StateChip } from "../../ui/states";
import k from "../../ui/kit.module.css";
import { ImportDialog, NameDialog } from "../library/LibraryDialogs";
import { LIBRARY, createDocument, deleteDocument, duplicate, exportYaml, invalidateLibrary, skeleton } from "../library/library";
import s from "../scenarios/scenarios.module.css";

type Tab = "collections" | "profiles";

/** Подписи и подсказки — msg(): переводятся при выводе, t(COLLECTION_STATE[s].label). */
export const COLLECTION_STATE: Record<S["CollectionItem"]["state"], { label: string; tone: ChipTone; hint: string }> = {
  ready: { label: msg("готова"), tone: "ok", hint: msg("метаданные полны, status green") },
  optimizing: { label: msg("оптимизируется"), tone: "warn", hint: msg("не green, прогон даст предупреждение index.not_green") },
  incomplete: {
    label: msg("недособрана"),
    tone: "fail",
    hint: msg("коллекция есть, метаданных ragglet нет: сборка не завершилась или коллекция старого контракта"),
  },
  foreign: { label: msg("чужая"), tone: "cancel", hint: msg("без ключа ragglet") },
};

const short = (value: unknown, n = 8) => (typeof value === "string" && value ? value.slice(0, n) : null);
export const hostOf = (endpoint: string) => endpoint.replace(/^https?:\/\//, "").replace(/\/$/, "");

type Dialogs = { kind: "new" } | { kind: "import" } | { kind: "duplicate"; id: string } | { kind: "meta"; item: S["CollectionItem"] } | null;

export default function IndexesPage() {
  const t = useT();
  const [params, setParams] = useSearchParams();
  const tab: Tab = params.get("tab") === "profiles" ? "profiles" : "collections";
  const indexes = useIndexes();
  const profiles = useProfiles();
  const stand = useStand();
  const navigate = useNavigate();
  const [dialog, setDialog] = useState<Dialogs>(null);
  const profileIds = useMemo(() => new Set((profiles.data ?? []).map((p) => p.id)), [profiles.data]);

  return (
    <Page>
      <PageHead
        title={t("Индексы")}
        actions={
          <>
            <Menu
              trigger={<Button>{t("Новый профиль ▾")}</Button>}
              items={[
                { label: t("Пустой…"), onSelect: () => setDialog({ kind: "new" }) },
                { label: t("Из YAML…"), onSelect: () => setDialog({ kind: "import" }) },
              ]}
            />
            <Menu
              trigger={<Button variant="primary">{t("Собрать индекс ▾")}</Button>}
              items={(profiles.data ?? []).map((p) => ({
                label: <span className="mono">{p.id}</span>,
                onSelect: () => navigate(`${LIBRARY.profile.route(p.id)}?build=1`),
              }))}
            />
          </>
        }
      />
      <Tabs<Tab>
        label={t("Индексы")}
        value={tab}
        onChange={(next) => setParams(next === "collections" ? {} : { tab: next }, { replace: true })}
        items={[
          { id: "collections", label: t("Коллекции") + (indexes.data ? ` · ${indexes.data.collections.length}` : "") },
          { id: "profiles", label: t("Профили") + (profiles.data ? ` · ${profiles.data.length}` : "") },
        ]}
      />
      {tab === "collections" ? (
        <Collections data={indexes.data} error={indexes.error} profileIds={profileIds} onMeta={(item) => setDialog({ kind: "meta", item })} />
      ) : (
        <Profiles data={profiles.data} error={profiles.error} onDuplicate={(id) => setDialog({ kind: "duplicate", id })} />
      )}

      {dialog?.kind === "new" && (
        <NameDialog
          kind="profile"
          open
          onClose={() => setDialog(null)}
          title={t("Новый профиль индекса")}
          initial="new_profile"
          confirm={t("Создать")}
          taken={profileIds}
          onSubmit={(name) => createDocument("profile", skeleton("profile", name, stand.data?.qdrant[0]?.url))}
        />
      )}
      {dialog?.kind === "duplicate" && (
        <NameDialog
          kind="profile"
          open
          onClose={() => setDialog(null)}
          title={t("Дублировать {id}", { id: dialog.id })}
          initial={`${dialog.id}_copy`}
          confirm={t("Дублировать")}
          taken={profileIds}
          onSubmit={(name) => duplicate("profile", dialog.id, name)}
        />
      )}
      <ImportDialog kind="profile" open={dialog?.kind === "import"} onClose={() => setDialog(null)} />
      <Dialog
        open={dialog?.kind === "meta"}
        onOpenChange={(open) => !open && setDialog(null)}
        wide
        title={dialog?.kind === "meta" ? t("Метаданные {name}", { name: dialog.item.name }) : ""}
        footer={<Button onClick={() => setDialog(null)}>{t("Закрыть")}</Button>}
      >
        {dialog?.kind === "meta" && (
          <pre className="mono small" style={{ margin: 0, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
            {JSON.stringify({ ragglet: dialog.item.ragglet, params: dialog.item.params, status: dialog.item.status, points: dialog.item.points, indexed_vectors: dialog.item.indexed_vectors, segments: dialog.item.segments }, null, 2)}
          </pre>
        )}
      </Dialog>
    </Page>
  );
}

function Collections({ data, error, profileIds, onMeta }: {
  data: S["IndexList"] | undefined;
  error: unknown;
  profileIds: ReadonlySet<string>;
  onMeta: (item: S["CollectionItem"]) => void;
}) {
  const t = useT();
  const down = data?.endpoints.filter((e) => !e.available) ?? [];
  return (
    <>
      {error !== null && error !== undefined && <ErrorBox error={error} />}
      {down.length > 0 && (
        <ProblemBox
          warning
          problem={{
            title: t("Часть Qdrant недоступна — их коллекций нет в списке"),
            hint: null,
            issues: down.map((e) => ({ code: "storage.unreachable", severity: "warning", path: null, where: e.url, message: e.error ?? t("недоступен") })),
          }}
        />
      )}
      <section className={`${k.card} ${k.flush}`}>
        <table className={`${k.tbl} ${s.tight}`} aria-label={t("Коллекции Qdrant")}>
          <thead>
            <tr>
              <th>{t("Коллекция")}</th>
              <th>Endpoint</th>
              <th>{t("Состояние")}</th>
              <th>{t("Датасет")}</th>
              <th>{t("Модель")}</th>
              <th>Chunking</th>
              <th className="r">Points</th>
              <th>{t("Собрана")}</th>
              <th>{t("Профиль")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data?.collections.map((c) => {
              const meta = c.ragglet ?? {};
              const state = COLLECTION_STATE[c.state];
              const profile = (meta.index_profile_id as string | undefined) ?? c.profiles[0];
              const chunking = meta.chunking_method
                ? `${String(meta.chunking_method)}${meta.chunk_size ? ` ${String(meta.chunk_size)}/${String(meta.chunk_overlap)}` : ""}`
                : null;
              return (
                <tr key={`${c.endpoint}/${c.name}`}>
                  <td className="mono nowrap">{c.name}</td>
                  <td className="mono muted nowrap">{hostOf(c.endpoint)}</td>
                  <td>
                    <Chip tone={state.tone} title={`${t(state.hint)} · status ${c.status}`}>
                      {t(state.label)}
                    </Chip>
                  </td>
                  <td className="nowrap">
                    {c.ragglet ? (
                      <>
                        {String(meta.dataset_source)} · {String(meta.dataset_id)} <span className="mono muted">{short(meta.corpus_sha256)}</span>
                      </>
                    ) : (
                      <span className="muted">{t("нет метаданных ragglet")}</span>
                    )}
                  </td>
                  <td className="nowrap">
                    {meta.embedding_model ? (
                      <>
                        {String(meta.embedding_model)} {short(meta.embedding_revision) && <span className="mono muted">@{short(meta.embedding_revision)}</span>}
                      </>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td className="nowrap">{chunking ?? <span className="muted">—</span>}</td>
                  <td className="r num">{fmtNumber(c.points)}</td>
                  <td className="muted nowrap">{fmtWhen(meta.built_at as string | undefined)}</td>
                  <td className="mono nowrap">
                    {profile ? (
                      profileIds.has(profile) ? (
                        <Link to={LIBRARY.profile.route(profile)}>{profile}</Link>
                      ) : (
                        <span className="muted">{profile}</span>
                      )
                    ) : (
                      <span className="muted">—</span>
                    )}
                    {!c.ragglet && profile && <span className="muted">?</span>}
                  </td>
                  <td>
                    <div className={s.acts}>
                      {profile && profileIds.has(profile) ? (
                        <LinkButton small to={`${LIBRARY.profile.route(profile)}?build=1`}>
                          {t("Пересобрать")}
                        </LinkButton>
                      ) : (
                        <Button small disabled title={t("Профиля этой коллекции нет в библиотеке")}>
                          {t("Пересобрать")}
                        </Button>
                      )}
                      {(c.ragglet || c.params) && (
                        <Button small onClick={() => onMeta(c)}>
                          {t("Метаданные")}
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {data && data.collections.length === 0 && <Empty>{t("Коллекций нет: соберите индекс по профилю.")}</Empty>}
      </section>
      <span className="small muted">
        {t("Состояния:")}{" "}
        {Object.values(COLLECTION_STATE).map((st, i) => (
          <span key={st.label}>
            {i > 0 && " · "}
            <b>{t(st.label)}</b> — {t(st.hint)}
          </span>
        ))}
        .
      </span>
    </>
  );
}

function Profiles({ data, error, onDuplicate }: { data: S["IndexProfileItem"][] | undefined; error: unknown; onDuplicate: (id: string) => void }) {
  const t = useT();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const navigate = useNavigate();

  const onDelete = async (item: S["IndexProfileItem"]) => {
    const ok = await confirm({
      title: t("Удалить профиль {id}?", { id: item.id }),
      body: <>{rich(t("Файл {file} будет удалён. Коллекции в Qdrant останутся."), { file: <span className="mono">configs/indexes/{item.id}.yaml</span> })}</>,
      confirm: t("Удалить"),
      danger: true,
    });
    if (!ok) return;
    try {
      await deleteDocument("profile", item.id, item.etag);
    } catch (e) {
      if (!(e instanceof ApiProblem && e.status === 409)) throw e;
    } finally {
      invalidateLibrary(qc, "profile");
    }
  };

  return (
    <>
      {error !== null && error !== undefined && <ErrorBox error={error} />}
      <section className={`${k.card} ${k.flush}`}>
        <table className={`${k.tbl} ${s.tight}`} aria-label={t("Профили индексов")}>
          <thead>
            <tr>
              <th>{t("Профиль")}</th>
              <th>{t("Датасет")}</th>
              <th>Chunking</th>
              <th>{t("Модель")}</th>
              <th>{t("Цели")}</th>
              <th>{t("Последняя сборка")}</th>
              <th>{t("Изменён")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data?.map((p) => {
              const href = LIBRARY.profile.route(p.id);
              return (
                <tr key={p.id}>
                  <td>
                    <Link className={`${k.rowLink} mono`} to={href}>
                      {p.id}
                    </Link>{" "}
                    {!p.valid && <Chip tone="warn">{t("черновик")}</Chip>}
                  </td>
                  <td className="nowrap">{p.dataset ? fmtDataset(p.dataset) : <span className="muted">—</span>}</td>
                  <td className="nowrap">{p.chunking ?? <span className="muted">—</span>}</td>
                  <td className="nowrap">{p.embedding_model ?? <span className="muted">—</span>}</td>
                  <td className="mono">{compactNames(p.targets.map((x) => x.collection)) || <span className="muted">—</span>}</td>
                  <td className="nowrap">
                    {p.last_build ? (
                      <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                        <StateChip state={p.last_build.state} />
                        <Link className="small" to={`/builds/${encodeURIComponent(p.last_build.id)}`}>
                          {fmtWhen(p.last_build.enqueued_at)}
                        </Link>
                      </span>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td className="muted nowrap">{fmtWhen(p.updated_at)}</td>
                  <td>
                    <div className={s.acts}>
                      <LinkButton small to={href}>
                        {t("Открыть")}
                      </LinkButton>
                      <Button small onClick={() => onDuplicate(p.id)}>
                        {t("Дублировать")}
                      </Button>
                      <Menu
                        trigger={
                          <Button small aria-label={t("Ещё: {id}", { id: p.id })}>
                            ⋯
                          </Button>
                        }
                        items={[
                          { label: t("Собрать…"), onSelect: () => navigate(`${href}?build=1`) },
                          { label: t("Экспорт YAML"), onSelect: () => void exportYaml("profile", p.id) },
                          { label: t("Удалить…"), onSelect: () => void onDelete(p), danger: true },
                        ]}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {data && data.length === 0 && <Empty>{t("Профилей нет.")}</Empty>}
      </section>
      <span className="small muted">
        {rich(t("Библиотека профилей — каталог {dir} на сервере."), { dir: <span className="mono">configs/indexes/</span> })}
      </span>
    </>
  );
}
