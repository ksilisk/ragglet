#!/usr/bin/env bash
# Сквозная проверка стенда через REST API (ragglet-35): то же, что tests/test_e2e_api.py, но
# curl-ом и с выводом на экран — чтобы руками убедиться, что стенд жив целиком.
#
#   1. GET /v1/stand — Qdrant и Redis доступны.
#   2. Сборка индекса из configs/indexes/demo.yaml: dry-run, 202, опрос до completed, /v1/indexes.
#   3. Прогон baseline_qdrant: dry-run, 422 на испорченном конфиге, очередь из двух, отмена
#      ожидающего, фазы, summary/manifest/snapshot, список, per-query, деталь запроса, файлы, zip, лог.
#   4. Отмена идущего прогона (repeats: 5).
#   5. SIGKILL сервера во время прогона → после рестарта failed/ServerRestart, воркер не остался;
#      SIGTERM во время прогона → failed/ServerShutdown.
#   6. Прогон CLI мимо API → в списке как импортированный.
#   7. «Повторить»: snapshot содержимым и та же ссылка на сценарий.
#
# Нужны Qdrant и Redis (docker compose -f docker/docker-compose.yaml --profile qdrant up -d qdrant redis),
# uv, curl, jq и lsof. Запуск из корня репозитория:
#
#   scripts/smoke_api.sh
#
# Сервер скрипт поднимает сам (`uv run ragglet serve` на порту $PORT, по умолчанию 8100), прогоны и
# сборки пишет во временный каталог — results/ не трогается; каталог печатается в конце и остаётся.
# Коллекция chunks_v1 пересобирается из того же демо-корпуса — как `ragglet index build
# configs/indexes/demo.yaml`. Идёт около трёх минут: каждый воркер импортирует torch и грузит модель.
set -euo pipefail

PORT="${PORT:-8100}"
BASE="http://127.0.0.1:$PORT/v1"
TMP="${TMPDIR:-/tmp}"
WORK="$(mktemp -d "${TMP%/}/ragglet-smoke.XXXXXX")"
OUT="$WORK/last.json"  # тело последнего ответа
SCENARIO=configs/scenarios/baseline_qdrant.yaml
PROFILE=configs/indexes/demo.yaml
SERVER=""

for tool in uv curl jq lsof; do
  command -v "$tool" >/dev/null || { echo "нужен $tool"; exit 1; }
done
[[ -f $SCENARIO ]] || { echo "запускайте из корня репозитория"; exit 1; }
if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null; then
  echo "порт $PORT занят: PORT=<свободный> $0"; exit 1
fi

# --- помощники

step() { printf '\n== %s\n' "$*"; }
ok() { printf 'ok   %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; jq . "$OUT" 2>/dev/null || cat "$OUT"; exit 1; }

# req МЕТОД ПУТЬ [JSON] — тело ответа в $OUT, код в $CODE
req() {
  local args=(-sS -o "$OUT" -w '%{http_code}' -X "$1" "$BASE$2")
  [[ $# -ge 3 ]] && args+=(-H 'content-type: application/json' -d "$3")
  CODE=$(curl "${args[@]}")
}
expect() { [[ $CODE == "$1" ]] || fail "$2: HTTP $CODE, ожидался $1"; ok "$2"; }
field() { jq -r "$@" "$OUT"; }
check() { [[ $(field "$1") == "true" ]] || fail "$2"; ok "$2"; }

# run_body ФАЙЛ [ПОЛЯ] — тело POST /v1/runs: YAML сценария содержимым (как его шлёт UI) и поля
# запроса объектом jq, например '{dry_run: true}'
run_body() {
  local extra='{}'
  [[ $# -ge 2 ]] && extra=$2
  jq -n --rawfile s "$1" "{scenario: \$s} + $extra"
}

# poll ПУТЬ — опрос раз в 0,5 с до конца задачи; печатает смены состояния и фазы с секундами от начала
poll() {
  local start=$SECONDS last="" key
  while :; do
    req GET "$1"
    key=$(field '"\(.state) \(.phase // "")"')
    if [[ $key != "$last" ]]; then
      printf '     +%3ds %-20s %s\n' $((SECONDS - start)) "$key" "$(field 'if .progress then "\(.progress.done) / \(.progress.total)" else "" end')"
      last=$key
    fi
    case $(field .state) in queued|running) sleep 0.5 ;; *) return ;; esac
  done
}

# wait_done_above ПУТЬ N — ждать, пока прогресс задачи перевалит за N
wait_done_above() {
  until req GET "$1"; [[ $(field '.progress.done // 0') -gt $2 ]]; do
    [[ $(field .state) =~ ^(queued|running)$ ]] || fail "задача закончилась раньше, чем ожидалось"
    sleep 0.3
  done
}

start_server() {
  RAGGLET_PORT=$PORT RAGGLET_ARTIFACTS_DIR="$WORK/runs" RAGGLET_BUILDS_DIR="$WORK/builds" \
    uv run ragglet serve >>"$WORK/server.log" 2>&1 &
  local start=$SECONDS
  until SERVER=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN); do  # pid python-процесса, а не обёртки uv
    ((SECONDS - start < 120)) || { cat "$WORK/server.log"; echo "сервер не поднялся"; exit 1; }
    sleep 0.5
  done
  ok "сервер pid $SERVER слушает :$PORT за $((SECONDS - start)) с"
}

stop_server() {  # stop_server СИГНАЛ — и дождаться выхода
  kill "-$1" "$SERVER"
  while kill -0 "$SERVER" 2>/dev/null; do sleep 0.2; done
  SERVER=""
}

worker_pid() { head -1 "$WORK/runs/$1/task.log" | sed -E 's/.*pid ([0-9]+).*/\1/'; }

trap '[[ -n $SERVER ]] && kill -TERM "$SERVER" 2>/dev/null; echo; echo "каталог проверки: $WORK"' EXIT

start_server

step "1. Стенд"
req GET /stand; expect 200 "GET /v1/stand"
check '[.qdrant[].available] | all' "Qdrant доступен: $(field '.qdrant[0].version')"
check '.redis.available' "Redis доступен: $(field .redis.version)"
ok "compute: $(field .compute.device), commit: $(field .commit | cut -c1-12)"

step "2. Сборка индекса $PROFILE"
req POST /builds "$(jq -n --rawfile p $PROFILE '{profile: $p, dry_run: true}')"; expect 200 "dry-run сборки"
ok "цели: $(field '[.targets[] | "\(.collection) (\(.status // "нет"))"] | join(", ")'), документов: $(field .documents)"
req POST /builds "$(jq -n --rawfile p $PROFILE '{profile: $p, profile_id: "demo"}')"; expect 202 "сборка поставлена"
BUILD=$(field .id)
poll "/builds/$BUILD"
[[ $(field .state) == completed ]] || fail "сборка $BUILD"
ok "сборка $BUILD: $(field .result.documents) документов, $(field '.finished_at')"
req GET /indexes; expect 200 "GET /v1/indexes"
check '.collections[] | select(.name == "chunks_v1") | .state == "ready" and .ragglet.index_profile_id == "demo"' \
  "chunks_v1 готова, built_at $(field '.collections[] | select(.name == "chunks_v1") | .ragglet.built_at')"

step "3. Прогон $SCENARIO"
req POST /runs "$(run_body $SCENARIO '{dry_run: true}')"; expect 200 "dry-run прогона"
ok "запросов $(field .dataset.queries), план $(field .plan.total), индекс собран $(field .index.built_at)"
sed 's/k: \[1, 3, 5, 10\]/k: [50]/' $SCENARIO >"$WORK/bad.yaml"
req POST /runs "$(run_body "$WORK/bad.yaml" '{dry_run: true}')"; expect 422 "испорченный конфиг"
ok "проблема: $(field '.issues[0] | "\(.path): \(.message)"')"

req POST /runs "$(run_body $SCENARIO '{scenario_id: "baseline_qdrant"}')"; expect 202 "прогон A поставлен"
A=$(field .id); ok "A = $A, $(field .state)"
req POST /runs "$(run_body $SCENARIO '{scenario_id: "baseline_qdrant"}')"; expect 202 "прогон B поставлен"
B=$(field .id); [[ $(field .position) == 1 ]] || fail "B не первый в очереди"; ok "B = $B, позиция 1"
req GET /tasks; expect 200 "GET /v1/tasks"
check "map(.id) == [\"$A\", \"$B\"] and .[1].state == \"queued\"" "A текущая, B в очереди"
req POST "/runs/$B/cancel"; expect 200 "отмена B"
check '.state == "cancelled"' "B отменена в очереди"
poll "/runs/$A"
[[ $(field .state) == completed ]] || fail "прогон $A"
ok "A: $(field '"\(.started_at) → \(.finished_at)"')"
req GET "/runs/$A?include=summary,manifest,scenario"; expect 200 "include=summary,manifest,scenario"
ok "nDCG@10 $(field '.summary.aggregate["doc_ndcg@10_mean_mean_over_repeats"]'), p50 $(field '.summary.aggregate.lat_total_p50_mean_over_repeats') мс, QPS $(field '.summary.aggregate.throughput_qps_mean_over_repeats')"
ok "manifest: индекс $(field .manifest.index.profile_id) от $(field .manifest.index.built_at), commit $(field .manifest.env.commit | cut -c1-12)"
check '.scenario_yaml | startswith("version: 1")' "snapshot сценария"
req GET /runs; expect 200 "GET /v1/runs"
jq -r '.items[] | "     \(.id)  \(.state)  \(.origin)  ndcg=\(.ndcg)  p95=\(.p95_ms)  qps=\(.qps)"' "$OUT"
check "any(.items[]; .id == \"$B\" and .state == \"cancelled\")" "B в списке как cancelled"
req GET "/runs/$A/queries?outcome=ok&limit=3"; expect 200 "per-query, outcome=ok"
ok "повторы $(field -c .repeats), строк $(field .total), файл $(field .source)"
QID=$(field '.items[0].qid')
req GET "/runs/$A/queries/$QID"; expect 200 "деталь запроса $QID"
check '.grades.available and .texts.available' "оценки из датасета и тексты чанков из Qdrant"
ok "релевантных $(field '.truth | length'), первый в выдаче: $(field '.results[0] | "\(.doc_id) relevant=\(.relevant)"')"
req GET "/runs/$A/artifacts"; expect 200 "список файлов"
ok "$(field '[.[].name] | join(", ")')"
curl -sS -o "$WORK/summary.json" "$BASE/runs/$A/artifacts/summary.json"
cmp -s "$WORK/summary.json" "$WORK/runs/$A/summary.json" || fail "summary.json не совпал с файлом"
ok "summary.json скачан байт в байт"
curl -sS -o "$WORK/$A.zip" "$BASE/runs/$A/artifacts.zip"
ok "zip: $(unzip -Z1 "$WORK/$A.zip" | wc -l | tr -d ' ') файлов"
req GET "/runs/$A/log?offset=0"; expect 200 "лог"
check '.complete' "лог дочитан: $(field .next_offset) байт"
field .text | sed 's/^/     | /'

step "4. Отмена идущего прогона"
sed 's/repeats: 2/repeats: 5/' $SCENARIO >"$WORK/repeats5.yaml"
req POST /runs "$(run_body "$WORK/repeats5.yaml")"; expect 202 "прогон на 5 повторов"
R=$(field .id)
wait_done_above "/runs/$R" 100
ok "идёт: $(field '"\(.progress.done) / \(.progress.total)"')"
req POST "/runs/$R/cancel"; expect 200 "отмена: ответ $(field .state)"
poll "/runs/$R"
check '.state == "cancelled" and .progress.done < .progress.total' "отменён на $(field .progress.done) / $(field .progress.total)"
[[ -f $WORK/runs/$R/per_query.repeat0.csv && ! -f $WORK/runs/$R/summary.json ]] || fail "файлы отменённого прогона"
ok "per-query первого повтора на месте, summary нет: $(ls "$WORK/runs/$R" | tr '\n' ' ')"

step "5. Падение и остановка сервера во время прогона"
sed 's/repeats: 2/repeats: 30/' $SCENARIO >"$WORK/long.yaml"
req POST /runs "$(run_body "$WORK/long.yaml")"; expect 202 "длинный прогон"
K=$(field .id)
wait_done_above "/runs/$K" 0
W=$(worker_pid "$K")
stop_server KILL; ok "SIGKILL серверу, воркер pid $W"
start=$SECONDS
while kill -0 "$W" 2>/dev/null; do
  ((SECONDS - start < 20)) || fail "воркер $W пережил сервер"
  sleep 0.5
done
ok "воркер завершился сам через $((SECONDS - start)) с"
start_server
req GET "/runs/$K"; expect 200 "запись после рестарта"
check '.state == "failed" and .error.type == "ServerRestart"' "failed / ServerRestart, прогресс $(field '.progress.done // "—"')"
req GET "/runs/$K/log"; expect 200 "лог после рестарта"
field .text | tail -2 | sed 's/^/     | /'
req GET /runs; expect 200 "список после рестарта: $(field .total) записей"

req POST /runs "$(run_body "$WORK/long.yaml")"; expect 202 "ещё один длинный прогон"
T=$(field .id)
wait_done_above "/runs/$T" 0
stop_server TERM; ok "SIGTERM серверу"
start_server
req GET "/runs/$T"; expect 200 "запись после остановки"
check '.state == "failed" and .error.type == "ServerShutdown"' "failed / ServerShutdown на $(field .progress.done) / $(field .progress.total)"

step "6. Прогон CLI мимо API"
sed "s#artifacts_dir: \"results/runs\"#artifacts_dir: \"$WORK/runs\"#" $SCENARIO >"$WORK/cli.yaml"
uv run ragglet bench run "$WORK/cli.yaml" >"$WORK/cli.log" 2>&1 || { tail -20 "$WORK/cli.log"; fail "ragglet bench run"; }
CLI=$(basename "$(ls -d "$WORK"/runs/baseline_qdrant_* | while read -r d; do [[ -f $d/run.json ]] || echo "$d"; done | tail -1)")
req GET /runs; expect 200 "GET /v1/runs"
check "any(.items[]; .id == \"$CLI\" and .origin == \"import\" and .state == \"completed\" and .ndcg != null)" \
  "$CLI импортирован с колонками сводки"

step "7. Повторить"
req GET "/runs/$A?include=scenario"; expect 200 "snapshot прогона A"
jq '{scenario: .scenario_yaml, scenario_id: .scenario_id}' "$OUT" >"$WORK/repeat.json"
req POST /runs "$(cat "$WORK/repeat.json")"; expect 202 "повтор поставлен"
P=$(field .id)
poll "/runs/$P"
check '.state == "completed" and .scenario_id == "baseline_qdrant" and .scenario_file == "unchanged"' \
  "повтор $P: та же ссылка на сценарий, файл не изменился"

step "Готово: все пункты прошли"
