#!/bin/sh
# Two independent vector sources for demo_sharded / s5_distributed_like.
set -eu
cd "$(dirname "$0")/.."
export COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yaml}${COMPOSE_PATH_SEPARATOR:-:}docker/compose.s5.yaml"
exec ./ragglet start --docker "$@"
