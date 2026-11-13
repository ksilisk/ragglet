#!/bin/sh
# Two independent vector sources for demo_sharded / s5_distributed_like.
set -eu
cd "$(dirname "$0")/.."
docker compose -f docker/docker-compose.yaml --profile qdrant-multi up -d --wait
export RAGGLET_QDRANT_ENDPOINTS=http://localhost:6333,http://localhost:6334
exec ./ragglet start --docker --no-services "$@"
