#!/usr/bin/env bash
# Bring up the Day 19 stack with Apple's `container` (github.com/apple/container).
#
# Why this file exists: `container` runs OCI images in lightweight VMs on Apple
# silicon, but it has NO compose implementation. docker-compose.yml therefore
# does not apply, and the three services have to be started one at a time.
# This script is the hand-rolled equivalent of `docker compose up -d` and keeps
# the SAME image tags, ports, env and volume names as docker-compose.yml, so
# the lab's .env and Feast config work unchanged either way.
#
# Requires: Apple silicon + macOS 26+.   Run: make container-up

set -euo pipefail

command -v container >/dev/null 2>&1 || {
  echo "[container] Apple 'container' not found."
  echo "[container] Install: https://github.com/apple/container   (Apple silicon, macOS 26+)"
  exit 1
}

if ! container system status 2>/dev/null | awk '/^status/{exit ($2=="running")?0:1}'; then
  echo "[container] starting the container system service…"
  container system start
fi

# Same names/tags as docker-compose.yml -- keep them in sync.
QDRANT_IMAGE="qdrant/qdrant:v1.19.0"   # must track qdrant-client (>=1.12,<2.0)
REDIS_IMAGE="redis:7-alpine"
POSTGRES_IMAGE="postgres:16-alpine"

start() {                       # start <name> <image> <extra args...>
  local name=$1 image=$2; shift 2
  if container ls --all 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$name"; then
    echo "[container] $name exists — restarting"
    container stop "$name" >/dev/null 2>&1 || true
    container rm "$name"   >/dev/null 2>&1 || true
  fi
  echo "[container] starting $name ($image)"
  container run -d --name "$name" "$@" "$image"
}

for v in day19-qdrant-storage day19-redis-data day19-postgres-data; do
  container volume create "$v" >/dev/null 2>&1 || true
done

start day19-qdrant "$QDRANT_IMAGE" \
  -p 6333:6333 -p 6334:6334 \
  -v day19-qdrant-storage:/qdrant/storage

start day19-redis "$REDIS_IMAGE" \
  -p 6379:6379 \
  -v day19-redis-data:/data

# PGDATA points at a SUBDIRECTORY of the mount, not the mount itself.
# Apple `container` presents a named volume as a formatted filesystem, so the
# mount point already contains lost+found; postgres' initdb refuses to
# initialise a non-empty data directory and the container exits. Docker's
# volumes are empty, which is why docker-compose.yml gets away without this.
start day19-postgres "$POSTGRES_IMAGE" \
  -p 5433:5432 \
  -e POSTGRES_USER=feast -e POSTGRES_PASSWORD=feast -e POSTGRES_DB=feast_offline \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v day19-postgres-data:/var/lib/postgresql/data

# `container` has no healthcheck concept, so poll ourselves. Two things are
# checked, because either alone gives false positives: the port must accept a
# connection AND the container must still be running (postgres in particular
# listens briefly during initdb and then exits if the data dir is unusable).
echo "[container] waiting for services (up to 90s)…"
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }
still_running() {
  container ls 2>/dev/null | awk -v n="$1" 'NR>1 && $1==n {found=1} END{exit found?0:1}'
}
ALL_UP=0
for _ in $(seq 1 90); do
  if port_open 6333 && port_open 6379 && port_open 5433 \
     && still_running day19-qdrant && still_running day19-redis \
     && still_running day19-postgres; then
    ALL_UP=1
    echo "[container] all three services are up and accepting connections"
    break
  fi
  sleep 1
done

if [ "$ALL_UP" != "1" ]; then
  echo
  echo "[container] TIMED OUT. Current state:"
  container ls --all
  for c in day19-qdrant day19-redis day19-postgres; do
    still_running "$c" || { echo "--- $c logs (tail) ---"; container logs "$c" 2>&1 | tail -15; }
  done
  exit 1
fi

echo
container ls
echo
echo "[container] Qdrant  http://localhost:6333/dashboard"
echo "[container] Redis   localhost:6379"
echo "[container] Postgres localhost:5433  (feast/feast, db feast_offline)"
echo "[container] next: bash setup-docker.sh    # venv + deps + seed"
echo "[container] stop:  make container-down"
