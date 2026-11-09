#!/bin/sh
# Replace the PostgreSQL data with a dump: scripts/restore.sh [file]
set -eu

compose_file="${COMPOSE_FILE:-docker-compose.prod.yml}"
file="${1:-backup/techpulse.dump}"

if [ ! -f "$file" ]; then
    echo "No dump at $file" >&2
    exit 1
fi

compose="docker compose -f $compose_file"
$compose up -d --wait postgres
# Nothing may write to the tables while they are being replaced.
$compose stop ingest api indexer
$compose exec -T postgres pg_restore -U techpulse -d techpulse --clean --if-exists --no-owner --exit-on-error < "$file"
$compose up -d
echo "Restored $file"
