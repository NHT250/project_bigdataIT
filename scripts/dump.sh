#!/bin/sh
# Save the PostgreSQL data to a file: scripts/dump.sh [file]
set -eu

compose_file="${COMPOSE_FILE:-docker-compose.prod.yml}"
file="${1:-backup/techpulse.dump}"

mkdir -p "$(dirname "$file")"
docker compose -f "$compose_file" exec -T postgres pg_dump -U techpulse -d techpulse -Fc > "$file"
echo "Saved $file"
