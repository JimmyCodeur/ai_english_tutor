#!/bin/bash
set -e

host="$1"
shift
cmd="$@"

until pg_isready -h "$host" -p "$POSTGRES_PORT" -U "$POSTGRES_USER"; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing command"
exec $cmd