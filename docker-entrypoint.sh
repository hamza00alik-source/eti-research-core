#!/bin/sh
set -eu
mkdir -p /app/data
if [ "$(id -u)" = "0" ]; then
  chown -R eti:eti /app/data
  exec gosu eti "$@"
fi
exec "$@"
