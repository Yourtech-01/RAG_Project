#!/usr/bin/env bash
# start.sh — runs on every boot of the Render web service.
# Render's free instance type has an ephemeral filesystem, so we re-ingest
# the small bundled demo doc set on every startup. This takes only a few
# seconds for the sample docs and keeps the demo self-contained (no manual
# ingestion step required after deploy).
set -e

echo "[start.sh] Ingesting demo documents..."
python pipeline.py --docs ./data/docs/ --reset

echo "[start.sh] Starting API on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
