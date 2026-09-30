#!/bin/sh
set -e

# Run database migrations before starting the application
echo "==> Applying database migrations with Alembic..."
alembic upgrade head

# Execute the container's CMD
exec "$@"
