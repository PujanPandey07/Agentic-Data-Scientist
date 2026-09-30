# core/migrations.py
import asyncio
import logging
from pathlib import Path
from alembic.config import Config
from alembic import command

logger = logging.getLogger(__name__)


def run_migrations_sync() -> None:
    """Run Alembic upgrade head synchronously."""
    try:
        # Locate alembic.ini relative to this file (Backend/app/core/migrations.py -> Backend/app/alembic.ini)
        ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
        if not ini_path.exists():
            ini_path = Path("alembic.ini")

        logger.info(f"Checking and applying database migrations with {ini_path}...")
        alembic_cfg = Config(str(ini_path))
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations applied successfully (head).")
    except Exception as exc:
        logger.error(f"Database migration failed: {exc}", exc_info=True)
        raise


async def apply_migrations_async() -> None:
    """Run migrations in an asynchronous thread during application startup."""
    await asyncio.to_thread(run_migrations_sync)
