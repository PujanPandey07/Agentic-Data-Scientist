#!/usr/bin/env python
"""
One-time migration: move existing JSON memory files → Postgres.

Run ONCE after applying the Alembic migration:
    cd Backend/app
    ../.venv/Scripts/python migrate_memory_to_pg.py

Safe to re-run — uses upsert so duplicates won't cause errors.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make sure app modules are importable when run from Backend/app
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from cache.pg_store import PgLongTermStore, PgShortTermStore
from core.db import engine, Base

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent / "cache" / "data"
LTM_DIR = DATA_ROOT / "long_term"
STM_DIR = DATA_ROOT / "conversations"


async def migrate_long_term(store: PgLongTermStore) -> int:
    if not LTM_DIR.exists():
        logger.info("No long-term memory directory found — skipping.")
        return 0

    count = 0
    for path in LTM_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        for item in data.get("memories", []):
            try:
                await store.upsert(
                    memory_id=item["memory_id"],
                    owner_id=item["owner_id"],
                    topic=item["topic"],
                    content=item["content"],
                    importance=float(item.get("importance", 0.7)),
                )
                count += 1
            except Exception as exc:
                logger.warning("Skipping memory %s: %s", item.get("memory_id"), exc)

    return count


async def migrate_short_term(store: PgShortTermStore) -> int:
    if not STM_DIR.exists():
        logger.info("No short-term memory directory found — skipping.")
        return 0

    count = 0
    for path in STM_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        if "conversation_id" not in data:
            logger.warning("Skipping %s: missing conversation_id", path.name)
            continue

        try:
            await store.set(data)
            count += 1
        except Exception as exc:
            logger.warning("Skipping conversation %s: %s", data.get("conversation_id"), exc)

    return count


async def main() -> None:
    ltm_store = PgLongTermStore()
    stm_store = PgShortTermStore()

    logger.info("Migrating long-term memories from %s …", LTM_DIR)
    ltm_count = await migrate_long_term(ltm_store)
    logger.info("  → migrated %d long-term memory entries", ltm_count)

    logger.info("Migrating short-term memories from %s …", STM_DIR)
    stm_count = await migrate_short_term(stm_store)
    logger.info("  → migrated %d short-term memory conversations", stm_count)

    logger.info("Done. You can now delete the cache/data/ directory if you wish.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
