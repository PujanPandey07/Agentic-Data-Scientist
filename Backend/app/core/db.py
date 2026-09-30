from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import DateTime
from fastapi import Request
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# db.py lives in Backend/app/core, so parents[2] is the Backend folder
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# Reads DATABASE_URL from the environment instead of hardcoding SQLite.
# Falls back to the old SQLite path only if DATABASE_URL isn't set.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///app.db")

engine = create_async_engine(DATABASE_URL)

async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    # Every Mapped[datetime] becomes TIMESTAMP WITH TIME ZONE, which matches
    # the timezone-aware datetime.now(timezone.utc) values used in models.py
    type_annotation_map = {
        datetime: DateTime(timezone=True),
    }


async def get_db_session(request: Request) -> AsyncSession:
    async with request.app.state.db_session() as session:
        yield session
