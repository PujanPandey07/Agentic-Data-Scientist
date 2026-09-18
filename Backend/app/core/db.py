import os
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Request
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Reads DATABASE_URL from the environment instead of hardcoding SQLite.
# Falls back to the old SQLite path only if DATABASE_URL isn't set, so
# nothing breaks if .env somehow isn't loaded yet.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///app.db")

engine = create_async_engine(DATABASE_URL)

async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db_session(request: Request) -> AsyncSession:
    async with request.app.state.db_session() as session:
        yield session
