# db.py
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Request
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# The engine is the thing that actually knows how to talk to the
# database file. "sqlite+aiosqlite:///" tells SQLAlchemy to use the
# aiosqlite driver under the hood (same driver the checkpointer uses,
# just now through SQLAlchemy instead of raw aiosqlite calls).
engine = create_async_engine("sqlite+aiosqlite:///app.db")

# A sessionmaker is a factory that hands out new Session objects.
# Think of it like Django's connection pool, but you explicitly ask
# it for a session each time you need to do DB work, rather than it
# being implicitly available on every model call.
async_session = async_sessionmaker(engine, expire_on_commit=False)


# Every model class inherits from this — same role as Django's
# models.Model base class.
class Base(DeclarativeBase):
    pass


# core/db.py — add this function to the existing file (keep everything else as-is)


async def get_db_session(request: Request) -> AsyncSession:
    """FastAPI dependency — hands a route a fresh AsyncSession scoped to
    just this request, using the sessionmaker created once at startup
    (see lifespan in main.py). Usage:
        async def my_route(session: AsyncSession = Depends(get_db_session)):
    """
    async with request.app.state.db_session() as session:
        yield session
