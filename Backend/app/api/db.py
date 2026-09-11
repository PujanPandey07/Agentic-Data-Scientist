# db.py
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
