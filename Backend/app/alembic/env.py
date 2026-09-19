from core.db import Base
import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Load Backend/.env so DATABASE_URL_PSYCOPG is available below — Alembic
# runs as a standalone script, so nothing else in the app has loaded it yet.
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the placeholder sqlalchemy.url from alembic.ini with the real
# connection string from the environment. We use DATABASE_URL_PSYCOPG (not
# DATABASE_URL) because Alembic's default tooling runs synchronously, and
# psycopg works fine sync — no need to fight asyncpg's async-only interface
# for a one-off migration script.
config.set_main_option("sqlalchemy.url", os.getenv("ALEMBIC_DATABASE_URL"))

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import your actual models so autogenerate can compare them against the
# real database and detect new/changed columns and tables.
import core.models  # noqa: F401 — imported for its side effect of
# registering all model classes onto Base.metadata; without this import,
# Base.metadata would be empty even though Base itself is imported above.

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
