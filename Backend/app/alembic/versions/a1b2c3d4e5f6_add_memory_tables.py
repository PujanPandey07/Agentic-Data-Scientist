"""add memory tables

Revision ID: a1b2c3d4e5f6
Revises: 4df9f4237e1d
Create Date: 2026-09-28 21:19:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "4df9f4237e1d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create long_term_memories and short_term_memories tables."""

    op.create_table(
        "long_term_memories",
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("owner_id", sa.String(), nullable=False),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("memory_id"),
        sa.UniqueConstraint("owner_id", "topic", name="uq_ltm_owner_topic"),
    )
    op.create_index(
        "ix_long_term_memories_owner_id",
        "long_term_memories",
        ["owner_id"],
        unique=False,
    )

    op.create_table(
        "short_term_memories",
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("conversation_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("recent_messages", JSONB(), nullable=False, server_default="[]"),
        sa.Column("current_goal", sa.Text(), nullable=True),
        sa.Column("recent_decisions", JSONB(), nullable=False, server_default="[]"),
        sa.Column("unresolved_questions", JSONB(), nullable=False, server_default="[]"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
    )


def downgrade() -> None:
    """Drop memory tables."""
    op.drop_table("short_term_memories")
    op.drop_index("ix_long_term_memories_owner_id", table_name="long_term_memories")
    op.drop_table("long_term_memories")
