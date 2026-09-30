"""add pipeline_family to conversations

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-28 22:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add pipeline_family column to conversations table."""
    op.add_column(
        "conversations",
        sa.Column(
            "pipeline_family",
            sa.String(),
            nullable=False,
            server_default="supervised",
        ),
    )


def downgrade() -> None:
    """Drop pipeline_family column from conversations table."""
    op.drop_column("conversations", "pipeline_family")
