"""document chunking strategy

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-01 20:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Documents indexed before this migration were all chunked recursively.
    op.add_column(
        "documents",
        sa.Column("chunker", sa.String(length=32), server_default="recursive", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column("chunk_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "chunk_params")
    op.drop_column("documents", "chunker")
