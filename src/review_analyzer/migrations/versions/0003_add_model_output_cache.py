"""Cache sentiment and embedding model outputs.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sentiment_cache",
        sa.Column("model", sa.String(), primary_key=True),
        sa.Column("text_hash", sa.String(64), primary_key=True),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
    )
    op.create_table(
        "embedding_cache",
        sa.Column("model", sa.String(), primary_key=True),
        sa.Column("text_hash", sa.String(64), primary_key=True),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("embedding_cache")
    op.drop_table("sentiment_cache")
