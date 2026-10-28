"""Add model-predicted sentiment columns.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("text_sentiment", sa.String(), nullable=True))
    op.add_column("reviews", sa.Column("sentiment_score", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("reviews") as batch:
        batch.drop_column("sentiment_score")
        batch.drop_column("text_sentiment")
