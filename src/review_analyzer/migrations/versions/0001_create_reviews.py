"""Create reviews table.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reviews",
        sa.Column("review_id", sa.String(), primary_key=True),
        sa.Column("app_id", sa.String(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("app_version", sa.String(), nullable=True),
        sa.Column("thumbs_up", sa.Integer(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_reviews_app_created", "reviews", ["app_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_reviews_app_created", table_name="reviews")
    op.drop_table("reviews")
